import { Encoding, slice, TruncationStrategy } from "tokenizers";

import { Model, ModelType } from "./models/model";
import { initModel } from "./models/model.factory";
import { DEFAULT_MODEL_PATH, QAOptions } from "./qa-options";
import { BertTokenizer, RobertaTokenizer, Tokenizer } from "./tokenizers";

interface Feature {
  contextStartIndex: number;
  encoding: Encoding;
  maxContextMap: ReadonlyMap<number, boolean>;
}

interface Span {
  length: number;
  startIndex: number;
}

export interface Answer {
  /**
   * Only provided if `timeIt` option was true when creating the QAClient
   */
  inferenceTime?: number;
  /**
   * Only provided if answer found
   */
  score?: number;
  /**
   * Only provided if answer found
   */
  text?: string;
  /**
   * Only provided if `timeIt` option was true when creating the QAClient
   */
  totalTime?: number;
}

export class QAClient {
  private constructor(
    private readonly model: Model,
    private readonly tokenizer: Tokenizer,
    private readonly timeIt?: boolean
  ) {}

  static async fromOptions(options?: QAOptions): Promise<QAClient> {
    const model =
      options?.model ?? (await initModel({ path: DEFAULT_MODEL_PATH, cased: true }));

    let tokenizer = options?.tokenizer;
    if (!tokenizer) {
      const tokenizerOptions = {
        modelPath: model.path,
        modelType: model.type,
        mergesPath: options?.mergesPath,
        vocabPath: options?.vocabPath,
        lowercase: !model.cased
      };

      switch (model.type) {
        case ModelType.Roberta:
          tokenizer = await RobertaTokenizer.fromOptions(tokenizerOptions);
          break;

        default:
          tokenizer = await BertTokenizer.fromOptions(tokenizerOptions);
          break;
      }
    }

    return new QAClient(model, tokenizer, options?.timeIt);
  }

  async predict(
    question: string,
    context: string,
    maxAnswerLength = 15
  ): Promise<Answer> {
    const totalStartTime = Date.now();
    const features = await this.getFeatures(question, context);

    const inferenceStartTime = Date.now();
    const [startLogits, endLogits] = await this.model.runInference(
      features.map(f => f.encoding)
    );
    const elapsedInferenceTime = Date.now() - inferenceStartTime;

    const answer = this.getAnswer(
      context,
      features,
      startLogits,
      endLogits,
      maxAnswerLength
    );

    const totalElapsedTime = Date.now() - totalStartTime;
    if (this.timeIt) {
      return {
        ...answer,
        inferenceTime: elapsedInferenceTime,
        totalTime: totalElapsedTime
      };
    } else {
      return { ...answer };
    }
  }

  private async getFeatures(
    question: string,
    context: string,
    stride = 128
  ): Promise<Feature[]> {
    this.tokenizer.setPadding(this.model.inputLength);
    this.tokenizer.setTruncation(this.model.inputLength, {
      strategy: TruncationStrategy.OnlySecond,
      stride
    });

    const encoding = await this.tokenizer.encode(question, context);
    const encodings = [encoding, ...encoding.overflowing];

    const spans: Span[] = encodings.map((e, i) => ({
      startIndex: (this.model.inputLength - stride) * i,
      length: this.model.inputLength
    }));

    const contextStartIndex = this.tokenizer.getContextStartIndex(encoding);
    return spans.map<Feature>((s, i) => {
      const maxContextMap = getMaxContextMap(spans, i, contextStartIndex);

      return {
        contextStartIndex,
        encoding: encodings[i],
        maxContextMap: maxContextMap
      };
    });
  }

  private getAnswer(
    context: string,
    features: Feature[],
    startLogits: number[][],
    endLogits: number[][],
    maxAnswerLength: number
  ): Answer | null {
    const answers: {
      feature: Feature;
      score: number;
      startIndex: number;
      endIndex: number;
      startLogits: number[];
      endLogits: number[];
    }[] = [];

    for (let i = 0; i < features.length; i++) {
      const feature = features[i];
      const starts = startLogits[i];
      const ends = endLogits[i];

      const contextLastIndex = this.tokenizer.getContextEndIndex(feature.encoding);
      const [filteredStartLogits, filteredEndLogits] = [starts, ends].map(logits =>
        logits
          .slice(feature.contextStartIndex, contextLastIndex)
          .map<[number, number]>((val, i) => [i + feature.contextStartIndex, val])
      );

      filteredEndLogits.sort((a, b) => b[1] - a[1]);
      for (const startLogit of filteredStartLogits) {
        filteredEndLogits.some(endLogit => {
          if (endLogit[0] < startLogit[0]) {
            return;
          }

          if (endLogit[0] - startLogit[0] + 1 > maxAnswerLength) {
            return;
          }

          if (!feature.maxContextMap.get(startLogit[0])) {
            return;
          }

          answers.push({
            feature,
            startIndex: startLogit[0],
            endIndex: endLogit[0],
            score: startLogit[1] + endLogit[1],
            startLogits: starts,
            endLogits: ends
          });

          return true;
        });
      }
    }

    if (!answers.length) {
      return null;
    }

    const answer = answers.sort((a, b) => b.score - a.score)[0];
    const offsets = answer.feature.encoding.offsets;

    const answerText = slice(
      context,
      offsets[answer.startIndex][0],
      offsets[answer.endIndex][1]
    );

    const startProbs = softMax(answer.startLogits);
    const endProbs = softMax(answer.endLogits);
    const probScore = startProbs[answer.startIndex] * endProbs[answer.endIndex];

    return {
      text: answerText.trim(),
      score: Math.round((probScore + Number.EPSILON) * 100) / 100
    };
  }
}

function getMaxContextMap(
  spans: Span[],
  spanIndex: number,
  contextStartIndex: number
): Map<number, boolean> {
  const map = new Map<number, boolean>();
  const spanLength = spans[spanIndex].length;

  let i = 0;
  while (i < spanLength) {
    const position = spans[spanIndex].startIndex + i;
    let bestScore = -1;
    let bestIndex = -1;

    for (const [ispan, span] of spans.entries()) {
      const spanEndIndex = span.startIndex + span.length - 1;
      if (position < span.startIndex || position > spanEndIndex) {
        continue;
      }

      const leftContext = position - span.startIndex;
      const rightContext = spanEndIndex - position;
      const score = Math.min(leftContext, rightContext) + 0.01 * span.length;
      if (score > bestScore) {
        bestScore = score;
        bestIndex = ispan;
      }
    }

    map.set(contextStartIndex + i, bestIndex === spanIndex);
    i++;
  }

  return map;
}

function softMax(values: number[]): number[] {
  const max = Math.max(...values);
  const exps = values.map(x => Math.exp(x - max));
  const expsSum = exps.reduce((a, b) => a + b);
  return exps.map(e => e / expsSum);
}
