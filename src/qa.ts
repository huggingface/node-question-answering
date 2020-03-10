import { exists as fsExists } from "fs";
import path from "path";
import { BertWordPieceTokenizer, Encoding, TruncationStrategy } from "tokenizers";
import { promisify } from "util";

import { Model } from "./models/model";
import { SavedModel } from "./models/saved-model.model";
import {
  DEFAULT_ASSETS_PATH,
  DEFAULT_MODEL_PATH,
  DEFAULT_VOCAB_PATH,
  QAOptions
} from "./qa-options";

interface Feature {
  contextLength: number;
  contextStartIndex: number;
  encoding: Encoding;
  maxContextMap: ReadonlyMap<number, boolean>;
}

interface Span {
  contextLength: number;
  contextStartIndex: number;
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
    private readonly tokenizer: BertWordPieceTokenizer,
    private readonly timeIt?: boolean
  ) {}

  static async fromOptions(options?: QAOptions): Promise<QAClient> {
    const model =
      options?.model ??
      (await SavedModel.fromOptions({ path: DEFAULT_MODEL_PATH, cased: true }));

    let tokenizer: BertWordPieceTokenizer;
    if (options?.tokenizer) {
      tokenizer = options.tokenizer;
    } else {
      let vocabPath = options?.vocabPath;
      if (!vocabPath) {
        if (options?.model?.params.path) {
          const existsAsync = promisify(fsExists);
          const fullPath = (await existsAsync(options.model.params.path))
            ? path.join(options.model.params.path, "vocab.txt")
            : path.join(DEFAULT_ASSETS_PATH, options.model.params.path, "vocab.txt");

          if (await existsAsync(fullPath)) {
            vocabPath = fullPath;
          }
        }

        vocabPath = vocabPath ?? DEFAULT_VOCAB_PATH;
      }

      tokenizer = await BertWordPieceTokenizer.fromOptions({
        vocabFile: vocabPath,
        lowercase: !model.params.cased
      });
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
      features.map(f => f.encoding.ids),
      features.map(f => f.encoding.attentionMask)
    );
    const elapsedInferenceTime = Date.now() - inferenceStartTime;

    const answer = this.getAnswer(features, startLogits, endLogits, maxAnswerLength);
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
    this.tokenizer.setPadding({ maxLength: this.model.params.shape[1] });
    this.tokenizer.setTruncation(this.model.params.shape[1], {
      strategy: TruncationStrategy.OnlySecond,
      stride
    });

    const encoding = await this.tokenizer.encode(question, context);
    const encodings = [encoding, ...encoding.overflowing];

    const questionLength =
      encoding.tokens.indexOf(this.tokenizer.configuration.sepToken) - 1; // Take [CLS] into account
    const questionLengthWithTokens = questionLength + 2;

    const spans: Span[] = encodings.map((e, i) => {
      const nbAddedTokens = e.specialTokensMask.reduce((acc, val) => acc + val, 0);
      const actualLength = e.length - nbAddedTokens;

      return {
        startIndex: i * stride,
        contextStartIndex: questionLengthWithTokens,
        contextLength: actualLength - questionLength,
        length: actualLength
      };
    });

    return spans.map<Feature>((s, i) => {
      const maxContextMap = getMaxContextMap(spans, i, stride, questionLengthWithTokens);

      return {
        contextLength: s.contextLength,
        contextStartIndex: s.contextStartIndex,
        encoding: encodings[i],
        maxContextMap: maxContextMap
      };
    });
  }

  private getAnswer(
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

      const contextLastIndex = feature.contextStartIndex + feature.contextLength - 1;
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
    const answerText = features[0].encoding.getOriginalString(
      offsets[answer.startIndex][0],
      offsets[answer.endIndex][1]
    );

    const startProbs = softMax(answer.startLogits);
    const endProbs = softMax(answer.endLogits);
    const probScore = startProbs[answer.startIndex] * endProbs[answer.endIndex];

    return {
      text: answerText,
      score: Math.round((probScore + Number.EPSILON) * 100) / 100
    };
  }
}

function getMaxContextMap(
  spans: Span[],
  spanIndex: number,
  stride: number,
  questionLengthWithTokens: number
): Map<number, boolean> {
  const map = new Map<number, boolean>();
  const spanLength = spans[spanIndex].length;

  let i = 0;
  while (i < spanLength) {
    const position = spanIndex * stride + i;
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

    map.set(questionLengthWithTokens + i, bestIndex === spanIndex);
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
