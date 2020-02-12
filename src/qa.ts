import path from "path";
import { BertWordPieceTokenizer, Encoding, TruncationStrategy } from "tokenizers";

import { LocalModel } from "./local.model";
import { Model } from "./model";
import { ModelOptions, QAOptions } from "./qa-options";
import { RemoteModel } from "./remote.model";

const DEFAULT_ASSETS_PATH = path.join(process.cwd(), "./.models");
const DEFAULT_MODEL_PATH = path.join(DEFAULT_ASSETS_PATH, "distilbert-cased");
const DEFAULT_VOCAB_PATH = path.join(DEFAULT_MODEL_PATH, "vocab.txt");

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

interface Answer {
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
    private readonly tokenizer: BertWordPieceTokenizer
  ) {}

  static async fromOptions(options?: QAOptions): Promise<QAClient> {
    const model = await this.getModel(options?.model);

    const tokenizer =
      options?.tokenizer ??
      (await BertWordPieceTokenizer.fromOptions({
        vocabFile: options?.vocabPath ?? DEFAULT_VOCAB_PATH,
        lowercase: !model.params.cased
      }));

    return new QAClient(model, tokenizer);
  }

  async predict(
    question: string,
    context: string,
    maxAnswerLength = 15
  ): Promise<Answer | null> {
    const features = await this.getFeatures(question, context);

    const [startLogits, endLogits] = await this.model.runInference(
      features.map(f => f.encoding.getIds()),
      features.map(f => f.encoding.getAttentionMask())
    );

    return this.getAnswer(features, startLogits, endLogits, maxAnswerLength);
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
    const encodings = [encoding, ...encoding.getOverflowing()];

    const questionLength =
      encoding.getTokens().indexOf(this.tokenizer.configuration.sepToken) - 1; // Take [CLS] into account
    const questionLengthWithTokens = questionLength + 2;

    const spans: Span[] = encodings.map((e, i) => {
      const specialTokensMask = e.getSpecialTokensMask();
      const nbAddedTokens = specialTokensMask.reduce((acc, val) => acc + val, 0);
      const actualLength = specialTokensMask.length - nbAddedTokens;

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
    const offsets = answer.feature.encoding.getOffsets();
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

  private static async getModel(options?: ModelOptions): Promise<Model> {
    if (options && options.remote) {
      return RemoteModel.fromOptions(options);
    } else {
      return LocalModel.fromOptions(options ?? { path: DEFAULT_MODEL_PATH, cased: true });
    }
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
