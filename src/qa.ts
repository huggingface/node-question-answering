import * as tf from "@tensorflow/tfjs-node";
import { NamedTensorMap } from "@tensorflow/tfjs-node";
import { TFSavedModel } from "@tensorflow/tfjs-node/dist/saved_model";
import path from "path";
import { BertWordPieceTokenizer, Encoding, TruncationStrategy } from "tokenizers";

import {
  ModelInputsNames,
  ModelOptions,
  ModelOutputNames,
  QAOptions
} from "./qa-options";

const ASSETS_PATH = path.join(process.cwd(), "./.models");
const MODEL_PATH = path.join(ASSETS_PATH, "distilbert-cased");
const VOCAB_PATH = path.join(MODEL_PATH, "vocab.txt");

const DEFAULT_MODEL: ModelParams = {
  cased: true,
  inputsNames: {
    attentionMask: "attention_mask",
    ids: "input_ids"
  },
  outputsNames: {
    endLogits: "output_1",
    startLogits: "output_0"
  },
  path: MODEL_PATH,
  shape: [-1, 384],
  signatureName: "serving_default"
};

interface ModelParams extends Required<ModelOptions> {
  inputsNames: Required<ModelInputsNames>;
  outputsNames: Required<ModelOutputNames>;
  shape: [number, number];
}

interface Span {
  contextLength: number;
  contextStartIndex: number;
  length: number;
  startIndex: number;
}

interface Feature {
  contextLength: number;
  contextStartIndex: number;
  encoding: Encoding;
  maxContextMap: ReadonlyMap<number, boolean>;
}

interface Answer {
  score: number;
  text: string;
}

export class QAClient {
  private constructor(
    private readonly model: TFSavedModel,
    private readonly modelParams: ModelParams,
    private readonly tokenizer: BertWordPieceTokenizer
  ) {}

  static async fromOptions(options?: QAOptions): Promise<QAClient> {
    let modelParams: ModelParams;
    if (!options?.model) {
      modelParams = DEFAULT_MODEL;
    } else {
      const modelGraph = await tf.node.getMetaGraphsFromSavedModel(options.model.path);
      modelParams = this.getModelParams(options.model, modelGraph[0]);
    }

    const model = await tf.node.loadSavedModel(modelParams.path);
    const tokenizer =
      options?.tokenizer ??
      (await BertWordPieceTokenizer.fromOptions({
        vocabFile: options?.vocabPath ?? VOCAB_PATH,
        lowercase: !modelParams.cased
      }));

    return new QAClient(model, modelParams, tokenizer);
  }

  async predict(
    question: string,
    context: string,
    maxAnswerLength = 15
  ): Promise<Answer | null> {
    const features = await this.getFeatures(question, context);

    const inputTensor = tf.tensor(
      features.map(f => f.encoding.getIds()),
      undefined,
      "int32"
    );

    const maskTensor = tf.tensor(
      features.map(f => f.encoding.getAttentionMask()),
      undefined,
      "int32"
    );

    const result = this.model.predict({
      [this.modelParams.inputsNames.ids]: inputTensor,
      [this.modelParams.inputsNames.attentionMask]: maskTensor
    }) as NamedTensorMap;

    let startLogits = (await result[this.modelParams.outputsNames.startLogits]
      .squeeze()
      .array()) as number[] | number[][];
    let endLogits = (await result[this.modelParams.outputsNames.endLogits]
      .squeeze()
      .array()) as number[] | number[][];

    if (isOneDimensional(startLogits)) {
      startLogits = [startLogits];
    }

    if (isOneDimensional(endLogits)) {
      endLogits = [endLogits];
    }

    return this.getAnswer(features, startLogits, endLogits, maxAnswerLength);
  }

  private async getFeatures(
    question: string,
    context: string,
    stride = 128
  ): Promise<Feature[]> {
    this.tokenizer.setPadding({ maxLength: this.modelParams.shape[1] });
    this.tokenizer.setTruncation(this.modelParams.shape[1], {
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

  private static getModelParams(
    modelOptions: ModelOptions,
    graph: tf.MetaGraph
  ): ModelParams {
    const partialParams: Omit<ModelParams, "path" | "shape"> = {
      cased: modelOptions.cased ?? false,
      inputsNames: {
        attentionMask:
          modelOptions.inputsNames?.attentionMask ??
          DEFAULT_MODEL.inputsNames.attentionMask,
        ids: modelOptions.inputsNames?.ids ?? DEFAULT_MODEL.inputsNames.ids
      },
      outputsNames: {
        endLogits:
          modelOptions.outputsNames?.endLogits ?? DEFAULT_MODEL.outputsNames.endLogits,
        startLogits:
          modelOptions.outputsNames?.startLogits ?? DEFAULT_MODEL.outputsNames.startLogits
      },
      signatureName: modelOptions.signatureName ?? DEFAULT_MODEL.signatureName
    };

    const signatureDef = graph.signatureDefs[partialParams.signatureName];
    if (!signatureDef) {
      throw new Error(`No signature matching name "${partialParams.signatureName}"`);
    }

    for (const inputName of Object.values(partialParams.inputsNames)) {
      if (!signatureDef.inputs[inputName]) {
        throw new Error(`No input matching name "${inputName}"`);
      }
    }

    for (const outputName of Object.values(partialParams.outputsNames)) {
      if (!signatureDef.outputs[outputName]) {
        throw new Error(`No output matching name "${outputName}"`);
      }
    }

    const rawShape = signatureDef.inputs[partialParams.inputsNames.ids!].shape as
      | number[]
      | { array: [number] }[];
    const shape =
      typeof rawShape[0] === "number"
        ? rawShape
        : (rawShape as { array: [number] }[]).map(s => s.array[0]);

    return {
      ...partialParams,
      path: modelOptions.path,
      shape: shape as [number, number]
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

function isOneDimensional(arr: number[] | number[][]): arr is number[] {
  return !Array.isArray(arr[0]);
}
