import * as tf from "@tensorflow/tfjs-node";
import { NamedTensorMap } from "@tensorflow/tfjs-node";
import { TFSavedModel } from "@tensorflow/tfjs-node/dist/saved_model";
import path from "path";
import { BertWordPieceTokenizer } from "tokenizers";

const ASSETS_PATH = path.join(__dirname, "../assets");
const MODEL_PATH = path.join(ASSETS_PATH, "distilbert");
const VOCAB_PATH = path.join(ASSETS_PATH, "vocab.txt");

const DEFAULT_MODEL: ModelParams = {
  inputsNames: {
    attentionMask: "attention_mask",
    ids: "input_ids"
  },
  outputsNames: {
    endLogits: "output_1",
    startLogits: "output_0"
  },
  path: MODEL_PATH,
  shape: [1, 384],
  signatureName: "serving_default"
};

export interface QAOptions {
  model?: ModelOptions;
  /**
   * Must match with the model input shape
   * @default 384
   */
  maxSequenceLength?: number;
  tokenizer?: BertWordPieceTokenizer;
  vocabPath?: string;
}

export interface ModelOptions {
  inputsNames?: ModelInputsNames;
  outputsNames?: ModelOutputNames;
  path: string;
  /**
   * @default "serving_default"
   */
  signatureName?: string;
}

export interface ModelInputsNames {
  /**
   * @default "inputs_ids"
   */
  ids?: string;
  /**
   * @default "attention_mask"
   */
  attentionMask?: string;
}

export interface ModelOutputNames {
  /**
   * @default "output_0"
   */
  startLogits?: string;
  /**
   * @default "output_1"
   */
  endLogits?: string;
}

interface ModelParams extends Required<ModelOptions> {
  inputsNames: Required<ModelInputsNames>;
  outputsNames: Required<ModelOutputNames>;
  shape: [number, number];
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
      const modelGraph = await tf.node.getMetaGraphsFromSavedModel(MODEL_PATH);
      modelParams = this.getModelParams(options.model, modelGraph[0]);
    }

    const model = await tf.node.loadSavedModel(modelParams.path);
    const tokenizer =
      options?.tokenizer ??
      (await BertWordPieceTokenizer.fromOptions({
        vocabFile: options?.vocabPath ?? VOCAB_PATH
      }));

    return new QAClient(model, modelParams, tokenizer);
  }

  async predict(
    question: string,
    context: string,
    maxAnswerLength = 15
  ): Promise<Answer | null> {
    const encoding = await this.tokenizer.encode(question, context);
    encoding.pad(this.modelParams.shape[1]);

    const inputTensor = tf.tensor(encoding.getIds(), this.modelParams.shape, "int32");
    const maskTensor = tf.tensor(
      encoding.getAttentionMask(),
      this.modelParams.shape,
      "int32"
    );

    const result = this.model.predict({
      [this.modelParams.inputsNames.ids]: inputTensor,
      [this.modelParams.inputsNames.attentionMask]: maskTensor
    }) as NamedTensorMap;

    const startLogits = (await result[this.modelParams.outputsNames.startLogits]
      .squeeze()
      .array()) as number[];
    const endLogits = (await result[this.modelParams.outputsNames.endLogits]
      .squeeze()
      .array()) as number[];

    const startProbs = softMax(startLogits);
    const endProbs = softMax(endLogits);

    const typeIds = encoding.getTypeIds();
    const contextFirstIndex = typeIds.findIndex(x => x === 1);
    const contextLastIndex =
      typeIds.findIndex((x, i) => i > contextFirstIndex && x === 0) - 1;

    const [sortedStartProbs, sortedEndProbs] = [startProbs, endProbs].map(logits =>
      logits
        .slice(contextFirstIndex, contextLastIndex)
        .map((val, i) => [i + contextFirstIndex, val])
        .sort((a, b) => b[1] - a[1])
    );

    for (const startLogit of sortedStartProbs) {
      for (const endLogit of sortedEndProbs) {
        if (endLogit[0] < startLogit[0]) {
          continue;
        }

        if (endLogit[0] - startLogit[0] + 1 > maxAnswerLength) {
          continue;
        }

        const text: string[] = [];
        const tokens = encoding.getTokens();
        for (let i = startLogit[0]; i <= endLogit[0]; i++) {
          text.push(tokens[i]);
        }

        const rawScore = startLogit[1] * endLogit[1];
        return {
          text: text.join(" "),
          score: Math.round((rawScore + Number.EPSILON) * 100) / 100
        };
      }
    }

    return null;
  }

  private static getModelParams(
    modelOptions: ModelOptions,
    graph: tf.MetaGraph
  ): ModelParams {
    const partialParams: Omit<ModelParams, "path" | "shape"> = {
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

    for (const inputName in partialParams.inputsNames) {
      if (!signatureDef.inputs[inputName]) {
        throw new Error(`No input matching name "${inputName}"`);
      }
    }

    for (const outputName in partialParams.outputsNames) {
      if (!signatureDef.outputs[outputName]) {
        throw new Error(`No output matching name "${outputName}"`);
      }
    }

    return {
      ...partialParams,
      path: modelOptions.path,
      shape: signatureDef.inputs[partialParams.inputsNames.ids!].shape as [number, number]
    };
  }
}

function softMax(values: number[]): number[] {
  const max = Math.max(...values);
  const exps = values.map(x => Math.exp(x - max));
  const expsSum = exps.reduce((a, b) => a + b);
  return exps.map(e => e / expsSum);
}
