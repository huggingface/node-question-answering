import * as tf from "@tensorflow/tfjs-node";
import { NamedTensorMap } from "@tensorflow/tfjs-node";
import { TFSavedModel } from "@tensorflow/tfjs-node/dist/saved_model";
import path from "path";
import { BertWordPieceTokenizer } from "tokenizers";

const ASSETS_PATH = path.join(__dirname, "../assets");
const MODEL_PATH = path.join(ASSETS_PATH, "distilbert");
const VOCAB_PATH = path.join(ASSETS_PATH, "vocab.txt");

export interface QAOptions {
  modelPath?: string;
  tokenizer?: BertWordPieceTokenizer;
  vocabPath?: string;
}

export interface Answer {
  score: number;
  text: string;
}

export class QAClient {
  private constructor(
    private model: TFSavedModel,
    private tokenizer: BertWordPieceTokenizer
  ) {}

  static async fromOptions(options?: QAOptions): Promise<QAClient> {
    const model = await tf.node.loadSavedModel(options?.modelPath ?? MODEL_PATH);
    const tokenizer =
      options?.tokenizer ??
      (await BertWordPieceTokenizer.fromOptions({
        vocabFile: options?.vocabPath ?? VOCAB_PATH
      }));

    return new QAClient(model, tokenizer);
  }

  async predict(question: string, context: string): Promise<Answer | null> {
    const sequenceLength = 384;

    const encoding = await this.tokenizer.encode(question, context);
    encoding.pad(sequenceLength);

    const inputTensor = tf.tensor(encoding.getIds(), [1, sequenceLength], "int32");
    const maskTensor = tf.tensor(
      encoding.getAttentionMask(),
      [1, sequenceLength],
      "int32"
    );

    const result = this.model.predict(
      // eslint-disable-next-line @typescript-eslint/camelcase
      { input_ids: inputTensor, attention_mask: maskTensor },
      { verbose: true }
    ) as NamedTensorMap;

    const startLogits = (await result["output_0"].squeeze().array()) as number[];
    const endLogits = (await result["output_1"].squeeze().array()) as number[];

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
}

function softMax(values: number[]): number[] {
  const max = Math.max(...values);
  const exps = values.map(x => Math.exp(x - max));
  const expsSum = exps.reduce((a, b) => a + b);
  return exps.map(e => e / expsSum);
}
