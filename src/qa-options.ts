import * as path from "path";
import { BertWordPieceTokenizer } from "tokenizers";

export const DEFAULT_ASSETS_PATH = path.join(process.cwd(), "./.models");
export const DEFAULT_MODEL_PATH = path.join(DEFAULT_ASSETS_PATH, "distilbert-cased");
export const DEFAULT_VOCAB_PATH = path.join(DEFAULT_MODEL_PATH, "vocab.txt");

export interface QAOptions {
  model?: ModelOptions;
  timeIt?: boolean;
  tokenizer?: BertWordPieceTokenizer;
  vocabPath?: string;
}

export interface ModelOptions {
  /**
   * @default false
   */
  cased?: boolean;
  inputsNames?: ModelInputsNames;
  outputsNames?: ModelOutputNames;
  path: string;
  remote?: boolean;
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
