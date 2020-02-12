import { BertWordPieceTokenizer } from "tokenizers";

export interface QAOptions {
  model?: ModelOptions;
  tokenizer?: BertWordPieceTokenizer;
  vocabPath?: string;
}

export interface ModelOptions {
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
