import * as path from "path";

import { Model } from "./models/model";
import { Tokenizer } from "./tokenizers";

export const ROOT_DIR = process.cwd();
export const DEFAULT_ASSETS_DIR = path.join(ROOT_DIR, "./.models");
export const DEFAULT_MODEL_NAME = "distilbert-base-cased-distilled-squad";
export const DEFAULT_VOCAB_PATH = path.join(DEFAULT_MODEL_NAME, "vocab.txt");

export interface QAOptions {
  /**
   * Inferred from the associated tokenizer configuration by default.
   * Ignore this setting if you use your own tokenizer in `tokenizer`.
   */
  cased?: boolean;
  /**
   * Model to use, defaults to a SavedModel Distilbert-base-cased finetuned on SQuAD
   */
  model?: Model;
  /**
   * Wether to time inference
   * @default false
   */
  timeIt?: boolean;
  /**
   * Custom tokenizer to use with the model, or tokenizer options
   * (in this case, defaults to the standard tokenizer associated with the model)
   */
  tokenizer?: Tokenizer | TokenizerOptions;
}

export interface TokenizerOptions {
  /**
   * Name of the merges file (if applicable to the tokenizer).
   * @default "merges.txt"
   */
  mergesFile?: string;
  /**
   * Directory under which the files needed by the tokenizer are located.
   * Can be absolute or relative to the root of the project.
   * Defaults to the model dir.
   */
  filesDir?: string;
  /**
   * Name of the vocab file (if applicable to the tokenizer).
   * @default "vocab.txt" | "vocab.json"
   */
  vocabFile?: string;
}
