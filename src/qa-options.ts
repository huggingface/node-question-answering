import * as path from "path";

import { Model } from "./models/model";
import { Tokenizer } from "./tokenizers";

export const DEFAULT_ASSETS_PATH = path.join(process.cwd(), "./.models");
export const DEFAULT_MODEL_PATH = path.join(DEFAULT_ASSETS_PATH, "distilbert-cased");
export const DEFAULT_VOCAB_PATH = path.join(DEFAULT_MODEL_PATH, "vocab.txt");

export interface QAOptions {
  model?: Model;
  mergesPath?: string;
  timeIt?: boolean;
  tokenizer?: Tokenizer;
  vocabPath?: string;
}
