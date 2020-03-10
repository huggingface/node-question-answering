import * as path from "path";
import { BertWordPieceTokenizer } from "tokenizers";

import { Model } from "./models/model";

export const DEFAULT_ASSETS_PATH = path.join(process.cwd(), "./.models");
export const DEFAULT_MODEL_PATH = path.join(DEFAULT_ASSETS_PATH, "distilbert-cased");
export const DEFAULT_VOCAB_PATH = path.join(DEFAULT_MODEL_PATH, "vocab.txt");

export interface QAOptions {
  model?: Model;
  timeIt?: boolean;
  tokenizer?: BertWordPieceTokenizer;
  vocabPath?: string;
}
