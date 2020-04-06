import { exists as fsExists } from "fs";
import path from "path";
import { BertWordPieceTokenizer, Encoding, getTokenContent } from "tokenizers";
import { promisify } from "util";

import { DEFAULT_VOCAB_PATH } from "../qa-options";
import { Tokenizer, TokenizerOptions } from "./tokenizer";

export class DistilbertTokenizer extends Tokenizer<BertWordPieceTokenizer> {
  static async fromOptions(options: TokenizerOptions): Promise<DistilbertTokenizer> {
    let vocabPath = options.vocabPath;
    if (!vocabPath) {
      const fullPath = path.join(options.modelPath, "vocab.txt");
      const existsAsync = promisify(fsExists);
      if (await existsAsync(fullPath)) {
        vocabPath = fullPath;
      }

      vocabPath = vocabPath ?? DEFAULT_VOCAB_PATH;
    }

    const tokenizer = await BertWordPieceTokenizer.fromOptions({
      vocabFile: vocabPath,
      lowercase: options.lowercase
    });

    return new DistilbertTokenizer(tokenizer);
  }

  getQuestionLength(encoding: Encoding): number {
    return (
      encoding.tokens.indexOf(getTokenContent(this.tokenizer.configuration.sepToken)) - 1 // Take cls token into account
    );
  }

  getContextStartIndex(encoding: Encoding): number {
    return this.getQuestionLength(encoding) + 2;
  }
}
