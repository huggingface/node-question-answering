import path from "path";
import {
  BertWordPieceOptions,
  BertWordPieceTokenizer,
  Encoding,
  getTokenContent,
  Token
} from "tokenizers";

import { DEFAULT_VOCAB_PATH } from "../qa-options";
import { exists } from "../utils";
import { FullTokenizerOptions, Tokenizer } from "./tokenizer";

export interface BertTokenizerOptions {
  clsToken: Token;
  maskToken: Token;
  padToken: Token;
  sepToken: Token;
  unkToken: Token;
}

export class BertTokenizer extends Tokenizer<BertWordPieceTokenizer> {
  static async fromOptions(
    options: FullTokenizerOptions<BertTokenizerOptions>
  ): Promise<BertTokenizer> {
    let vocabPath = options.vocabFile;
    if (!vocabPath) {
      const fullPath = path.join(options.filesDir, "vocab.txt");
      if (await exists(fullPath)) {
        vocabPath = fullPath;
      }

      vocabPath = vocabPath ?? DEFAULT_VOCAB_PATH;
    }

    const tokenizerOptions: BertWordPieceOptions = {
      vocabFile: vocabPath,
      lowercase: options.lowercase
    };

    const tokens: (keyof BertTokenizerOptions)[] = [
      "clsToken",
      "maskToken",
      "padToken",
      "sepToken",
      "unkToken"
    ];

    for (const token of tokens) {
      if (options[token]) {
        tokenizerOptions[token] = options[token];
      }
    }

    const tokenizer = await BertWordPieceTokenizer.fromOptions(tokenizerOptions);
    return new BertTokenizer(tokenizer);
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
