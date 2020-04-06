import { exists as fsExists } from "fs";
import path from "path";
import {
  AddedToken,
  ByteLevelBPETokenizer,
  Encoding,
  getTokenContent,
  Token
} from "tokenizers";
import { robertaProcessing } from "tokenizers/dist/bindings/post-processors";
import { PaddingConfiguration } from "tokenizers/dist/bindings/tokenizer";
import { promisify } from "util";

import { Tokenizer, TokenizerOptions } from "./tokenizer";

export interface RobertaTokenizerOptions {
  clsToken: Token;
  eosToken: Token;
  maskToken: Token;
  padToken: Token;
  unkToken: Token;
}

export class RobertaTokenizer extends Tokenizer<ByteLevelBPETokenizer> {
  private readonly clsToken: Token;
  private readonly eosToken: Token;
  private readonly maskToken: Token;
  private readonly padToken: Token;
  private readonly unkToken: Token;

  constructor(tokenizer: ByteLevelBPETokenizer, options: RobertaTokenizerOptions) {
    super(tokenizer);

    this.clsToken = options.clsToken;
    this.eosToken = options.eosToken;
    this.maskToken = options.maskToken;
    this.padToken = options.padToken;
    this.unkToken = options.unkToken;
  }

  static async fromOptions(options: TokenizerOptions): Promise<RobertaTokenizer> {
    const existsAsync = promisify(fsExists);
    let vocabFile = options.vocabPath;

    if (!vocabFile) {
      const fullPath = path.join(options.modelPath, "vocab.json");
      if (await existsAsync(fullPath)) {
        vocabFile = fullPath;
      }

      if (!vocabFile) {
        throw new Error(
          "Unable to find a vocab file. Make sure to provide its path in the options"
        );
      }
    }

    let mergesFile = options.mergesPath;
    if (!mergesFile) {
      const fullPath = path.join(options.modelPath, "merges.txt");
      if (await existsAsync(fullPath)) {
        mergesFile = fullPath;
      }

      if (!mergesFile) {
        throw new Error(
          "Unable to find a merges file. Make sure to provide its path in the options"
        );
      }
    }

    const tokenizer = await ByteLevelBPETokenizer.fromOptions({
      addPrefixSpace: true,
      mergesFile,
      vocabFile
    });

    const clsToken = "<s>";
    const eosToken = "</s>";
    const maskToken = new AddedToken("<mask>", { leftStrip: true });
    const padToken = "<pad>";
    const unkToken = "<unk>";

    const postProcessor = robertaProcessing(
      // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
      [eosToken, tokenizer.tokenToId(eosToken)!],
      // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
      [clsToken, tokenizer.tokenToId(clsToken)!]
    );

    tokenizer.setPostProcessor(postProcessor);
    tokenizer.addSpecialTokens([clsToken, eosToken, maskToken, padToken, unkToken]);

    return new RobertaTokenizer(tokenizer, {
      clsToken,
      eosToken,
      maskToken,
      padToken,
      unkToken
    });
  }

  getQuestionLength(encoding: Encoding): number {
    return encoding.tokens.indexOf(getTokenContent(this.eosToken)) - 1; // Take cls token into account
  }

  getContextStartIndex(encoding: Encoding): number {
    return this.getQuestionLength(encoding) + 3;
  }

  /**
   * Enable/change padding with specified options
   * @param maxLength Padding length
   * @override
   */
  setPadding(maxLength: number): Readonly<PaddingConfiguration> {
    const padToken = getTokenContent(this.padToken);
    return this.tokenizer.setPadding({
      maxLength,
      padToken: padToken,
      padId: this.tokenizer.tokenToId(padToken)
    });
  }
}
