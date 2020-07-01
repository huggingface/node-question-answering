import path from "path";
import {
  AddedToken,
  ByteLevelBPETokenizer,
  Encoding,
  getTokenContent,
  PaddingConfiguration,
  Token
} from "tokenizers";
import { robertaProcessing } from "tokenizers/bindings/post-processors";

import { exists } from "../utils";
import { FullTokenizerOptions, Tokenizer } from "./tokenizer";

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

  static async fromOptions(
    options: FullTokenizerOptions<RobertaTokenizerOptions>
  ): Promise<RobertaTokenizer> {
    let vocabFile = options.vocabFile;

    if (!vocabFile) {
      const fullPath = path.join(options.filesDir, "vocab.json");
      if (await exists(fullPath)) {
        vocabFile = fullPath;
      }

      if (!vocabFile) {
        throw new Error(
          "Unable to find a vocab file. Make sure to provide its path in the options"
        );
      }
    }

    let mergesFile = options.mergesFile;
    if (!mergesFile) {
      const fullPath = path.join(options.filesDir, "merges.txt");
      if (await exists(fullPath)) {
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

    const clsToken = options.clsToken ?? "<s>";
    const eosToken = options.eosToken ?? "</s>";
    const maskToken =
      options.maskToken ?? new AddedToken("<mask>", true, { leftStrip: true });
    const padToken = options.padToken ?? "<pad>";
    const unkToken = options.unkToken ?? "<unk>";

    const eosString = getTokenContent(eosToken);
    const clsString = getTokenContent(clsToken);
    const postProcessor = robertaProcessing(
      // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
      [eosString, tokenizer.tokenToId(eosString)!],
      // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
      [clsString, tokenizer.tokenToId(clsString)!]
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
