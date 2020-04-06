import { Encoding, TruncationOptions } from "tokenizers";
import {
  PaddingConfiguration,
  TruncationConfiguration
} from "tokenizers/dist/bindings/tokenizer";
import { BaseTokenizer } from "tokenizers/dist/implementations/tokenizers/base.tokenizer";

// import {
//   PaddingConfiguration,
//   TruncationConfiguration
// } from "tokenizers/bindings/tokenizer";
import { ModelType } from "../models/model";

export interface TokenizerOptions {
  lowercase?: boolean;
  mergesPath?: string;
  modelPath: string;
  modelType: ModelType;
  vocabPath?: string;
}

export abstract class Tokenizer<T extends BaseTokenizer<object> = BaseTokenizer<object>> {
  constructor(protected tokenizer: T) {}

  abstract getQuestionLength(encoding: Encoding): number;

  abstract getContextStartIndex(encoding: Encoding): number;

  /**
   * Get the last index of the context of an encoding
   * @param encoding Encoding for which to return last context index
   * @virtual
   */
  getContextEndIndex(encoding: Encoding): number {
    const nbAddedTokens = encoding.specialTokensMask.reduce((acc, val) => acc + val, 0);
    const actualLength = encoding.length - nbAddedTokens;
    const contextLength = actualLength - this.getQuestionLength(encoding);

    return this.getContextStartIndex(encoding) + contextLength - 1;
  }

  encode(sequence: string, pair?: string, addSpecialTokens = true): Promise<Encoding> {
    return this.tokenizer.encode(sequence, pair, addSpecialTokens);
  }

  /**
   * Enable/change padding with specified options
   * @param maxLength Padding length
   * @virtual
   */
  setPadding(maxLength: number): Readonly<PaddingConfiguration> {
    return this.tokenizer.setPadding({ maxLength });
  }

  setTruncation(
    maxLength: number,
    options?: TruncationOptions
  ): Readonly<TruncationConfiguration> {
    return this.tokenizer.setTruncation(maxLength, options);
  }
}
