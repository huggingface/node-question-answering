import { getModelType, ModelType } from "../models/model";
import { DEFAULT_ASSETS_DIR } from "../qa-options";
import { getAbsolutePath, getVocab, TokenMappingKey } from "../utils";
import { BertTokenizer, BertTokenizerOptions } from "./bert.tokenizer";
import { RobertaTokenizer, RobertaTokenizerOptions } from "./roberta.tokenizer";
import { FullTokenizerOptions, Tokenizer } from "./tokenizer";

interface TokenizerFactoryBaseOptions {
  lowercase?: boolean;
  /**
   * Name of the merges file (if applicable to the tokenizer)
   * @default "merges.txt"
   */
  mergesFile?: string;
  /**
   * Directory under which the files needed by the tokenizer are located.
   * Can be absolute or relative to the root of the project.
   * If no corresponding files are found AND a `modelName` is provided, an attempt will be made to download them.
   */
  filesDir: string;
  /**
   * Fully qualified name of the model (including the author if applicable)
   * @example "distilbert-base-uncased-distilled-squad"
   * @example "deepset/bert-base-cased-squad2"
   */
  modelName?: string;
  /**
   * Type of the model (inferred from model name by default)
   */
  modelType?: ModelType;
  /**
   * Name of the vocab file (if applicable to the tokenizer)
   * @default "vocab.txt" | "vocab.json"
   */
  vocabFile?: string;
}

export interface RobertaTokenizerFactoryOptions
  extends TokenizerFactoryBaseOptions,
    Partial<RobertaTokenizerOptions> {
  modelType: ModelType.Roberta;
}

export interface BertTokenizerFactoryOptions
  extends TokenizerFactoryBaseOptions,
    Partial<RobertaTokenizerOptions> {
  modelType?: ModelType.Bert | ModelType.Distilbert;
}

export type TokenizerFactoryOptions =
  | RobertaTokenizerFactoryOptions
  | BertTokenizerFactoryOptions;

const TOKEN_KEYS_MAPPING: Record<
  TokenMappingKey,
  keyof FullTokenizerOptions<BertTokenizerOptions & RobertaTokenizerOptions>
> = {
  // eslint-disable-next-line @typescript-eslint/camelcase
  cls_token: "clsToken",
  // eslint-disable-next-line @typescript-eslint/camelcase
  eos_token: "eosToken",
  // eslint-disable-next-line @typescript-eslint/camelcase
  mask_token: "maskToken",
  // eslint-disable-next-line @typescript-eslint/camelcase
  pad_token: "padToken",
  // eslint-disable-next-line @typescript-eslint/camelcase
  sep_token: "sepToken",
  // eslint-disable-next-line @typescript-eslint/camelcase
  unk_token: "unkToken"
};

export async function initTokenizer(
  options: TokenizerFactoryOptions
): Promise<Tokenizer> {
  let modelType = options.modelType;
  if (!modelType) {
    if (!options.modelName) {
      throw new Error(
        "Either a model type or a model name must be provided to init a tokenizer"
      );
    }

    modelType = getModelType(options.modelName);
  }

  const fullOptions: FullTokenizerOptions<BertTokenizerOptions &
    RobertaTokenizerOptions> = {
    ...options,
    filesDir: getAbsolutePath(options.filesDir, DEFAULT_ASSETS_DIR),
    modelType
  };

  if (options.modelName) {
    const vocabConfig = await getVocab(
      {
        dir: fullOptions.filesDir,
        modelName: options.modelName,
        mergesFile: options.mergesFile,
        vocabFile: options.vocabFile
      },
      true
    );

    if (
      typeof fullOptions.lowercase === "undefined" &&
      typeof vocabConfig.tokenizer.do_lower_case === "boolean"
    ) {
      fullOptions.lowercase = vocabConfig.tokenizer.do_lower_case;
    }

    for (const [key, mapping] of Object.entries(TOKEN_KEYS_MAPPING)) {
      if (
        fullOptions[mapping] === undefined &&
        typeof vocabConfig.tokensMapping[key as TokenMappingKey] === "string"
      ) {
        (fullOptions[mapping] as string) = vocabConfig.tokensMapping[
          key as TokenMappingKey
        ] as string;
      }
    }
  }

  if (typeof fullOptions.lowercase === "undefined") {
    if (options.modelName?.toLocaleLowerCase().includes("uncased")) {
      fullOptions.lowercase = true;
    } else if (options.modelName?.toLocaleLowerCase().includes("cased")) {
      fullOptions.lowercase = false;
    }
  }

  let tokenizer: Tokenizer;
  switch (fullOptions.modelType) {
    case ModelType.Roberta:
      tokenizer = await RobertaTokenizer.fromOptions(fullOptions);
      break;

    default:
      tokenizer = await BertTokenizer.fromOptions(fullOptions);
      break;
  }

  return tokenizer;
}
