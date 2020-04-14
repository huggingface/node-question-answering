import fs from "fs";
import https from "https";
import fetch from "node-fetch";
import path from "path";
import ProgressBar from "progress";
import shell from "shelljs";
import tar from "tar";
import { promisify } from "util";

import { getModelType, ModelType } from "./models/model";
import { ROOT_DIR } from "./qa-options";
import { LocalRuntime, RuntimeType } from "./runtimes/runtime";

export const exists = promisify(fs.exists);

/**
 * Ensures a directory exists, creates as needed.
 */
export async function ensureDir(dirPath: string, recursive = true): Promise<void> {
  if (!(await exists(dirPath))) {
    recursive ? shell.mkdir("-p", dirPath) : shell.mkdir(dirPath);
  }
}

export interface DownloadOptions {
  model: string;
  dir: string;
  format?: RuntimeType;
  force?: boolean;
  fullDir?: boolean;
}

/**
 * Download a model with associated vocabulary files
 * @param options Download options
 */
export async function downloadModelWithVocab(options: DownloadOptions): Promise<void> {
  const modelFormat = options.format ?? RuntimeType.SavedModel;

  const assetsDir = getAbsolutePath(options.dir);
  const modelDir = path.join(assetsDir, options.model);

  if (options.force) {
    shell.rm("-rf", modelDir);
  }

  if (modelFormat !== RuntimeType.Remote) {
    await downloadModel({
      dir: modelDir,
      format: modelFormat,
      name: options.model,
      verbose: true
    });
  }

  await getVocab({
    dir: modelDir,
    modelName: options.model,
    verbose: true
  });

  shell.echo("\nModel successfully downloaded!");
}

export interface ModelDownloadOptions {
  /**
   * Absolute path to the directory under which download model
   */
  dir: string;
  format: LocalRuntime;
  name: string;
  verbose?: boolean;
}

export async function downloadModel(model: ModelDownloadOptions): Promise<void> {
  const modelDir = path.join(
    model.dir,
    model.format === RuntimeType.TFJS ? RuntimeType.TFJS : ""
  );

  if (await exists(modelDir)) {
    const exit = (): void =>
      void model.verbose &&
      shell.echo(
        `Model ${model.name} (format: ${model.format}) already exists, doing nothing...`
      );

    if (model.format === RuntimeType.TFJS) {
      return exit();
    } else if (model.format === RuntimeType.SavedModel) {
      if (await exists(path.join(modelDir, "saved_model.pb"))) {
        return exit();
      }
    }
  }

  await ensureDir(modelDir);
  shell.echo("Downloading model...");

  let url: string;
  // eslint-disable-next-line @typescript-eslint/no-use-before-define
  if (HF_MODELS_MAPPING[model.name]) {
    // eslint-disable-next-line @typescript-eslint/no-use-before-define
    const defaultUrl = HF_MODELS_MAPPING[model.name][model.format];
    if (!defaultUrl) {
      throw new Error(
        `This model does not appear to be available in ${model.format} format`
      );
    }

    url = defaultUrl;
  } else {
    url = getHfUrl(model.name, "saved_model.tar.gz");
  }

  await new Promise((resolve, reject) => {
    // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
    https.get(url, res => {
      const bar = new ProgressBar("[:bar] :percent :etas", {
        width: 30,
        total: parseInt(res.headers["content-length"] ?? "0", 10)
      });

      res
        .on("data", chunk => bar.tick(chunk.length))
        .pipe(tar.x({ cwd: modelDir }))
        .on("close", resolve)
        .on("error", reject);
    });
  });
}

export interface VocabDownloadOptions extends VocabFiles {
  /**
   * Absolute path to the directory under which download vocab files
   */
  dir: string;
  modelName: string;
  verbose?: boolean;
}

interface VocabFiles {
  /**
   * Name of the merges file (if applicable to the tokenizer)
   * @default "merges.txt"
   */
  mergesFile?: string;
  /**
   * Name of the vocab file (if applicable to the tokenizer)
   * @default "vocab.txt" | "vocab.json"
   */
  vocabFile?: string;
}

type VocabFilesKey = keyof VocabFiles;
type ConfigFilesKey = "tokenizer_config.json" | "special_tokens_map.json";

const VOCAB_CONFIG_KEYS: ConfigFilesKey[] = [
  "tokenizer_config.json",
  "special_tokens_map.json"
];

export interface VocabConfiguration {
  tokenizer: {
    do_lower_case?: boolean;
  };
  tokensMapping: Partial<Record<TokenMappingKey, string>>;
}

export type TokenMappingKey =
  | "cls_token"
  | "eos_token"
  | "mask_token"
  | "pad_token"
  | "sep_token"
  | "unk_token";

const VOCAB_MAPPING: Partial<Record<ModelType, VocabFiles>> = {
  [ModelType.Roberta]: { mergesFile: "merges.txt", vocabFile: "vocab.json" }
};

const DEFAULT_VOCAB = { vocabFile: { name: "vocab.txt" } };

type VocabReturn<TReturnConfig> = TReturnConfig extends true ? VocabConfiguration : void;

export async function getVocab<TReturnConfig extends boolean>(
  options: VocabDownloadOptions,
  returnConfig?: TReturnConfig
): Promise<VocabReturn<TReturnConfig>> {
  await ensureDir(options.dir);

  const modelType = getModelType(options.modelName);
  if (!modelType) {
    throw new Error(
      "The model name does not allow to infer the associated tokenizer and thus which vocab files to download"
    );
  }

  let vocabFiles: Partial<Record<
    VocabFilesKey | ConfigFilesKey,
    { name: string; url?: string; optional?: boolean }
  >>;
  // eslint-disable-next-line @typescript-eslint/no-use-before-define
  const hfVocabUrl: string | undefined = HF_VOCAB_FILES_MAPPING[options.modelName];
  if (hfVocabUrl) {
    vocabFiles = { vocabFile: { name: "vocab.txt", url: hfVocabUrl } };
  } else {
    const mapping = VOCAB_MAPPING[modelType];
    if (mapping) {
      vocabFiles = {};
      for (const key of Object.keys(mapping) as VocabFilesKey[]) {
        // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
        vocabFiles[key] = { name: mapping[key]! };
      }
    } else {
      vocabFiles = DEFAULT_VOCAB;
    }

    for (const file of ["mergesFile", "vocabFile"] as VocabFilesKey[]) {
      if (options[file]) {
        // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
        vocabFiles[file] = { name: options[file]! };
      }
    }
  }

  const vocabConfig: VocabConfiguration = {
    tokenizer: {},
    tokensMapping: {}
  };

  for (const file of VOCAB_CONFIG_KEYS) {
    vocabFiles[file] = { name: file, optional: true };
  }

  for (const vocabFile of Object.values(vocabFiles)) {
    if (!vocabFile) {
      continue;
    }

    const file = path.join(options.dir, vocabFile.name);
    if (!(await exists(file))) {
      shell.echo(`Downloading ${vocabFile.name}...`);

      const url = vocabFile.url ?? getHfUrl(options.modelName, vocabFile.name);

      const response = await fetch(url);
      if (!response.ok) {
        if (vocabFile.optional !== true) {
          throw new Error(`Unable to download ${vocabFile.name} at ${url}`);
        } else {
          continue;
        }
      }

      const rawValue = await response.text();
      await fs.promises.writeFile(file, rawValue);

      if (returnConfig && VOCAB_CONFIG_KEYS.includes(vocabFile.name as ConfigFilesKey)) {
        const configKey = getVocabConfigKey(vocabFile.name as ConfigFilesKey);
        vocabConfig[configKey] = JSON.parse(rawValue);
      }
    } else {
      options.verbose && shell.echo(`${vocabFile.name} already exists, doing nothing...`);

      if (returnConfig && VOCAB_CONFIG_KEYS.includes(vocabFile.name as ConfigFilesKey)) {
        try {
          const configFile = await fs.promises.readFile(file, { encoding: "utf-8" });
          const configKey = getVocabConfigKey(vocabFile.name as ConfigFilesKey);
          vocabConfig[configKey] = JSON.parse(configFile);
        } catch (error) {
          // Nothing
        }
      }
    }
  }

  if (returnConfig === true) {
    return vocabConfig as VocabReturn<TReturnConfig>;
  }

  return void 0 as VocabReturn<TReturnConfig>;
}

function getHfUrl(model: string, file: string): string {
  return `https://cdn.huggingface.co/${model}/${file}`;
}

function getVocabConfigKey(filename: ConfigFilesKey): keyof VocabConfiguration {
  return filename === "tokenizer_config.json" ? "tokenizer" : "tokensMapping";
}

export function getAbsolutePath(pathToCheck?: string, rootDir = ROOT_DIR): string {
  if (!pathToCheck) {
    return rootDir;
  }

  return path.isAbsolute(pathToCheck) ? pathToCheck : path.join(rootDir, pathToCheck);
}

const HF_VOCAB_FILES_MAPPING: Record<string, string> = {
  /** DistilBERT */
  "distilbert-base-uncased-distilled-squad":
    "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-vocab.txt",
  "distilbert-base-cased-distilled-squad":
    "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-vocab.txt",
  /** BERT */
  "bert-large-uncased-whole-word-masking-finetuned-squad":
    "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-finetuned-squad-vocab.txt",
  "bert-large-cased-whole-word-masking-finetuned-squad":
    "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-finetuned-squad-vocab.txt"
};

interface DefaultModel {
  [RuntimeType.SavedModel]: string;
  [RuntimeType.TFJS]?: string;
}

const HF_MODELS_MAPPING: Record<string, DefaultModel> = {
  /** BERT */
  "bert-large-cased-whole-word-masking-finetuned-squad": {
    [RuntimeType.SavedModel]:
      "https://cdn.huggingface.co/bert-large-cased-whole-word-masking-finetuned-squad-saved_model.tar.gz"
  },
  "bert-large-uncased-whole-word-masking-finetuned-squad": {
    [RuntimeType.SavedModel]:
      "https://cdn.huggingface.co/bert-large-uncased-whole-word-masking-finetuned-squad-saved_model.tar.gz"
  },
  /** DistilBERT */
  "distilbert-base-cased-distilled-squad": {
    [RuntimeType.SavedModel]:
      "https://cdn.huggingface.co/distilbert-base-cased-distilled-squad-384-saved_model.tar.gz",
    [RuntimeType.TFJS]:
      "https://cdn.huggingface.co/distilbert-base-cased-distilled-squad-384-tfjs.tar.gz"
  },
  "distilbert-base-uncased-distilled-squad": {
    [RuntimeType.SavedModel]:
      "https://cdn.huggingface.co/distilbert-base-uncased-distilled-squad-384-saved_model.tar.gz"
  }
};
