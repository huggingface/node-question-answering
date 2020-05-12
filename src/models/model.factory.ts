import path from "path";

import { DEFAULT_ASSETS_DIR } from "../qa-options";
import { Remote } from "../runtimes/remote.runtime";
import { LocalRuntime, Runtime, RuntimeOptions, RuntimeType } from "../runtimes/runtime";
import { SavedModel } from "../runtimes/saved-model.runtime";
import { TFJS } from "../runtimes/tfjs.runtime";
import { downloadModel, getAbsolutePath } from "../utils";
import { BertModel } from "./bert.model";
import { DistilbertModel } from "./distilbert.model";
import {
  getModelType,
  Model,
  ModelInputsNames,
  ModelOutputNames,
  ModelType
} from "./model";
import { RobertaModel } from "./roberta.model";

interface CommonModelOptions {
  inputsNames?: ModelInputsNames;
  /**
   * Fully qualified name of the model (including the author if applicable)
   * @example "distilbert-base-uncased-distilled-squad"
   * @example "deepset/bert-base-cased-squad2"
   */
  name: string;
  outputsNames?: ModelOutputNames;
  /**
   * Type of "runtime" to use for model inference: SavedModel, TFJS or remote.
   * @default RuntimeType.SavedModel
   */
  runtime?: RuntimeType;
  /**
   * @default "serving_default"
   */
  signatureName?: string;
  /**
   * Type of the model (inferred from model name by default)
   */
  type?: ModelType;
}

export interface RemoteModelOptions extends CommonModelOptions {
  /**
   * - If `runtime` is `Runtime.Remote` it must be the url at which the model is exposed.
   */
  path: string;
  runtime: RuntimeType.Remote;
}

export interface LocalModelOptions extends CommonModelOptions {
  /**
   * - For local SavedModel and TFJS, corresponds to the path of the location at which the models are located.
   * It can be absolute or relative to the root of the project.
   * Defaults to a `.models` directory at the root of the project (created if needed).
   */
  path?: string;
  runtime?: LocalRuntime;
}

export interface SavedModelOptions extends LocalModelOptions {
  runtime: RuntimeType.SavedModel;
  /**
   * Nb max of workers to instantiate for this model (values <= 0 will be ignored)
   * @default 5
   */
  workersMax?: number;
}

export type ModelFactoryOptions =
  | LocalModelOptions
  | SavedModelOptions
  | RemoteModelOptions;

export async function initModel(options: ModelFactoryOptions): Promise<Model> {
  const runtimeType = options.runtime ?? RuntimeType.SavedModel;

  let modelDir: string;
  if (options.runtime !== RuntimeType.Remote) {
    const assetsDir = getAbsolutePath(options.path, DEFAULT_ASSETS_DIR);
    modelDir = path.join(assetsDir, options.name);
    if (runtimeType !== RuntimeType.Remote) {
      await downloadModel({
        dir: modelDir,
        format: runtimeType,
        name: options.name
      });
    }
  } else {
    modelDir = options.path;
  }

  const modelType = options.type ?? getModelType(options.name);
  let ModelClass: typeof BertModel | typeof DistilbertModel | typeof RobertaModel;
  switch (modelType) {
    case ModelType.Roberta:
      ModelClass = RobertaModel;
      break;

    case ModelType.Bert:
      ModelClass = BertModel;
      break;

    default:
      ModelClass = DistilbertModel;
  }

  const runtimeOptions: RuntimeOptions = {
    inputs: ModelClass.inputs,
    inputsNames: options.inputsNames,
    outputsNames: options.outputsNames,
    path: modelDir,
    signatureName: options.signatureName
  };

  if (isSavedModelOptions(options)) {
    if (options.workersMax && options.workersMax > 0) {
      runtimeOptions.workersMax = options.workersMax;
    }
  }

  let runtime: Runtime;
  switch (runtimeType) {
    case RuntimeType.Remote:
      runtime = await Remote.fromOptions(runtimeOptions);
      break;

    case RuntimeType.TFJS:
      runtime = await TFJS.fromOptions(runtimeOptions);
      break;

    default:
      runtime = await SavedModel.fromOptions(runtimeOptions);
  }

  return new ModelClass(options.name, modelDir, runtime);
}

function isSavedModelOptions(options: ModelFactoryOptions): options is SavedModelOptions {
  return !options.runtime || options.runtime === RuntimeType.SavedModel;
}
