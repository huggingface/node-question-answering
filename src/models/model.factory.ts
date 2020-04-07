import { Remote } from "../runtimes/remote.runtime";
import { Runtime, RuntimeOptions, RuntimeType } from "../runtimes/runtime";
import { SavedModel } from "../runtimes/saved-model.runtime";
import { TFJS } from "../runtimes/tfjs.runtime";
import { BertModel } from "./bert.model";
import { DistilbertModel } from "./distilbert.model";
import { Model, ModelOptions, ModelType } from "./model";
import { RobertaModel } from "./roberta.model";

export async function initModel(options: ModelOptions): Promise<Model> {
  const modelType = getModelType(options);

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
    path: options.path,
    signatureName: options.signatureName
  };

  let runtime: Runtime;
  const runtimeType = options.runtime ?? RuntimeType.SavedModel;
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

  return new ModelClass(runtime, options.cased);
}

/**
 * Infer model type from model path
 * @param options Model options
 * @throws If no model type inferred
 */
function getModelType(options: ModelOptions): ModelType {
  if (options.type) {
    return options.type;
  }

  const types = Object.entries(ModelType);
  for (const [name, type] of types) {
    if (options.path.toLowerCase().includes(name.toLowerCase())) {
      return type;
    }
  }

  throw new Error(
    "Impossible to determine the type of the model. You can specify it manually by providing the `type` in the  options"
  );
}
