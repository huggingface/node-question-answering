import * as tf from "@tensorflow/tfjs-node";
import { NamedTensorMap } from "@tensorflow/tfjs-node";
import { TFSavedModel } from "@tensorflow/tfjs-node/dist/saved_model";

import { Model, ModelParams } from "./model";
import { ModelOptions } from "./qa-options";

export class LocalModel extends Model {
  private constructor(private model: TFSavedModel, public params: ModelParams) {
    super();
  }

  async runInference(
    ids: number[][],
    attentionMask: number[][]
  ): Promise<[number[][], number[][]]> {
    const inputTensor = tf.tensor(ids, undefined, "int32");
    const maskTensor = tf.tensor(attentionMask, undefined, "int32");

    const result = this.model.predict({
      [this.params.inputsNames.ids]: inputTensor,
      [this.params.inputsNames.attentionMask]: maskTensor
    }) as NamedTensorMap;

    let startLogits = (await result[this.params.outputsNames.startLogits]
      .squeeze()
      .array()) as number[] | number[][];
    let endLogits = (await result[this.params.outputsNames.endLogits]
      .squeeze()
      .array()) as number[] | number[][];

    if (isOneDimensional(startLogits)) {
      startLogits = [startLogits];
    }

    if (isOneDimensional(endLogits)) {
      endLogits = [endLogits];
    }

    return [startLogits, endLogits];
  }

  static async fromOptions(options: ModelOptions): Promise<LocalModel> {
    const modelGraph = (await tf.node.getMetaGraphsFromSavedModel(options.path))[0];
    const fullParams = this.computeParams(options, modelGraph);

    const model = await tf.node.loadSavedModel(fullParams.path);
    return new LocalModel(model, fullParams);
  }
}

function isOneDimensional(arr: number[] | number[][]): arr is number[] {
  return !Array.isArray(arr[0]);
}
