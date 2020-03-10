import * as tf from "@tensorflow/tfjs-node";
import { TFSavedModel } from "@tensorflow/tfjs-node/dist/saved_model";
import { exists } from "fs";
import * as path from "path";
import { promisify } from "util";

import { DEFAULT_ASSETS_PATH } from "../qa-options";
import { isOneDimensional, Model, ModelOptions, ModelParams } from "./model";

export class SavedModel extends Model {
  private constructor(private model: TFSavedModel, public params: ModelParams) {
    super();
  }

  async runInference(
    ids: number[][],
    attentionMask: number[][]
  ): Promise<[number[][], number[][]]> {
    const result = tf.tidy(() => {
      const inputTensor = tf.tensor(ids, undefined, "int32");
      const maskTensor = tf.tensor(attentionMask, undefined, "int32");

      return this.model.predict({
        [this.params.inputsNames.ids]: inputTensor,
        [this.params.inputsNames.attentionMask]: maskTensor
      }) as tf.NamedTensorMap;
    });

    let [startLogits, endLogits] = await Promise.all([
      result[this.params.outputsNames.startLogits].squeeze().array() as Promise<
        number[] | number[][]
      >,
      result[this.params.outputsNames.endLogits].squeeze().array() as Promise<
        number[] | number[][]
      >
    ]);

    tf.dispose(result);

    if (isOneDimensional(startLogits)) {
      startLogits = [startLogits];
    }

    if (isOneDimensional(endLogits)) {
      endLogits = [endLogits];
    }

    return [startLogits, endLogits];
  }

  static async fromOptions(options: ModelOptions): Promise<SavedModel> {
    options.path = (await promisify(exists)(options.path))
      ? options.path
      : path.join(DEFAULT_ASSETS_PATH, options.path);

    const modelGraph = (await tf.node.getMetaGraphsFromSavedModel(options.path))[0];
    const fullParams = this.computeParams(options, modelGraph);

    const model = await tf.node.loadSavedModel(fullParams.path);
    return new SavedModel(model, fullParams);
  }
}
