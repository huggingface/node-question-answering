import * as tf from "@tensorflow/tfjs-node";
import { TFSavedModel } from "@tensorflow/tfjs-node/dist/saved_model";

import { Logits, ModelInput } from "../models/model";
import { isOneDimensional } from "../utils";
import { FullParams, Runtime, RuntimeOptions } from "./runtime";

export class SavedModel extends Runtime {
  private constructor(private model: TFSavedModel, params: Readonly<FullParams>) {
    super(params);
  }

  async runInference(
    ids: number[][],
    attentionMask: number[][],
    tokenTypeIds?: number[][]
  ): Promise<[Logits, Logits]> {
    const result = tf.tidy(() => {
      const inputTensor = tf.tensor(ids, undefined, "int32");
      const maskTensor = tf.tensor(attentionMask, undefined, "int32");

      const modelInputs = {
        [this.params.inputsNames[ModelInput.Ids]]: inputTensor,
        [this.params.inputsNames[ModelInput.AttentionMask]]: maskTensor
      };

      if (tokenTypeIds && this.params.inputsNames[ModelInput.TokenTypeIds]) {
        const tokenTypesTensor = tf.tensor(tokenTypeIds, undefined, "int32");
        // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
        modelInputs[this.params.inputsNames[ModelInput.TokenTypeIds]!] = tokenTypesTensor;
      }

      return this.model.predict(modelInputs) as tf.NamedTensorMap;
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

  static async fromOptions(options: RuntimeOptions): Promise<SavedModel> {
    const modelGraph = (await tf.node.getMetaGraphsFromSavedModel(options.path))[0];
    const fullParams = this.computeParams(options, modelGraph);

    const model = await tf.node.loadSavedModel(options.path);
    return new SavedModel(model, fullParams);
  }
}
