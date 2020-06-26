import * as tf from "@tensorflow/tfjs-node";
import { TFSavedModel } from "@tensorflow/tfjs-node/dist/saved_model";

import { Logits, ModelInput } from "../models/model";
import { isOneDimensional } from "../utils";
import { Pool } from "./pool-real";
import { FullParams, Runtime, RuntimeOptions } from "./runtime";
// import { Pool } from "./pool";

export class SavedModel extends Runtime {
  private static pool = new Pool();
  // private static worker = new SavedModelWorker();
  // private worker: SavedModelWorker;

  private constructor(params: Readonly<FullParams>) {
    super(params);
    // SavedModel.worker.loadModel(params);
    // this.worker = new SavedModelWorker(params);
    SavedModel.pool.queueTask({ type: "load", params });
  }

  async runInference(
    ids: number[][],
    attentionMask: number[][],
    tokenTypeIds?: number[][]
  ): Promise<[Logits, Logits]> {
    // const result = tf.tidy(() => {
    //   const batchIds = [
    //     ids[0],
    //     ids[0],
    //     ids[0],
    //     ids[0],
    //     ids[0],
    //     ids[0],
    //     ids[0],
    //     ids[0],
    //     ids[0],
    //     ids[0]
    //   ];
    //   const batchMask = [
    //     attentionMask[0],
    //     attentionMask[0],
    //     attentionMask[0],
    //     attentionMask[0],
    //     attentionMask[0],
    //     attentionMask[0],
    //     attentionMask[0],
    //     attentionMask[0],
    //     attentionMask[0],
    //     attentionMask[0]
    //   ];

    //   const inputTensor = tf.tensor(batchIds, undefined, "int32");
    //   const maskTensor = tf.tensor(batchMask, undefined, "int32");

    //   const modelInputs = {
    //     [this.params.inputsNames[ModelInput.Ids]]: inputTensor,
    //     [this.params.inputsNames[ModelInput.AttentionMask]]: maskTensor
    //   };

    //   if (tokenTypeIds && this.params.inputsNames[ModelInput.TokenTypeIds]) {
    //     const batchTypes = [
    //       tokenTypeIds[0],
    //       tokenTypeIds[0],
    //       tokenTypeIds[0],
    //       tokenTypeIds[0],
    //       tokenTypeIds[0],
    //       tokenTypeIds[0],
    //       tokenTypeIds[0],
    //       tokenTypeIds[0],
    //       tokenTypeIds[0],
    //       tokenTypeIds[0]
    //     ];
    //     const tokenTypesTensor = tf.tensor(batchTypes, undefined, "int32");
    //     // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
    //     modelInputs[this.params.inputsNames[ModelInput.TokenTypeIds]!] = tokenTypesTensor;
    //   }

    //   return this.model.predict(modelInputs) as tf.NamedTensorMap;
    // });

    // let [startLogits, endLogits] = await Promise.all([
    //   result[this.params.outputsNames.startLogits].squeeze().array() as Promise<
    //     number[] | number[][]
    //   >,
    //   result[this.params.outputsNames.endLogits].squeeze().array() as Promise<
    //     number[] | number[][]
    //   >
    // ]);

    // tf.dispose(result);

    // if (isOneDimensional(startLogits)) {
    //   startLogits = [startLogits];
    // }

    // if (isOneDimensional(endLogits)) {
    //   endLogits = [endLogits];
    // }

    // return [startLogits, endLogits];

    // return this.worker.queueInference(
    return SavedModel.pool.queueTask({
      type: "infer",
      path: this.params.path,
      inputs: {
        ids,
        attentionMask,
        tokenTypeIds
      }
      // this.params.path,
      // ids,
      // attentionMask,
      // tokenTypeIds
    });
  }

  static async fromOptions(options: RuntimeOptions): Promise<SavedModel> {
    const modelGraph = (await tf.node.getMetaGraphsFromSavedModel(options.path))[0];
    const fullParams = this.computeParams(options, modelGraph);

    // const model = await tf.node.loadSavedModel(options.path);
    // const worker = await SavedModelWorker.fromOptions(fullParams);
    return new SavedModel(fullParams);
  }
}
