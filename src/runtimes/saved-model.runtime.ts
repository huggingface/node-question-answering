import * as tf from "@tensorflow/tfjs-node";

import { Logits } from "../models/model";
import { FullParams, Runtime, RuntimeOptions } from "./runtime";
import { SavedModelWorker } from "./saved-model.worker";

export class SavedModel extends Runtime {
  private static worker = new SavedModelWorker();

  private constructor(params: Readonly<FullParams>) {
    super(params);
  }

  async runInference(
    ids: number[][],
    attentionMask: number[][],
    tokenTypeIds?: number[][]
  ): Promise<[Logits, Logits]> {
    return SavedModel.worker.queueInference(
      this.params.path,
      ids,
      attentionMask,
      tokenTypeIds
    );
  }

  static async fromOptions(options: RuntimeOptions): Promise<SavedModel> {
    const modelGraph = (await tf.node.getMetaGraphsFromSavedModel(options.path))[0];
    const fullParams = this.computeParams(options, modelGraph);

    await SavedModel.worker.loadModel(fullParams);
    return new SavedModel(fullParams);
  }
}
