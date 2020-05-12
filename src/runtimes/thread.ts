import * as tf from "@tensorflow/tfjs-node";
import { TFSavedModel } from "@tensorflow/tfjs-node/dist/saved_model";
import { AsyncResource } from "async_hooks";
import { MessagePort, parentPort, threadId } from "worker_threads";

import { Logits, ModelInput } from "../models/model";
import { isOneDimensional } from "../utils";
import { FullParams } from "./runtime";

interface Options {
  maxInactiveTime: number;
}

interface Value {
  _id: number;
  data: {
    ids: number[][];
    attentionMask: number[][];
    tokenTypeIds?: number[][];
  };
}

class SavedModelThread extends AsyncResource {
  // private interval: NodeJS.Timeout;
  private initPort?: MessagePort;
  private lastTask: number;
  private maxInactiveTime: number;
  private model?: TFSavedModel;
  private params!: FullParams;
  private parent?: MessagePort;

  constructor(opts?: Options) {
    super("worker-thread-pool:pioardi");
    // this.opts = opts;
    this.maxInactiveTime = opts?.maxInactiveTime ?? 1000 * 60;
    this.lastTask = Date.now();
    // this.params = opts.params;

    // keep the worker active
    // this.interval = setInterval(this.checkAlive.bind(this), this.maxInactiveTime / 2);
    // this.checkAlive.bind(this)();

    parentPort?.on("message", value => {
      switch (value.type) {
        case "infer":
          this.runInAsyncScope(this.runTask.bind(this), this, value);
          break;

        case "init":
          this.parent = value.parent;
          this.initPort = value.initPort;
          this.params = value.params;
          this.initModel(value.params.path);
          break;

        case "kill":
          // here is time to kill this thread, just clearing the interval
          // clearInterval(this.interval);
          this.emitDestroy();
          break;
      }
    });
  }

  private async initModel(path: string): Promise<void> {
    console.log("init model...", threadId);

    console.log("num models before init", threadId, tf.node.getNumOfSavedModels());
    this.model = await tf.node.loadSavedModel(path);
    console.log("num models after init", threadId, tf.node.getNumOfSavedModels());
    this.initPort?.postMessage({ status: "loaded" });
    this.initPort?.close();
  }

  // private checkAlive(): void {
  //   if (Date.now() - this.lastTask > this.maxInactiveTime) {
  //     this.parent?.postMessage({ kill: 1 });
  //   }
  // }

  private async runTask(value: Value): Promise<void> {
    try {
      const { ids, attentionMask, tokenTypeIds } = value.data;
      // const logits = await this.runInference(ids, attentionMask, tokenTypeIds);
      // this.parent?.postMessage({ logits, _id: value._id });
      this.lastTask = Date.now();
    } catch (e) {
      this.parent?.postMessage({ error: e, _id: value._id });
      this.lastTask = Date.now();
    }
  }

  // private async runInference(
  //   ids: number[][],
  //   attentionMask: number[][],
  //   tokenTypeIds?: number[][]
  // ): Promise<[Logits, Logits]> {
  //   console.log("num models in infer", threadId, tf.node.getNumOfSavedModels());

  //   const result = tf.tidy(() => {
  //     const inputTensor = tf.tensor(ids, undefined, "int32");
  //     const maskTensor = tf.tensor(attentionMask, undefined, "int32");

  //     const modelInputs = {
  //       [this.params.inputsNames[ModelInput.Ids]]: inputTensor,
  //       [this.params.inputsNames[ModelInput.AttentionMask]]: maskTensor
  //     };

  //     if (tokenTypeIds && this.params.inputsNames[ModelInput.TokenTypeIds]) {
  //       const tokenTypesTensor = tf.tensor(tokenTypeIds, undefined, "int32");
  //       // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
  //       modelInputs[this.params.inputsNames[ModelInput.TokenTypeIds]!] = tokenTypesTensor;
  //     }

  //     return this.model?.predict(modelInputs) as tf.NamedTensorMap;
  //   });

  //   let [startLogits, endLogits] = await Promise.all([
  //     result[this.params.outputsNames.startLogits].squeeze().array() as Promise<
  //       number[] | number[][]
  //     >,
  //     result[this.params.outputsNames.endLogits].squeeze().array() as Promise<
  //       number[] | number[][]
  //     >
  //   ]);

  //   tf.dispose(result);

  //   if (isOneDimensional(startLogits)) {
  //     startLogits = [startLogits];
  //   }

  //   if (isOneDimensional(endLogits)) {
  //     endLogits = [endLogits];
  //   }

  //   return [startLogits, endLogits];
  // }
}

export default new SavedModelThread();
