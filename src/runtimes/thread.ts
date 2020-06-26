import * as tf from "@tensorflow/tfjs-node";
import { TFSavedModel } from "@tensorflow/tfjs-node/dist/saved_model";
import { AsyncResource } from "async_hooks";
import { MessagePort, parentPort, threadId, workerData } from "worker_threads";

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

let initPort: MessagePort;
let model: TFSavedModel;
const params: FullParams = workerData;

// class SavedModelThread extends AsyncResource {
//   // private interval: NodeJS.Timeout;
//   private initPort?: MessagePort;
//   private lastTask: number;
//   private maxInactiveTime: number;
//   private model?: TFSavedModel;
//   private params!: FullParams;
//   // private parent?: MessagePort;

//   constructor(opts?: Options) {
//     super("worker-thread-pool:pioardi");
//     // this.opts = opts;
//     this.maxInactiveTime = opts?.maxInactiveTime ?? 1000 * 60;
//     this.lastTask = Date.now();
// this.params = opts.params;

// keep the worker active
// this.interval = setInterval(this.checkAlive.bind(this), this.maxInactiveTime / 2);
// this.checkAlive.bind(this)();

parentPort?.on("message", value => {
  switch (value.type) {
    case "infer":
      runTask(value);
      // this.runInAsyncScope(this.runTask.bind(this), this, value);
      break;

    case "init":
      // this.parent = value.parent;
      initPort = value.initPort;
      // this.params = value.params;
      initModel(value.params.path);
      break;

    // case "kill":
    //   // here is time to kill this thread, just clearing the interval
    //   // clearInterval(this.interval);
    //   this.emitDestroy();
    //   break;
  }
});
// }

async function initModel(path: string): Promise<void> {
  console.log("init model...", threadId);

  console.log("num models before init", threadId, tf.node.getNumOfSavedModels());
  model = await tf.node.loadSavedModel(path);
  console.log("num models after init", threadId, tf.node.getNumOfSavedModels());
  initPort?.postMessage({ status: "loaded" });
  initPort?.close();
}

// private checkAlive(): void {
//   if (Date.now() - this.lastTask > this.maxInactiveTime) {
//     this.parent?.postMessage({ kill: 1 });
//   }
// }

async function runTask(value: Value): Promise<void> {
  try {
    const { ids, attentionMask, tokenTypeIds } = value.data;
    const logits = await runInference(ids, attentionMask, tokenTypeIds);
    parentPort?.postMessage({ logits, _id: value._id });
    // this.lastTask = Date.now();
  } catch (e) {
    parentPort?.postMessage({ error: e, _id: value._id });
    // this.lastTask = Date.now();
  }
}

async function runInference(
  ids: number[][],
  attentionMask: number[][],
  tokenTypeIds?: number[][]
): Promise<[Logits, Logits]> {
  // console.log("num models in infer", threadId, tf.node.getNumOfSavedModels());

  const result = tf.tidy(() => {
    const inputTensor = tf.tensor(ids, undefined, "int32");
    const maskTensor = tf.tensor(attentionMask, undefined, "int32");

    const modelInputs = {
      [params.inputsNames[ModelInput.Ids]]: inputTensor,
      [params.inputsNames[ModelInput.AttentionMask]]: maskTensor
    };

    if (tokenTypeIds && params.inputsNames[ModelInput.TokenTypeIds]) {
      const tokenTypesTensor = tf.tensor(tokenTypeIds, undefined, "int32");
      // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
      modelInputs[params.inputsNames[ModelInput.TokenTypeIds]!] = tokenTypesTensor;
    }

    return model?.predict(modelInputs) as tf.NamedTensorMap;
  });

  let [startLogits, endLogits] = await Promise.all([
    result[params.outputsNames.startLogits].squeeze().array() as Promise<
      number[] | number[][]
    >,
    result[params.outputsNames.endLogits].squeeze().array() as Promise<
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
// }

// export default new SavedModelThread();
