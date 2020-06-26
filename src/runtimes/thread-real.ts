import * as tf from "@tensorflow/tfjs-node";
import { TFSavedModel } from "@tensorflow/tfjs-node/dist/saved_model";
import { AsyncResource } from "async_hooks";
import { MessagePort, parentPort, threadId, workerData } from "worker_threads";

import { Logits, ModelInput } from "../models/model";
import { isOneDimensional } from "../utils";
import { InferenceWorkerTask, PoolTask } from "./pool-real";
import { FullParams } from "./runtime";

// interface Options {
//   maxInactiveTime: number;
// }

// interface Value {
//   _id: number;
//   model: string;
//   inputs: {
//     ids: number[][];
//     attentionMask: number[][];
//     tokenTypeIds?: number[][];
//   };
// }

// let loadPort: MessagePort;
// let inferencePort: MessagePort;
// let model: TFSavedModel;
// const params: FullParams = workerData;

const modelsMap = new Map<string, TFSavedModel>();
// const modelParamsMap = new Map<string, FullParams>();

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

parentPort?.on("message", async (value: PoolTask) => {
  try {
    switch (value.type) {
      case "infer":
        await runTask(value);
        // this.runInAsyncScope(this.runTask.bind(this), this, value);
        break;

      // case "init":
      //   loadPort = value.loadPort;
      //   inferencePort = value.inferencePort;
      //   // value.initPort.postMessage({ status: "loaded" });
      //   value.initPort.close();
      //   break;

      case "load":
        await initModel(value.params);
        break;

      // case "kill":
      //   // here is time to kill this thread, just clearing the interval
      //   // clearInterval(this.interval);
      //   this.emitDestroy();
      //   break;
    }
  } catch (error) {
    parentPort?.postMessage({ type: value.type, error });
  }
});
// }

async function initModel(params: FullParams): Promise<void> {
  console.log("init model...", threadId);

  console.log("num models before init", threadId, tf.node.getNumOfSavedModels());

  modelsMap.set(params.path, await tf.node.loadSavedModel(params.path));
  // modelParamsMap.set(params.path, params);

  console.log("num models after init", threadId, tf.node.getNumOfSavedModels());
  parentPort?.postMessage({ type: "load" });
}

// private checkAlive(): void {
//   if (Date.now() - this.lastTask > this.maxInactiveTime) {
//     this.parent?.postMessage({ kill: 1 });
//   }
// }

async function getModel(params: FullParams): Promise<TFSavedModel> {
  let model = modelsMap.get(params.path);
  if (!model) {
    model = await tf.node.loadSavedModel(params.path);
    modelsMap.set(params.path, model);
  }

  return model;
}

async function runTask(value: InferenceWorkerTask): Promise<void> {
  const { ids, attentionMask, tokenTypeIds } = value.inputs;
  // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
  const logits = await runInference(value.params!, ids, attentionMask, tokenTypeIds);
  parentPort?.postMessage({ type: "infer", data: logits });
  // this.lastTask = Date.now();
}

async function runInference(
  modelParams: FullParams,
  ids: number[][],
  attentionMask: number[][],
  tokenTypeIds?: number[][]
): Promise<[Logits, Logits]> {
  const model = await getModel(modelParams);

  const result = tf.tidy(() => {
    const inputTensor = tf.tensor(ids, undefined, "int32");
    const maskTensor = tf.tensor(attentionMask, undefined, "int32");

    const modelInputs = {
      [modelParams.inputsNames[ModelInput.Ids]]: inputTensor,
      [modelParams.inputsNames[ModelInput.AttentionMask]]: maskTensor
    };

    if (tokenTypeIds && modelParams.inputsNames[ModelInput.TokenTypeIds]) {
      const tokenTypesTensor = tf.tensor(tokenTypeIds, undefined, "int32");
      // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
      modelInputs[modelParams.inputsNames[ModelInput.TokenTypeIds]!] = tokenTypesTensor;
    }

    return model?.predict(modelInputs) as tf.NamedTensorMap;
  });

  let [startLogits, endLogits] = await Promise.all([
    result[modelParams.outputsNames.startLogits].squeeze().array() as Promise<
      number[] | number[][]
    >,
    result[modelParams.outputsNames.endLogits].squeeze().array() as Promise<
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
