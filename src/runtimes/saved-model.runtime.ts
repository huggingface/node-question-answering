import * as tf from "@tensorflow/tfjs-node";
import { Tensor } from "@tensorflow/tfjs-node";
import {
  getInputAndOutputNodeNameFromMetaGraphInfo,
  TFSavedModel
} from "@tensorflow/tfjs-node/dist/saved_model";
import { join as joinPaths } from "path";
import {
  MessageChannel,
  MessagePort,
  receiveMessageOnPort,
  SHARE_ENV,
  Worker
} from "worker_threads";

import { Logits, ModelInput } from "../models/model";
import { ROOT_DIR } from "../qa-options";
import { isOneDimensional } from "../utils";
import { Pool } from "./pool";
import { FullParams, Runtime, RuntimeOptions } from "./runtime";
import { LoadingStatus, Message, WorkerParams } from "./worker-message";

enum WorkerStatus {
  Idle = "idle",
  Busy = "busy"
}

// class SavedModelWorker {
//   private _initPort: MessagePort;
//   private _loaded?: true;
//   private _status: WorkerStatus;
//   private _worker: Worker;

//   // constructor(params: WorkerParams, sessionId: number, status = WorkerStatus.Idle) {
//   constructor(params: WorkerParams, status = WorkerStatus.Idle) {
//     console.log("num models in ctor", tf.node.getNumOfSavedModels());
//     this._status = status;

//     const initChannel = new MessageChannel();
//     this._worker = new Worker(joinPaths(ROOT_DIR, "runtimes/saved-model.worker.js"), {
//       env: SHARE_ENV
//     });

//     this._initPort = initChannel.port2;
//     this.postMessage(
//       { messageType: "load", initPort: initChannel.port1, params, sessionId: 0 },
//       [initChannel.port1]
//     );
//   }

//   get status(): Readonly<WorkerStatus> {
//     return this._status;
//   }

//   async runInference(
//     ids: number[][],
//     attentionMask: number[][],
//     tokenTypeIds?: number[][]
//   ): Promise<[Logits, Logits]> {
//     this._status = WorkerStatus.Busy;
//     await this.loaded();

//     const logits = new Promise<[Logits, Logits]>((resolve, reject) => {
//       this._worker.once("message", logits => {
//         this._status = WorkerStatus.Idle;
//         resolve(logits);
//       });
//     });

//     this.postMessage({ ids, attentionMask, tokenTypeIds, messageType: "infer" });
//     return logits;
//   }

//   private postMessage(m: Message, transferList?: (ArrayBuffer | MessagePort)[]): void {
//     this._worker.postMessage(m, transferList);
//   }

//   private loaded(): Promise<void> {
//     console.log("waiting for load...");

//     if (this._loaded) {
//       return Promise.resolve();
//     }

//     const status = receiveMessageOnPort(this._initPort) as
//       | { message: LoadingStatus }
//       | undefined;
//     console.log("status", status);

//     if (status) {
//       if (status.message.status === "loaded") {
//         this._loaded = true;
//         return Promise.resolve();
//       } else {
//         // TODO: If failed loading
//         return Promise.reject();
//       }
//     } else {
//       return new Promise((resolve, reject) => {
//         this._initPort.once("message", ({ status }: LoadingStatus) => {
//           console.log("initPort", status);

//           if (status === "loaded") {
//             this._loaded = true;
//             resolve();
//           }

//           // TODO: If failed loading
//         });
//       });
//     }
//   }
// }

export class SavedModel extends Runtime {
  // private pool: Pool;
  // private workerParams: WorkerParams;

  private constructor(
    params: Readonly<FullParams>,
    private model: TFSavedModel // tfInputsOutputs: Record<string, string>[], // private maxWorkers = 5
  ) {
    super(params);

    // this.workerParams = {
    //   inputsNames: params.inputsNames,
    //   outputsNames: params.outputsNames,
    //   tfInputsNames: tfInputsOutputs[0],
    //   tfOutputsNames: tfInputsOutputs[1],
    //   path: params.path
    // };

    // this.pool = new Pool(params);

    // const firstWorker = new SavedModelWorker(
    //   this.workerParams,
    //   // (this.model as any).sessionId
    // );
    // this.pool.push(firstWorker);
  }

  async runInference(
    ids: number[][],
    attentionMask: number[][],
    tokenTypeIds?: number[][]
  ): Promise<[Logits, Logits]> {
    // return this.pool.runInference(ids, attentionMask, tokenTypeIds);
    // let idleWorker = this.pool.find(w => w.status === WorkerStatus.Idle);

    // if (!idleWorker) {
    //   if (this.pool.length < this.maxWorkers) {
    //     console.log("adding worker");
    //     idleWorker = new SavedModelWorker(
    //       this.workerParams,
    //       // (this.model as any).sessionId
    //     );
    //     this.pool.push(idleWorker);
    //   } else {
    //     let counter = 0;
    //     do {
    //       counter++;
    //       await new Promise(r => setTimeout(r, 200));
    //       idleWorker = this.pool.find(w => w.status === WorkerStatus.Idle);
    //     } while (!idleWorker && counter < 25);

    //     if (!idleWorker) {
    //       throw new Error("Unable to find an idle worker");
    //     }
    //   }
    // }

    // console.log(this.pool.length);
    // return idleWorker.runInference(ids, attentionMask, tokenTypeIds);

    const result = await new Promise<tf.NamedTensorMap>((resolve, reject) => {
      // const result = tf.tidy(() => {
      const inputTensor = tf.tensor(ids, undefined, "int32");
      const maskTensor = tf.tensor(attentionMask, undefined, "int32");

      const modelInputs = {
        [this.params.inputsNames[ModelInput.Ids]]: inputTensor,
        [this.params.inputsNames[ModelInput.AttentionMask]]: maskTensor
      };

      let tokenTypesTensor: Tensor | undefined;
      if (tokenTypeIds && this.params.inputsNames[ModelInput.TokenTypeIds]) {
        tokenTypesTensor = tf.tensor(tokenTypeIds, undefined, "int32");
        // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
        modelInputs[this.params.inputsNames[ModelInput.TokenTypeIds]!] = tokenTypesTensor;
      }

      // return this.model.predict(modelInputs) as tf.NamedTensorMap;
      this.model.predict(modelInputs, r => {
        // console.log("Inside callback");

        tf.dispose(inputTensor);
        tf.dispose(maskTensor);
        tokenTypesTensor && tf.dispose(tokenTypesTensor);

        resolve(r as tf.NamedTensorMap);
      });
    });
    // });

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

    // const tfInputsOutputs = getInputAndOutputNodeNameFromMetaGraphInfo(
    //   [modelGraph],
    //   ["serve"],
    //   fullParams.signatureName
    // );

    const model = await tf.node.loadSavedModel(options.path);
    return new SavedModel(fullParams, model);
  }
}
