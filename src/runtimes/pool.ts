import * as tf from "@tensorflow/tfjs-node";
import { resolve } from "dns";
import { EventEmitter } from "events";
import { join as joinPaths } from "path";
import {
  MessageChannel,
  MessagePort,
  receiveMessageOnPort,
  SHARE_ENV,
  Worker
} from "worker_threads";

import { Logits } from "../models/model";
import { ROOT_DIR } from "../qa-options";
import { FullParams } from "./runtime";
import { InferenceMessage, LoadingStatus, Message, WorkerParams } from "./worker-message";

interface InferenceTask {
  resolve: (value?: [Logits, Logits] | PromiseLike<[Logits, Logits]>) => void;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  reject: (reason?: any) => void;
  inputs: {
    ids: number[][];
    attentionMask: number[][];
    tokenTypeIds?: number[][];
  };
}

export class SavedModelWorker extends Worker {
  //extends Worker {
  // public port: MessagePort;

  private loaded?: true;
  // private initPort: MessagePort;
  private queue: InferenceTask[] = [];

  constructor(params: FullParams) {
    super(joinPaths(ROOT_DIR, "runtimes/thread.js"), {
      env: SHARE_ENV,
      workerData: params
    });
    // const { port1, port2 } = new MessageChannel();
    const initChannel = new MessageChannel();
    // this.port = port2;

    const initPort = initChannel.port2;
    initPort.once("message", ({ status }: LoadingStatus) => {
      console.log("initPort", status, this.threadId);
      if (status === "loaded") {
        this.loaded = true;
        if (this.queue.length) {
          this.run();
        }
      }
      // TODO: better handling in case of error during init
    });
    this.postMessage({ type: "init", initPort: initChannel.port1, params }, [
      initChannel.port1
    ]);
  }

  // static fromOptions(params: FullParams): Promise<SavedModelWorker> {
  //   const worker = new Worker(joinPaths(ROOT_DIR, "runtimes/thread.js"), {
  //     env: SHARE_ENV,
  //     workerData: params
  //   });

  //   // const { port1, port2 } = new MessageChannel();
  //   const initChannel = new MessageChannel();
  //   // worker.port = port2;
  //   // worker.initPort = initChannel.port2;

  //   return new Promise((resolve, reject) => {
  //     initChannel.port2.once("message", ({ status }: LoadingStatus) => {
  //       console.log("initPort", status, worker.threadId);

  //       if (status === "loaded") {
  //         const savedModelWorker = new SavedModelWorker(worker);
  //         resolve(savedModelWorker);
  //       } else {
  //         reject();
  //       }

  //       // TODO: better handling in case of error during init
  //     });

  //     worker.postMessage({ type: "init", initPort: initChannel.port1, params }, [
  //       initChannel.port1
  //     ]);
  //   });
  // }

  // loaded(): Promise<void> {
  //   console.log("waiting for load...");

  //   if (this._loaded) {
  //     console.log("already loaded");

  //     return Promise.resolve();
  //   }

  //   const status = receiveMessageOnPort(this.initPort) as
  //     | { message: LoadingStatus }
  //     | undefined;
  //   console.log("status", status);

  //   if (status) {
  //     if (status.message.status === "loaded") {
  //       this._loaded = true;
  //       return Promise.resolve();
  //     } else {
  //       // TODO: If failed loading
  //       return Promise.reject();
  //     }
  //   } else {
  //     return new Promise((resolve, reject) => {
  //       this.initPort.once("message", ({ status }: LoadingStatus) => {
  //         console.log("initPort", status, this.threadId);

  //         if (status === "loaded") {
  //           this._loaded = true;
  //           if (this.queue.length)
  //           resolve();
  //         }

  //         // TODO: If failed loading
  //       });
  //     });
  //   }
  // }

  queueInference(
    ids: number[][],
    attentionMask: number[][],
    tokenTypeIds?: number[][]
  ): Promise<[Logits, Logits]> {
    return new Promise<[Logits, Logits]>((resolve, reject) => {
      const inferenceTask: InferenceTask = {
        resolve,
        reject,
        inputs: { ids, attentionMask, tokenTypeIds }
      };

      this.queue.push(inferenceTask);
      if (this.queue.length === 1) {
        this.run();
      }
    });
  }

  private run(): void {
    const task = this.queue[0];
    this.once("message", data => {
      if (data.logits) {
        task.resolve(data.logits);
      } else {
        task.reject(data.error);
      }

      this.queue.shift();
      if (this.queue.length) {
        this.run();
      }
    });

    this.postMessage({
      type: "infer",
      data: task.inputs
    });
  }
}

// interface WorkerMessage {
//   _id: number;
//   // eslint-disable-next-line @typescript-eslint/no-explicit-any
//   error?: any;
//   logits: [Logits, Logits];
// }

// interface PoolOptions {
//   // eslint-disable-next-line @typescript-eslint/no-explicit-any
//   errorHandler?: (error: any) => void;
//   exitHandler?: (code: number) => void;
//   onlineHandler?: () => void;
//   /**
//    * @default 1000
//    */
//   maxTasks?: number;
//   /**
//    * @default 1
//    */
//   minThreads?: number;
//   /**
//    * @default 5
//    */
//   maxThreads?: number;
// }

// export class Pool {
//   private _id = 0;

//   private emitter = new EventEmitter();
//   private nextWorker = 0;
//   private tasks = new Map<SavedModelWorker, number>();
//   private workers = new Map<string, SavedModelWorker>();

//   // constructor(private workerParams: FullParams, private opts?: PoolOptions) {
//   constructor(private opts?: PoolOptions) {
//     // for (let i = 0; i < (opts?.minThreads ?? 1); i++) {
//     //   this.newWorker();
//     // }
//   }

//   async destroy(): Promise<void> {
//     for (const [_, worker] of this.workers) {
//       await worker.terminate();
//     }
//   }

//   async runInference(
//     path: string,
//     ids: number[][],
//     attentionMask: number[][],
//     tokenTypeIds?: number[][]
//   ): Promise<[Logits, Logits]> {
//     // configure worker to handle message with the specified task
//     const worker = this.getWorker(path);
//     // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
//     this.tasks.set(worker, this.tasks.get(worker)! + 1);
//     const id = ++this._id;
//     console.log("num models in ctor", tf.node.getNumOfSavedModels());
//     const res = this.execute(worker, id);
//     worker.postMessage({
//       type: "infer",
//       data: { ids, attentionMask, tokenTypeIds },
//       _id: id
//     });
//     return res;
//   }

//   private async execute(worker: SavedModelWorker, id: number): Promise<[Logits, Logits]> {
//     await worker.loaded();
//     return new Promise((resolve, reject) => {
//       const listener = (message: WorkerMessage): void => {
//         if (message._id === id) {
//           worker.port.removeListener("message", listener);
//           // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
//           this.tasks.set(worker, this.tasks.get(worker)! - 1);
//           if (message.error) reject(message.error);
//           else resolve(message.logits);
//         }
//       };

//       worker.port.on("message", listener);
//     });
//   }

//   queueTask(
//     path: string,
//     ids: number[][],
//     attentionMask: number[][],
//     tokenTypeIds?: number[][]
//   ): Promise<[Logits, Logits]> {
//     const worker = this.workers.get(path)!;
//   }

//   private getWorker(path: string): SavedModelWorker {
//     const worker = this.workers.get(path)!;

//     if (this.tasks.get(worker) === 0) {
//       return worker;
//     } else {
//     }

//     let worker;
//     for (const entry of this.tasks) {
//       if (entry[1] === 0) {
//         worker = entry[0];
//         break;
//       }
//     }

//     if (worker) {
//       // a worker is free, use it
//       return worker;
//     } else {
//       if (this.workers.length === (this.opts?.maxThreads ?? 5)) {
//         this.emitter.emit("FullPool");
//         if (this.workers.length - 1 === this.nextWorker) {
//           this.nextWorker = 0;
//           return this.workers[this.nextWorker];
//         } else {
//           this.nextWorker++;
//           return this.workers[this.nextWorker];
//         }
//       }

//       // all workers are busy create a new worker
//       const worker = this.addWorker();
//       worker.port.on("message", message => {
//         if (message.kill) {
//           worker.postMessage({ type: "kill" });
//           worker.terminate();
//         }
//       });

//       return worker;
//     }
//   }

//   addWorker(params: FullParams): SavedModelWorker {
//     const worker = new SavedModelWorker(params);
//     this.opts?.errorHandler && worker.on("error", this.opts.errorHandler);
//     this.opts?.onlineHandler && worker.on("online", this.opts.onlineHandler);
//     // TODO handle properly when a thread exit
//     this.opts?.exitHandler && worker.on("exit", this.opts.exitHandler);

//     this.workers.set(params.path, worker);

//     // we will attach a listener for every task,
//     // when task is completed the listener will be removed
//     // but to avoid warnings we are increasing the max listeners size
//     worker.port.setMaxListeners(this.opts?.maxTasks ?? 1000);

//     this.tasks.set(worker, 0);
//     return worker;
//   }
// }
