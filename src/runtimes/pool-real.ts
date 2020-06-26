import * as tf from "@tensorflow/tfjs-node";
import { any } from "@tensorflow/tfjs-node";
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
  type: "infer";
  path: string;
  params?: FullParams;
  inputs: {
    ids: number[][];
    attentionMask: number[][];
    tokenTypeIds?: number[][];
  };
}

interface ModelLoadingTask {
  type: "load";
  params: FullParams;
}

export type WorkerTask = InferenceTask | ModelLoadingTask;
export type WorkerTaskReturn<T extends WorkerTask> = T extends InferenceTask
  ? [Logits, Logits]
  : void;

interface WorkerPromise<TResult> {
  resolve: (value?: TResult | PromiseLike<TResult>) => void;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  reject: (reason?: any) => void;
}

export interface InferenceWorkerTask
  extends InferenceTask,
    WorkerPromise<WorkerTaskReturn<InferenceTask>> {}

export interface ModelLoadingWorkerTask
  extends ModelLoadingTask,
    WorkerPromise<WorkerTaskReturn<ModelLoadingTask>> {}

export type PoolTask = InferenceWorkerTask | ModelLoadingWorkerTask;

// type WorkerTask<T extends PoolTask = PoolTask> = T & WorkerPromise<PoolTaskReturn<T>>;

// interface InferenceTask {
//   type: "infer";
//   path: string;
//   resolve: (value?: [Logits, Logits] | PromiseLike<[Logits, Logits]>) => void;
//   // eslint-disable-next-line @typescript-eslint/no-explicit-any
//   reject: (reason?: any) => void;
//   inputs: {
//     ids: number[][];
//     attentionMask: number[][];
//     tokenTypeIds?: number[][];
//   };
// }

// interface ModelLoadingTask {
//   type: "load";
//   params: FullParams;
// }

// export class SavedModelWorker extends Worker {
//   //extends Worker {
//   // public port: MessagePort;
//   // public readonly id = SavedModelWorker.nextId++;
//   private loaded?: true;
//   // private initPort: MessagePort;
//   private queue: WorkerTask[] = [];

//   // private static nextId = 0;

//   constructor() {
//     super(joinPaths(ROOT_DIR, "runtimes/thread-real.js"), {
//       env: SHARE_ENV
//     });
//     // const { port1, port2 } = new MessageChannel();
//     const initChannel = new MessageChannel();
//     // this.port = port2;

//     const initPort = initChannel.port2;
//     initPort.once("message", ({ status }: LoadingStatus) => {
//       console.log("initPort", status, this.threadId);
//       if (status === "loaded") {
//         this.loaded = true;
//         if (this.queue.length) {
//           this.run();
//         }
//       }
//       // TODO: better handling in case of error during init
//     });
//     this.postMessage({ type: "init", initPort: initChannel.port1, params }, [
//       initChannel.port1
//     ]);
//   }

//   // static fromOptions(params: FullParams): Promise<SavedModelWorker> {
//   //   const worker = new Worker(joinPaths(ROOT_DIR, "runtimes/thread.js"), {
//   //     env: SHARE_ENV,
//   //     workerData: params
//   //   });

//   //   // const { port1, port2 } = new MessageChannel();
//   //   const initChannel = new MessageChannel();
//   //   // worker.port = port2;
//   //   // worker.initPort = initChannel.port2;

//   //   return new Promise((resolve, reject) => {
//   //     initChannel.port2.once("message", ({ status }: LoadingStatus) => {
//   //       console.log("initPort", status, worker.threadId);

//   //       if (status === "loaded") {
//   //         const savedModelWorker = new SavedModelWorker(worker);
//   //         resolve(savedModelWorker);
//   //       } else {
//   //         reject();
//   //       }

//   //       // TODO: better handling in case of error during init
//   //     });

//   //     worker.postMessage({ type: "init", initPort: initChannel.port1, params }, [
//   //       initChannel.port1
//   //     ]);
//   //   });
//   // }

//   // loaded(): Promise<void> {
//   //   console.log("waiting for load...");

//   //   if (this._loaded) {
//   //     console.log("already loaded");

//   //     return Promise.resolve();
//   //   }

//   //   const status = receiveMessageOnPort(this.initPort) as
//   //     | { message: LoadingStatus }
//   //     | undefined;
//   //   console.log("status", status);

//   //   if (status) {
//   //     if (status.message.status === "loaded") {
//   //       this._loaded = true;
//   //       return Promise.resolve();
//   //     } else {
//   //       // TODO: If failed loading
//   //       return Promise.reject();
//   //     }
//   //   } else {
//   //     return new Promise((resolve, reject) => {
//   //       this.initPort.once("message", ({ status }: LoadingStatus) => {
//   //         console.log("initPort", status, this.threadId);

//   //         if (status === "loaded") {
//   //           this._loaded = true;
//   //           if (this.queue.length)
//   //           resolve();
//   //         }

//   //         // TODO: If failed loading
//   //       });
//   //     });
//   //   }
//   // }

//   // queueInference(
//   //   ids: number[][],
//   //   attentionMask: number[][],
//   //   tokenTypeIds?: number[][]
//   // ): Promise<[Logits, Logits]> {
//   //   return new Promise<[Logits, Logits]>((resolve, reject) => {
//   //     const inferenceTask: InferenceTask = {
//   //       resolve,
//   //       reject,
//   //       inputs: { ids, attentionMask, tokenTypeIds }
//   //     };

//   //     this.queue.push(inferenceTask);
//   //     if (this.queue.length === 1) {
//   //       this.run();
//   //     }
//   //   });
//   // }

//   // private run(): void {
//   //   const task = this.queue[0];
//   //   this.once("message", data => {
//   //     if (data.logits) {
//   //       task.resolve(data.logits);
//   //     } else {
//   //       task.reject(data.error);
//   //     }

//   //     this.queue.shift();
//   //     if (this.queue.length) {
//   //       this.run();
//   //     }
//   //   });

//   //   this.postMessage({
//   //     type: "infer",
//   //     data: task.inputs
//   //   });
//   // }
// }

// interface WorkerMessage {
//   _id: number;
//   // eslint-disable-next-line @typescript-eslint/no-explicit-any
//   error?: any;
//   logits: [Logits, Logits];
// }

interface BaseWorkerMessage {
  error?: any;
}

interface ModelLoadedWorkerMessage extends BaseWorkerMessage {
  type: "load";
  // data: string;
}
interface InferredWorkerMessage extends BaseWorkerMessage {
  type: "infer";
  data: [Logits, Logits];
}

type WorkerMessage = ModelLoadedWorkerMessage | InferredWorkerMessage;

interface PoolOptions {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  errorHandler?: (error: any) => void;
  exitHandler?: (code: number) => void;
  onlineHandler?: () => void;
  /**
   * @default 1000
   */
  maxTasks?: number;
  /**
   * @default 1
   */
  minThreads?: number;
  /**
   * @default 5
   */
  maxThreads?: number;
}

export class Pool {
  // private _id = 0;

  // private emitter = new EventEmitter();
  // private nextWorker = 0;
  // private tasks = new Map<SavedModelWorker, number>();
  private models = new Map<string, FullParams>();
  private workers: [Worker, boolean][] = [];
  private queue: PoolTask[] = [];

  // constructor(private workerParams: FullParams, private opts?: PoolOptions) {
  constructor(private opts?: PoolOptions) {
    for (let i = 0; i < (opts?.minThreads ?? 1); i++) {
      this.addWorker();
    }
  }

  async destroy(): Promise<void> {
    for (const [worker] of this.workers) {
      await worker.terminate();
    }
  }

  // async runInference(
  //   path: string,
  //   ids: number[][],
  //   attentionMask: number[][],
  //   tokenTypeIds?: number[][]
  // ): Promise<[Logits, Logits]> {
  //   // configure worker to handle message with the specified task
  //   const worker = this.getWorker(path);
  //   // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
  //   this.tasks.set(worker, this.tasks.get(worker)! + 1);
  //   const id = ++this._id;
  //   console.log("num models in ctor", tf.node.getNumOfSavedModels());
  //   const res = this.execute(worker, id);
  //   worker.postMessage({
  //     type: "infer",
  //     data: { ids, attentionMask, tokenTypeIds },
  //     _id: id
  //   });
  //   return res;
  // }

  // private async execute(worker: SavedModelWorker, id: number): Promise<[Logits, Logits]> {
  //   await worker.loaded();
  //   return new Promise((resolve, reject) => {
  //     const listener = (message: WorkerMessage): void => {
  //       if (message._id === id) {
  //         worker.port.removeListener("message", listener);
  //         // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
  //         this.tasks.set(worker, this.tasks.get(worker)! - 1);
  //         if (message.error) reject(message.error);
  //         else resolve(message.logits);
  //       }
  //     };

  //     worker.port.on("message", listener);
  //   });
  // }

  queueTask<T extends WorkerTask>(task: T): Promise<WorkerTaskReturn<T>> {
    // const worker = this.getWorker();
    // this.tasks.set(worker, this.tasks.get(worker)! + 1);

    // const taskId = this._id++;
    return new Promise<WorkerTaskReturn<T>>((resolve, reject) => {
      let workerTask: PoolTask;
      if (task.type === "load") {
        const loadPoolTask = task as ModelLoadingTask;
        workerTask = {
          type: "load",
          params: loadPoolTask.params,
          resolve: resolve as (value?: void | PromiseLike<void>) => void,
          reject
        };
      } else {
        const inferPoolTask = task as InferenceTask;
        workerTask = {
          type: "infer",
          inputs: inferPoolTask.inputs,
          path: inferPoolTask.path,
          resolve: resolve as (
            value?: [Logits, Logits] | PromiseLike<[Logits, Logits]>
          ) => void,
          reject
        };
      }

      this.queue.push(workerTask);
      this.runNextTask();
    });
  }

  private runNextTask(): void {
    const worker = this.getWorker();
    if (!worker) {
      return;
    }

    const taskIndex = this.queue.findIndex(
      t => t.type === "load" || this.models.has(t.path)
    );
    if (taskIndex === -1) {
      return;
    }

    this.toggleWorkerStatus(worker, true);
    const task = this.queue.splice(taskIndex, 1)[0];
    if (task.type === "infer") {
      task.params = this.models.get(task.path);
    }

    worker.once("message", (value: WorkerMessage) => {
      if (value.error) {
        task?.reject(value.error);
      }

      if (value.type === "load") {
        // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
        this.models.set(task.params!.path, task.params!);
        task.resolve();
      } else {
        const inferTask = task as InferenceWorkerTask;
        inferTask.resolve(value.data);
      }

      this.toggleWorkerStatus(worker, false);
      this.runNextTask();
    });

    const workerTask: Partial<WorkerTask> = { type: task.type, params: task.params };
    if (task.type === "infer") {
      (workerTask as InferenceTask).inputs = task.inputs;
    }

    worker.postMessage(workerTask);
  }

  private getWorker(): Worker | null {
    let worker;
    for (const entry of this.workers) {
      if (entry[1] === false) {
        worker = entry[0];
        break;
      }
    }

    if (worker) {
      // a worker is free, use it
      return worker;
    }

    if (this.workers.length < (this.opts?.maxThreads ?? 5)) {
      this.addWorker();
      // this.emitter.emit("FullPool");
      // if (this.workers.length - 1 === this.nextWorker) {
      //   this.nextWorker = 0;
      //   return this.workers[this.nextWorker];
      // } else {
      //   this.nextWorker++;
      //   return this.workers[this.nextWorker];
      // }
    }

    return null;
    // all workers are busy create a new worker
    // worker.port.on("message", message => {
    //   if (message.kill) {
    //     worker.postMessage({ type: "kill" });
    //     worker.terminate();
    //   }
    // });
  }

  private toggleWorkerStatus(worker: Worker, running: boolean): void {
    for (let index = 0; index < this.workers.length; index++) {
      const [w] = this.workers[index];
      if (w.threadId === worker.threadId) {
        this.workers[index] = [w, running];
        break;
      }
    }
  }

  private addWorker(): void {
    const worker = new Worker(joinPaths(ROOT_DIR, "runtimes/thread-real.js"), {
      env: SHARE_ENV
    });

    worker.once("online", () => {
      this.opts?.onlineHandler?.();
      this.toggleWorkerStatus(worker, false);
      this.runNextTask();
    });

    this.opts?.errorHandler && worker.on("error", this.opts.errorHandler);
    // TODO handle properly when a thread exit
    this.opts?.exitHandler && worker.on("exit", this.opts.exitHandler);

    // we will attach a listener for every task,
    // when task is completed the listener will be removed
    // but to avoid warnings we are increasing the max listeners size
    // worker.port.setMaxListeners(this.opts?.maxTasks ?? 1000);

    this.workers.push([worker, false]);
    // this.tasks.set(worker, 0);
    // return worker;
  }
}
