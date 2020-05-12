import * as tf from "@tensorflow/tfjs-node";
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
import { LoadingStatus } from "./worker-message";

class SavedModelWorker extends Worker {
  public port: MessagePort;

  private _loaded?: true;
  private initPort: MessagePort;

  constructor(params: FullParams) {
    super(joinPaths(ROOT_DIR, "runtimes/thread.js"), { env: SHARE_ENV });

    const { port1, port2 } = new MessageChannel();
    const initChannel = new MessageChannel();
    this.port = port2;
    this.initPort = initChannel.port2;

    this.postMessage(
      { type: "init", parent: port1, initPort: initChannel.port1, params },
      [port1, initChannel.port1]
    );
  }

  loaded(): Promise<void> {
    console.log("waiting for load...");

    if (this._loaded) {
      console.log("already loaded");

      return Promise.resolve();
    }

    const status = receiveMessageOnPort(this.initPort) as
      | { message: LoadingStatus }
      | undefined;
    console.log("status", status);

    if (status) {
      if (status.message.status === "loaded") {
        this._loaded = true;
        return Promise.resolve();
      } else {
        // TODO: If failed loading
        return Promise.reject();
      }
    } else {
      return new Promise((resolve, reject) => {
        this.initPort.once("message", ({ status }: LoadingStatus) => {
          console.log("initPort", status, this.threadId);

          if (status === "loaded") {
            this._loaded = true;
            resolve();
          }

          // TODO: If failed loading
        });
      });
    }
  }
}

interface WorkerMessage {
  _id: number;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  error?: any;
  logits: [Logits, Logits];
}

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
  private _id = 0;

  private emitter = new EventEmitter();
  private nextWorker = 0;
  private tasks = new Map<SavedModelWorker, number>();
  private workers: SavedModelWorker[] = [];

  constructor(private workerParams: FullParams, private opts?: PoolOptions) {
    for (let i = 1; i <= (opts?.minThreads ?? 1); i++) {
      this.newWorker();
    }
  }

  async destroy(): Promise<void> {
    for (const worker of this.workers) {
      await worker.terminate();
    }
  }

  async runInference(
    ids: number[][],
    attentionMask: number[][],
    tokenTypeIds?: number[][]
  ): Promise<[Logits, Logits]> {
    // configure worker to handle message with the specified task
    const worker = this.chooseWorker();
    // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
    this.tasks.set(worker, this.tasks.get(worker)! + 1);
    const id = ++this._id;
    console.log("num models in ctor", tf.node.getNumOfSavedModels());
    const res = this.execute(worker, id);
    worker.postMessage({
      type: "infer",
      data: { ids, attentionMask, tokenTypeIds },
      _id: id
    });
    return res;
  }

  private async execute(worker: SavedModelWorker, id: number): Promise<[Logits, Logits]> {
    await worker.loaded();
    return new Promise((resolve, reject) => {
      const listener = (message: WorkerMessage): void => {
        if (message._id === id) {
          worker.port.removeListener("message", listener);
          // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
          this.tasks.set(worker, this.tasks.get(worker)! - 1);
          if (message.error) reject(message.error);
          else resolve(message.logits);
        }
      };

      worker.port.on("message", listener);
    });
  }

  private chooseWorker(): SavedModelWorker {
    let worker;
    for (const entry of this.tasks) {
      if (entry[1] === 0) {
        worker = entry[0];
        break;
      }
    }

    if (worker) {
      // a worker is free, use it
      return worker;
    } else {
      if (this.workers.length === (this.opts?.maxThreads ?? 5)) {
        this.emitter.emit("FullPool");
        if (this.workers.length - 1 === this.nextWorker) {
          this.nextWorker = 0;
          return this.workers[this.nextWorker];
        } else {
          this.nextWorker++;
          return this.workers[this.nextWorker];
        }
      }

      // all workers are busy create a new worker
      const worker = this.newWorker();
      worker.port.on("message", message => {
        if (message.kill) {
          worker.postMessage({ type: "kill" });
          worker.terminate();
        }
      });

      return worker;
    }
  }

  private newWorker(): SavedModelWorker {
    const worker = new SavedModelWorker(this.workerParams);
    this.opts?.errorHandler && worker.on("error", this.opts.errorHandler);
    this.opts?.onlineHandler && worker.on("online", this.opts.onlineHandler);
    // TODO handle properly when a thread exit
    this.opts?.exitHandler && worker.on("exit", this.opts.exitHandler);

    this.workers.push(worker);

    // we will attach a listener for every task,
    // when task is completed the listener will be removed
    // but to avoid warnings we are increasing the max listeners size
    worker.port.setMaxListeners(this.opts?.maxTasks ?? 1000);

    this.tasks.set(worker, 0);
    return worker;
  }
}
