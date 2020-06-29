import { join as joinPaths } from "path";
import { MessageChannel, MessagePort, SHARE_ENV, Worker } from "worker_threads";

import { Logits } from "../models/model";
import { FullParams } from "./runtime";
import { InferenceMessage, InitMessage } from "./worker-message";

interface InferenceTask {
  id: number;
  model: string;
  onsuccess: (value?: [Logits, Logits] | PromiseLike<[Logits, Logits]>) => void;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  onerror: (reason?: any) => void;
  inputs: {
    ids: number[][];
    attentionMask: number[][];
    tokenTypeIds?: number[][];
  };
}

interface ModelInfos {
  loaded: boolean;
  onloaded?: () => void;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  onloaderror?: (error: any) => void;
}

export class SavedModelWorker extends Worker {
  private inferencePort: MessagePort;
  private initPort: MessagePort;
  private loaded?: true;
  private loadPort: MessagePort;
  private models = new Map<string, ModelInfos>();
  private queues = new Map<string, InferenceTask[]>();
  private taskId = 0;

  constructor() {
    super(joinPaths(__filename, "../saved-model.worker-thread.js"), {
      env: SHARE_ENV
    });

    const initChannel = new MessageChannel();
    this.initPort = initChannel.port2;

    const loadChannel = new MessageChannel();
    this.loadPort = loadChannel.port2;

    const inferenceChannel = new MessageChannel();
    this.inferencePort = inferenceChannel.port2;
    this.inferencePort.setMaxListeners(1000000);

    this.once("online", () => {
      this.initPort.once("close", () => (this.loaded = true));
      const message: InitMessage = {
        type: "init",
        initPort: initChannel.port1,
        loadPort: loadChannel.port1,
        inferencePort: inferenceChannel.port1
      };

      this.postMessage(message, [
        initChannel.port1,
        loadChannel.port1,
        inferenceChannel.port1
      ]);
    });

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    this.loadPort.on("message", (data: { model: string; error?: any }) => {
      const modelInfos = this.models.get(data.model);
      if (data.error) {
        modelInfos?.onloaderror?.(data.error);
        return;
      }

      modelInfos?.onloaded?.();
      this.models.set(data.model, { loaded: true });
      this.run(data.model);
    });
  }

  loadModel(params: FullParams): Promise<void> {
    return new Promise<void>((resolve, reject) => {
      this.models.set(params.path, {
        loaded: false,
        onloaded: resolve,
        onloaderror: reject
      });

      this.queues.set(params.path, []);

      if (this.loaded) {
        this.postMessage({ type: "load", params });
      } else {
        this.initPort.once("close", () => {
          this.postMessage({ type: "load", params });
        });
      }
    });
  }

  queueInference(
    modelPath: string,
    ids: number[][],
    attentionMask: number[][],
    tokenTypeIds?: number[][]
  ): Promise<[Logits, Logits]> {
    const taskId = this.taskId++;
    return new Promise<[Logits, Logits]>((resolve, reject) => {
      const inferenceTask: InferenceTask = {
        id: taskId,
        model: modelPath,
        onsuccess: resolve,
        onerror: reject,
        inputs: { ids, attentionMask, tokenTypeIds }
      };

      const model = this.models.get(inferenceTask.model);
      if (!model?.loaded) {
        const queue = this.queues.get(inferenceTask.model);
        queue && queue.push(inferenceTask);
      } else {
        this.runTask(inferenceTask);
      }
    });
  }

  private run(model: string): void {
    const queue = this.queues.get(model) ?? [];
    const queueLength = queue.length;
    for (let i = 0; i < queueLength; i++) {
      this.runTask(queue[i]);
    }

    queue.splice(0, queueLength);
  }

  private runTask(task: InferenceTask): void {
    const listener = (data: {
      _id: number;
      logits: [Logits, Logits];
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      error?: any;
    }): void => {
      // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
      if (data._id != task!.id) {
        return;
      } else {
        this.inferencePort.removeListener("message", listener);
      }

      if (data.logits) {
        // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
        task!.onsuccess(data.logits);
      } else {
        // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
        task!.onerror(data.error);
      }
    };

    this.inferencePort.on("message", listener);
    const message: InferenceMessage = {
      _id: task.id,
      type: "infer",
      inputs: task.inputs,
      model: task.model
    };

    this.postMessage(message);
  }
}
