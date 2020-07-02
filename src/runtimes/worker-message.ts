import { MessagePort } from "worker_threads";

import { FullParams } from "./runtime";

export interface BaseMessage {
  type: "load" | "infer" | "init";
}

export interface InferenceMessage extends BaseMessage {
  _id: number;
  inputs: {
    ids: number[][];
    attentionMask: number[][];
    tokenTypeIds?: number[][];
  };
  model: string;
  type: "infer";
}

export interface LoadMessage extends BaseMessage {
  params: FullParams;
  type: "load";
}

export interface InitMessage extends BaseMessage {
  inferencePort: MessagePort;
  initPort: MessagePort;
  loadPort: MessagePort;
  type: "init";
}

export type Message = LoadMessage | InferenceMessage | InitMessage;
