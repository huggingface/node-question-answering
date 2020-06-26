import { MessagePort } from "worker_threads";

import { ModelOutputNames } from "../models/model";
import { RuntimeInputsNames } from "./runtime";

export interface BaseMessage {
  messageType: "load" | "infer";
}

export interface InferenceMessage extends BaseMessage {
  messageType: "infer";
  ids: number[][];
  attentionMask: number[][];
  tokenTypeIds?: number[][];
}

export interface LoadMessage extends BaseMessage {
  messageType: "load";
  initPort: MessagePort;
  params: WorkerParams;
  sessionId: number;
}

export type Message = LoadMessage | InferenceMessage;

export interface LoadingStatus {
  status: "loaded";
}

export interface WorkerParams {
  inputsNames: RuntimeInputsNames;
  outputsNames: Required<ModelOutputNames>;
  tfInputsNames: Record<string, string>;
  tfOutputsNames: Record<string, string>;
  path: string;
}
