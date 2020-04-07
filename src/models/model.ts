import { Encoding } from "tokenizers";

import { Runtime, RuntimeType } from "../runtimes/runtime";

export enum ModelInput {
  AttentionMask = "attentionMask",
  Ids = "inputIds",
  TokenTypeIds = "tokenTypeIds"
}

export enum ModelType {
  Distilbert = "distilbert",
  Roberta = "roberta"
}

export abstract class Model {
  public readonly cased: boolean;
  public readonly inputLength: number;
  public readonly path: string;
  public abstract readonly type: ModelType;

  constructor(protected runtime: Runtime, cased?: boolean) {
    this.cased = !!cased;
    this.inputLength = runtime.params.shape[1];
    this.path = runtime.params.path;
  }

  abstract runInference(encodings: Encoding[]): Promise<[number[][], number[][]]>;
}

export interface ModelOptions {
  /**
   * @default false
   */
  cased?: boolean;
  inputsNames?: ModelInputsNames;
  type?: ModelType;
  outputsNames?: ModelOutputNames;
  path: string;
  runtime?: RuntimeType;
  /**
   * @default "serving_default"
   */
  signatureName?: string;
}

export interface ModelInputsNames {
  /**
   * @default "inputs_ids"
   */
  [ModelInput.Ids]?: string;
  /**
   * @default "attention_mask"
   */
  [ModelInput.AttentionMask]?: string;
  /**
   * @default "token_type_ids"
   */
  [ModelInput.TokenTypeIds]?: string;
}

export interface ModelOutputNames {
  /**
   * @default "output_0"
   */
  startLogits?: string;
  /**
   * @default "output_1"
   */
  endLogits?: string;
}
