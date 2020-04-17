import { Encoding } from "tokenizers";

import { Runtime } from "../runtimes/runtime";

export enum ModelInput {
  AttentionMask = "attentionMask",
  Ids = "inputIds",
  TokenTypeIds = "tokenTypeIds"
}

export enum ModelType {
  Distilbert = "distilbert",
  Roberta = "roberta",
  Bert = "bert" // AFTER roberta, to be sure model type inference works
}

export type Logits = number[][];

export abstract class Model {
  public readonly inputLength: number;
  public abstract readonly type: ModelType;

  constructor(
    public readonly name: string,
    public readonly path: string,
    protected runtime: Runtime
  ) {
    this.inputLength = runtime.params.shape[1];
    this.path = path;
  }

  abstract runInference(encodings: Encoding[]): Promise<[Logits, Logits]>;
}

/**
 * Infer model type from model path
 * @param modelName Model name
 * @throws If no model type inferred
 */
export function getModelType(modelName: string): ModelType {
  const types = Object.entries(ModelType);
  for (const [name, type] of types) {
    if (modelName.toLowerCase().includes(name.toLowerCase())) {
      return type;
    }
  }

  throw new Error(
    "Impossible to determine the type of the model. You can specify it manually by providing the `type` in the options"
  );
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
