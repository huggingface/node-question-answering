import { Encoding } from "tokenizers";

import { Model, ModelInput, ModelType } from "./model";

export class RobertaModel extends Model {
  public static readonly inputs: ReadonlyArray<ModelInput> = [
    ModelInput.AttentionMask,
    ModelInput.Ids
  ];

  public readonly type = ModelType.Roberta;

  runInference(encodings: Encoding[]): Promise<[number[][], number[][]]> {
    return this.runtime.runInference(
      encodings.map(e => e.ids),
      encodings.map(e => e.attentionMask)
    );
  }
}
