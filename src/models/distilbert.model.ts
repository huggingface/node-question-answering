import { Encoding } from "tokenizers";

import { Model, ModelInput, ModelType } from "./model";

export class DistilbertModel extends Model {
  public static readonly inputs: ReadonlyArray<ModelInput> = [
    ModelInput.AttentionMask,
    ModelInput.Ids
  ];

  public readonly type = ModelType.Distilbert;

  runInference(encodings: Encoding[]): Promise<[number[][], number[][]]> {
    return this.runtime.runInference(
      encodings.map(e => e.ids),
      encodings.map(e => e.attentionMask)
    );
  }
}
