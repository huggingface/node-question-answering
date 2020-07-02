import { Encoding } from "tokenizers";

import { Logits, Model, ModelInput, ModelType } from "./model";

export class ElectraModel extends Model {
  public static readonly inputs: ReadonlyArray<ModelInput> = [
    ModelInput.AttentionMask,
    ModelInput.Ids
  ];

  public readonly type = ModelType.Electra;

  runInference(encodings: Encoding[]): Promise<[Logits, Logits]> {
    return this.runtime.runInference(
      encodings.map(e => e.ids),
      encodings.map(e => e.attentionMask)
    );
  }
}
