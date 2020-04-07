import { Encoding } from "tokenizers";

import { Logits, Model, ModelInput, ModelType } from "./model";

export class BertModel extends Model {
  public static readonly inputs: ReadonlyArray<ModelInput> = [
    ModelInput.AttentionMask,
    ModelInput.Ids,
    ModelInput.TokenTypeIds
  ];

  public readonly type = ModelType.Bert;

  runInference(encodings: Encoding[]): Promise<[Logits, Logits]> {
    return this.runtime.runInference(
      encodings.map(e => e.ids),
      encodings.map(e => e.attentionMask),
      encodings.map(e => e.typeIds)
    );
  }
}
