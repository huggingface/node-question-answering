import fetch from "node-fetch";

import {
  Model,
  ModelOptions,
  ModelParams,
  PartialMetaGraph,
  PartialModelTensorInfo
} from "./model";

export class RemoteModel extends Model {
  private constructor(public params: ModelParams) {
    super();
  }

  async runInference(
    ids: number[][],
    attentionMask: number[][]
  ): Promise<[number[][], number[][]]> {
    const result = await fetch(`${this.params.path}:predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        inputs: {
          [this.params.inputsNames.ids]: ids,
          [this.params.inputsNames.attentionMask]: attentionMask
        },
        signature_name: this.params.signatureName
      })
    }).then(r => r.json());

    const startLogits = result.outputs[
      this.params.outputsNames.startLogits
    ] as number[][];
    const endLogits = result.outputs[this.params.outputsNames.endLogits] as number[][];

    return [startLogits, endLogits];
  }

  static async fromOptions(options: ModelOptions): Promise<RemoteModel> {
    const modelGraph = await this.getRemoteMetaGraph(options.path);
    const fullParams = this.computeParams(options, modelGraph);

    return new RemoteModel(fullParams);
  }

  private static async getRemoteMetaGraph(url: string): Promise<PartialMetaGraph> {
    const httpResult = await fetch(`${url}/metadata`).then(r => r.json());

    const rawSignatureDef = httpResult.metadata.signature_def.signature_def;
    const signatures = Object.keys(rawSignatureDef).filter(k => !k.startsWith("__"));
    const parsedSignatures = signatures.map(k => {
      const signature = rawSignatureDef[k];
      const parsedInputs = Object.entries<any>(signature.inputs).map<
        Record<string, PartialModelTensorInfo>
      >(([key, val]) => {
        return {
          [key]: {
            shape: val.tensor_shape.dim.reduce(
              (
                acc: number[],
                val: {
                  size: string;
                }
              ) => [...acc, parseInt(val.size)],
              []
            )
          }
        };
      });
      return {
        [k]: {
          inputs: Object.assign({}, ...parsedInputs),
          outputs: Object.assign(
            {},
            ...Object.keys(signature.outputs).map(key => ({ [key]: {} }))
          )
        }
      };
    });

    return {
      signatureDefs: Object.assign({}, ...parsedSignatures)
    };
  }
}
