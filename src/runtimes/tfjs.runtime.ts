import * as tf from "@tensorflow/tfjs-node";
import * as path from "path";

import { Logits, ModelInput } from "../models/model";
import { isOneDimensional } from "../utils";
import {
  FullParams,
  ModelDefaults,
  PartialMetaGraph,
  Runtime,
  RuntimeOptions
} from "./runtime";

export class TFJS extends Runtime {
  private constructor(private model: tf.GraphModel, params: Readonly<FullParams>) {
    super(params);
  }

  protected static get defaults(): Readonly<ModelDefaults> {
    return {
      ...super.defaults,
      outputsNames: {
        endLogits: "Identity",
        startLogits: "Identity_1"
      }
    };
  }

  async runInference(
    ids: number[][],
    attentionMask: number[][],
    tokenTypeIds?: number[][]
  ): Promise<[Logits, Logits]> {
    const result = tf.tidy(() => {
      const inputTensor = tf.tensor(ids, undefined, "int32");
      const maskTensor = tf.tensor(attentionMask, undefined, "int32");

      const modelInputs = {
        [this.params.inputsNames[ModelInput.Ids]]: inputTensor,
        [this.params.inputsNames[ModelInput.AttentionMask]]: maskTensor
      };

      if (tokenTypeIds && this.params.inputsNames[ModelInput.TokenTypeIds]) {
        const tokenTypesTensor = tf.tensor(tokenTypeIds, undefined, "int32");
        // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
        modelInputs[this.params.inputsNames[ModelInput.TokenTypeIds]!] = tokenTypesTensor;
      }

      return this.model.predict(modelInputs) as tf.Tensor[];
    });

    let [startLogits, endLogits] = await Promise.all([
      result[0].squeeze().array() as Promise<number[] | number[][]>,
      result[1].squeeze().array() as Promise<number[] | number[][]>
    ]);

    tf.dispose(result);

    if (isOneDimensional(startLogits)) {
      startLogits = [startLogits];
    }

    if (isOneDimensional(endLogits)) {
      endLogits = [endLogits];
    }

    return [startLogits, endLogits];
  }

  static async fromOptions(options: RuntimeOptions): Promise<TFJS> {
    const fullPath = path.join(options.path, "tfjs");

    const model = await tf.loadGraphModel(`file:///${fullPath}/model.json`);
    const modelGraph: PartialMetaGraph = {
      signatureDefs: {
        [options.signatureName ?? "serving_default"]: {
          inputs: Object.assign(
            {},
            ...model.inputs.map(i => ({ [i.name]: { shape: i.shape } }))
          ),
          outputs: Object.assign({}, ...model.outputs.map(o => ({ [o.name]: {} })))
        }
      }
    };

    const fullParams = this.computeParams({ ...options, path: fullPath }, modelGraph);
    return new TFJS(model, fullParams);
  }
}
