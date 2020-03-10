import * as tf from "@tensorflow/tfjs-node";
import { exists } from "fs";
import * as path from "path";
import { promisify } from "util";

import { DEFAULT_ASSETS_PATH } from "../qa-options";
import {
  isOneDimensional,
  Model,
  ModelDefaults,
  ModelOptions,
  ModelParams,
  PartialMetaGraph
} from "./model";

export class TFJSModel extends Model {
  private constructor(private model: tf.GraphModel, public params: ModelParams) {
    super();
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
    attentionMask: number[][]
  ): Promise<[number[][], number[][]]> {
    const result = tf.tidy(() => {
      const inputTensor = tf.tensor(ids, undefined, "int32");
      const maskTensor = tf.tensor(attentionMask, undefined, "int32");

      return this.model.predict({
        [this.params.inputsNames.ids]: inputTensor,
        [this.params.inputsNames.attentionMask]: maskTensor
      }) as tf.Tensor[];
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

  static async fromOptions(options: ModelOptions): Promise<TFJSModel> {
    options.path = (await promisify(exists)(options.path))
      ? options.path
      : path.join(DEFAULT_ASSETS_PATH, options.path, "tfjs");

    const model = await tf.loadGraphModel(`file:///${options.path}/model.json`);
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

    const fullParams = this.computeParams(options, modelGraph);
    return new TFJSModel(model, fullParams);
  }
}
