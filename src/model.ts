import { ModelInputsNames, ModelOptions, ModelOutputNames } from "./qa-options";

const MODEL_DEFAULTS = {
  inputsNames: {
    attentionMask: "attention_mask",
    ids: "input_ids"
  },
  outputsNames: {
    endLogits: "output_1",
    startLogits: "output_0"
  },
  signatureName: "serving_default"
};

export abstract class Model {
  abstract params: Readonly<ModelParams>;

  abstract runInference(
    ids: number[][],
    attentionMask: number[][]
  ): Promise<[number[][], number[][]]>;

  protected static computeParams(
    options: ModelOptions,
    graph: PartialMetaGraph
  ): ModelParams {
    const partialParams: Omit<ModelParams, "shape"> = {
      cased: options.cased ?? false,
      inputsNames: {
        attentionMask:
          options.inputsNames?.attentionMask ?? MODEL_DEFAULTS.inputsNames.attentionMask,
        ids: options.inputsNames?.ids ?? MODEL_DEFAULTS.inputsNames.ids
      },
      outputsNames: {
        endLogits:
          options.outputsNames?.endLogits ?? MODEL_DEFAULTS.outputsNames.endLogits,
        startLogits:
          options.outputsNames?.startLogits ?? MODEL_DEFAULTS.outputsNames.startLogits
      },
      path: options.path,
      signatureName: options.signatureName ?? MODEL_DEFAULTS.signatureName
    };

    const signatureDef = graph.signatureDefs[partialParams.signatureName];
    if (!signatureDef) {
      throw new Error(`No signature matching name "${partialParams.signatureName}"`);
    }

    for (const inputName of Object.values(partialParams.inputsNames)) {
      if (!signatureDef.inputs[inputName]) {
        throw new Error(`No input matching name "${inputName}"`);
      }
    }

    for (const outputName of Object.values(partialParams.outputsNames)) {
      if (!signatureDef.outputs[outputName]) {
        throw new Error(`No output matching name "${outputName}"`);
      }
    }

    const rawShape = signatureDef.inputs[partialParams.inputsNames.ids!].shape as
      | number[]
      | { array: [number] }[];
    const shape =
      typeof rawShape[0] === "number"
        ? rawShape
        : (rawShape as { array: [number] }[]).map(s => s.array[0]);

    return {
      ...partialParams,
      shape: shape as [number, number]
    };
  }
}

export interface ModelParams {
  cased?: boolean;
  inputsNames: Required<ModelInputsNames>;
  outputsNames: Required<ModelOutputNames>;
  path: string;
  shape: [number, number];
  signatureName: string;
}

export interface PartialMetaGraph {
  signatureDefs: {
    [key: string]: {
      inputs: {
        [key: string]: PartialModelTensorInfo;
      };
      outputs: {
        [key: string]: {};
      };
    };
  };
}

export interface PartialModelTensorInfo {
  shape?: number[];
}
