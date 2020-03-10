const MODEL_DEFAULTS: ModelDefaults = {
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

  protected static get defaults(): Readonly<ModelDefaults> {
    return MODEL_DEFAULTS;
  }

  abstract runInference(
    ids: number[][],
    attentionMask: number[][]
  ): Promise<[number[][], number[][]]>;

  protected static computeParams(
    options: ModelOptions,
    graph: PartialMetaGraph,
    defaults: Readonly<ModelDefaults> = this.defaults
  ): ModelParams {
    const partialParams: Omit<ModelParams, "shape"> = {
      cased: options.cased ?? false,
      inputsNames: {
        attentionMask:
          options.inputsNames?.attentionMask ?? defaults.inputsNames.attentionMask,
        ids: options.inputsNames?.ids ?? defaults.inputsNames.ids
      },
      outputsNames: {
        endLogits: options.outputsNames?.endLogits ?? defaults.outputsNames.endLogits,
        startLogits:
          options.outputsNames?.startLogits ?? defaults.outputsNames.startLogits
      },
      path: options.path,
      signatureName: options.signatureName ?? defaults.signatureName
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

export function isOneDimensional(arr: number[] | number[][]): arr is number[] {
  return !Array.isArray(arr[0]);
}

export interface ModelDefaults {
  inputsNames: Required<ModelInputsNames>;
  outputsNames: Required<ModelOutputNames>;
  signatureName: string;
}

export interface ModelOptions {
  /**
   * @default false
   */
  cased?: boolean;
  inputsNames?: ModelInputsNames;
  outputsNames?: ModelOutputNames;
  path: string;
  /**
   * @default "serving_default"
   */
  signatureName?: string;
}

export interface ModelInputsNames {
  /**
   * @default "inputs_ids"
   */
  ids?: string;
  /**
   * @default "attention_mask"
   */
  attentionMask?: string;
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
