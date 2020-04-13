import { Logits, ModelInput, ModelInputsNames, ModelOutputNames } from "../models/model";

const MODEL_DEFAULTS: ModelDefaults = {
  inputsNames: {
    [ModelInput.AttentionMask]: "attention_mask",
    [ModelInput.Ids]: "input_ids",
    [ModelInput.TokenTypeIds]: "token_type_ids"
  },
  outputsNames: {
    endLogits: "output_1",
    startLogits: "output_0"
  },
  signatureName: "serving_default"
};

export enum RuntimeType {
  Remote = "remote",
  SavedModel = "saved_model",
  TFJS = "tfjs"
}

export type LocalRuntime = RuntimeType.SavedModel | RuntimeType.TFJS;

export abstract class Runtime {
  constructor(public params: Readonly<FullParams>) {}

  protected static get defaults(): Readonly<ModelDefaults> {
    return MODEL_DEFAULTS;
  }

  abstract runInference(
    ids: number[][],
    attentionMask: number[][],
    tokenTypeIds?: number[][]
  ): Promise<[Logits, Logits]>;

  protected static computeParams(
    options: RuntimeOptions,
    graph: PartialMetaGraph,
    defaults: Readonly<ModelDefaults> = this.defaults
  ): FullParams {
    const inputsNames = options.inputs.reduce<ModelInputsNames>(
      (obj, input) => ({
        ...obj,
        [input]: options.inputsNames?.[input] ?? defaults.inputsNames[input]
      }),
      {}
    );

    const partialParams: Omit<FullParams, "shape"> = {
      inputsNames: inputsNames as RuntimeInputsNames,
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

    // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
    const rawShape = signatureDef.inputs[partialParams.inputsNames[ModelInput.Ids]!]
      .shape as number[] | { array: [number] }[];
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

export interface RuntimeOptions {
  inputs: ReadonlyArray<ModelInput>;
  inputsNames?: ModelInputsNames;
  outputsNames?: ModelOutputNames;
  path: string;
  /**
   * @default "serving_default"
   */
  signatureName?: string;
}

export interface RuntimeInputsNames {
  [ModelInput.Ids]: string;
  [ModelInput.AttentionMask]: string;
  [ModelInput.TokenTypeIds]?: string;
}

export interface FullParams {
  inputsNames: RuntimeInputsNames;
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
