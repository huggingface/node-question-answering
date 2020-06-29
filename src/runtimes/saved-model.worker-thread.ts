import * as tf from "@tensorflow/tfjs-node";
import { TFSavedModel } from "@tensorflow/tfjs-node/dist/saved_model";
import { MessagePort, parentPort } from "worker_threads";

import { Logits, ModelInput } from "../models/model";
import { isOneDimensional } from "../utils";
import { FullParams } from "./runtime";
import { InferenceMessage, Message } from "./worker-message";

let loadPort: MessagePort;
let inferencePort: MessagePort;

const modelsMap = new Map<string, TFSavedModel>();
const modelParamsMap = new Map<string, FullParams>();

parentPort?.on("message", (value: Message) => {
  switch (value.type) {
    case "infer":
      runInference(value);
      break;

    case "init":
      loadPort = value.loadPort;
      inferencePort = value.inferencePort;
      value.initPort.close();
      break;

    case "load":
      initModel(value.params);
      break;
  }
});

async function initModel(params: FullParams): Promise<void> {
  try {
    modelsMap.set(params.path, await tf.node.loadSavedModel(params.path));
    modelParamsMap.set(params.path, params);
    loadPort.postMessage({ model: params.path });
  } catch (error) {
    loadPort.postMessage({ model: params.path, error });
  }
}

async function runInference(value: InferenceMessage): Promise<void> {
  try {
    const { ids, attentionMask, tokenTypeIds } = value.inputs;
    const logits = await predict(value.model, ids, attentionMask, tokenTypeIds);
    inferencePort.postMessage({ logits, _id: value._id });
  } catch (error) {
    inferencePort.postMessage({ error, _id: value._id });
  }
}

async function predict(
  modelPath: string,
  ids: number[][],
  attentionMask: number[][],
  tokenTypeIds?: number[][]
): Promise<[Logits, Logits]> {
  // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
  const model = modelsMap.get(modelPath)!;
  // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
  const params = modelParamsMap.get(modelPath)!;

  const result = tf.tidy(() => {
    const inputTensor = tf.tensor(ids, undefined, "int32");
    const maskTensor = tf.tensor(attentionMask, undefined, "int32");

    const modelInputs = {
      [params.inputsNames[ModelInput.Ids]]: inputTensor,
      [params.inputsNames[ModelInput.AttentionMask]]: maskTensor
    };

    if (tokenTypeIds && params.inputsNames[ModelInput.TokenTypeIds]) {
      const tokenTypesTensor = tf.tensor(tokenTypeIds, undefined, "int32");
      // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
      modelInputs[params.inputsNames[ModelInput.TokenTypeIds]!] = tokenTypesTensor;
    }

    return model?.predict(modelInputs) as tf.NamedTensorMap;
  });

  let [startLogits, endLogits] = await Promise.all([
    result[params.outputsNames.startLogits].squeeze().array() as Promise<
      number[] | number[][]
    >,
    result[params.outputsNames.endLogits].squeeze().array() as Promise<
      number[] | number[][]
    >
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
