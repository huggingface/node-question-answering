import * as tf from "@tensorflow/tfjs-node";
import { NodeJSKernelBackend } from "@tensorflow/tfjs-node/dist/nodejs_kernel_backend";
import { TFSavedModel } from "@tensorflow/tfjs-node/dist/saved_model";
import { parentPort, threadId } from "worker_threads";

import { Logits, ModelInput } from "../models/model";
import { isOneDimensional } from "../utils";
import { LoadingStatus, Message, WorkerParams } from "./worker-message";

let model: TFSavedModel;
let params: WorkerParams;

parentPort?.on("message", async (m: Message) => {
  switch (m.messageType) {
    case "load": {
      console.log("num models before load", threadId, tf.node.getNumOfSavedModels());
      params = m.params;

      try {
        model = await tf.node.loadSavedModel(params.path);
        console.log("num models after load", threadId, tf.node.getNumOfSavedModels());
        // const backend = tf.backend() as NodeJSKernelBackend;
        // model = new TFSavedModel(
        //   m.sessionId,
        //   0,
        //   params.tfInputsNames,
        //   params.tfOutputsNames,
        //   backend
        // );
      } catch (error) {
        // TODO
        console.log(error);
      }

      const answer: LoadingStatus = { status: "loaded" };
      m.initPort.postMessage(answer);
      m.initPort.close();
      break;
    }

    case "infer": {
      console.log("num models in infer", threadId, tf.node.getNumOfSavedModels());
      const { ids, attentionMask, tokenTypeIds } = m;
      // const logits = await runInference(ids, attentionMask, tokenTypeIds);
      // parentPort?.postMessage(logits);
      break;
    }
  }
});

// async function runInference(
//   ids: number[][],
//   attentionMask: number[][],
//   tokenTypeIds?: number[][]
// ): Promise<[Logits, Logits]> {
//   const result = tf.tidy(() => {
//     const inputTensor = tf.tensor(ids, undefined, "int32");
//     const maskTensor = tf.tensor(attentionMask, undefined, "int32");

//     const modelInputs = {
//       [params.inputsNames[ModelInput.Ids]]: inputTensor,
//       [params.inputsNames[ModelInput.AttentionMask]]: maskTensor
//     };

//     if (tokenTypeIds && params.inputsNames[ModelInput.TokenTypeIds]) {
//       const tokenTypesTensor = tf.tensor(tokenTypeIds, undefined, "int32");
//       // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
//       modelInputs[params.inputsNames[ModelInput.TokenTypeIds]!] = tokenTypesTensor;
//     }

//     return model.predict(modelInputs) as tf.NamedTensorMap;
//   });

//   let [startLogits, endLogits] = await Promise.all([
//     result[params.outputsNames.startLogits].squeeze().array() as Promise<
//       number[] | number[][]
//     >,
//     result[params.outputsNames.endLogits].squeeze().array() as Promise<
//       number[] | number[][]
//     >
//   ]);

//   tf.dispose(result);

//   if (isOneDimensional(startLogits)) {
//     startLogits = [startLogits];
//   }

//   if (isOneDimensional(endLogits)) {
//     endLogits = [endLogits];
//   }

//   return [startLogits, endLogits];
// }
