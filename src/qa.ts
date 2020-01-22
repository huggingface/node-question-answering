import path from 'path';
import * as tf from '@tensorflow/tfjs-node';
import { TFSavedModel } from '@tensorflow/tfjs-node/dist/saved_model';
import { NamedTensorMap } from '@tensorflow/tfjs-node';
import { BertWordPieceTokenizer } from "tokenizers";

const ASSETS_PATH = path.join(__dirname, '../assets');
const MODEL_PATH  = path.join(ASSETS_PATH, 'distilbert');
const VOCAB_PATH  = path.join(ASSETS_PATH, 'vocab.txt');

export interface QAOptions {
  modelPath?: string;
  tokenizer?: BertWordPieceTokenizer;
  vocabPath?: string;
}

export class QAClient {
  private constructor(private model: TFSavedModel, private tokenizer: BertWordPieceTokenizer) {}

  static async fromOptions(options?: QAOptions): Promise<QAClient> {
    const model = await tf.node.loadSavedModel(options?.modelPath ?? MODEL_PATH);
    const tokenizer = options?.tokenizer ?? await BertWordPieceTokenizer.fromOptions({ vocabFile: options?.vocabPath ?? VOCAB_PATH });
  
    return new QAClient(model, tokenizer);
  }

  async predict(question: string, context: string): Promise<string | null> {
    const sequenceLength = 384;

    const encoding = await this.tokenizer.encode(question, context);
    encoding.pad(sequenceLength);
    
    const inputTensor = tf.tensor(encoding.getIds(), [1, sequenceLength], 'int32');
    const maskTensor  = tf.tensor(encoding.getAttentionMask(), [1, sequenceLength], 'int32');
    const result      = this.model.predict({ 'input_ids': inputTensor, 'attention_mask': maskTensor }, { verbose: true }) as NamedTensorMap;

    const startLogits = await result['output_0'].squeeze().array() as number[];
    const endLogits   = await result['output_1'].squeeze().array() as number[];

    const typeIds = encoding.getTypeIds();
    const contextFirstIndex = typeIds.findIndex(x => x === 1);
    const contextLastIndex  = typeIds.findIndex((x, i) => i > contextFirstIndex && x === 0) - 1;
    const [sortedStartLogits, sortedEndLogits] =
      [startLogits, endLogits].map(logits => 
        logits
          .slice(contextFirstIndex, contextLastIndex)
          .map((val, i) => [i+contextFirstIndex, val])
          .sort((a, b) => b[1] - a[1])
      );
    
    const answerIndexes: { start?: number, end?: number } = {};
    for (const startLogit of sortedStartLogits) {
      for (const endLogit of sortedEndLogits) {
        if (endLogit[0] < startLogit[0]) {
          continue
        }
    
        answerIndexes.start = startLogit[0];
        answerIndexes.end   = endLogit[0];
        break;
      }
    
      if (answerIndexes.start != undefined && answerIndexes.end != undefined) {
        break;
      }
    }
    
    const answerText: string[] = [];
    const tokens = encoding.getTokens();
    if (answerIndexes.start != undefined && answerIndexes.end != undefined) {
      for (let i = answerIndexes.start; i <= answerIndexes.end; i++) {
        answerText.push(tokens[i]);
      }
    }

    return answerText.length ? answerText.join(' ') : null;
  }
}
