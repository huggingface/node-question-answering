# Question Answering for Node.js

[![npm version](https://badge.fury.io/js/question-answering.svg)](https://www.npmjs.com/package/question-answering)

Run question answering locally, directly in Node.js: no Python or C++ code needed!

## Installation

```bash
npm install question-answering
```

## Simple example

```typescript
import { QAClient } from "question-answering";

const text = `
  Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season.
  The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24–10 to earn their third Super Bowl title. The game was played on February 7, 2016, at Levi's Stadium in the San Francisco Bay Area at Santa Clara, California.
  As this was the 50th Super Bowl, the league emphasized the "golden anniversary" with various gold-themed initiatives, as well as temporarily suspending the tradition of naming each Super Bowl game with Roman numerals (under which the game would have been known as "Super Bowl L"), so that the logo could prominently feature the Arabic numerals 50.
`;

const question = "Who won the Super Bowl?";

const qaClient = QAClient.fromOptions();
const answer = qaClient.predict(question, text);

console.log(answer); // { text: 'Denver Broncos', score: 0.37 }
```

## Details

This package makes use of the [tokenizers](https://github.com/huggingface/tokenizers) library (built with Rust) to process the input text. It then runs the [DistilBERT](https://arxiv.org/abs/1910.01108) model (97% of BERT’s performance on GLUE) fine-tuned for Question Answering thanks to [TensorFlow.js](https://www.tensorflow.org/js). The default model and the vocabulary are automatically downloaded when installing the package and everything runs locally.

You can provide your own options when instantating a `QAClient`:

```typescript
const qaClient = QAClient.fromOptions({
  // model?: ModelOptions;
  // tokenizer?: BertWordPieceTokenizer;
  vocabPath: "../myVocab.txt"
});
```

Thanks to [the native execution of SavedModel format](https://groups.google.com/a/tensorflow.org/d/msg/tfjs/Xtf6s1Bpkr0/7-Eqn8soAwAJ) in TFJS, the performance is similar to the one using TensorFlow in Python:

![Inference latency of MobileNet v2 between native execution in Node.js against converted execution and core Python TF on both CPU and GPU](https://lh4.googleusercontent.com/aTAHknwotexVqj_5sENZIKpsh-EsP8AuDaBupZEjuTBMzAcPbkuLP-LHuhvPoGpEmSCPpMr9MXj2up6GHbo0BNwzTY779GMzZx5EeljBNfkjQzUO-i5IO1XKMTuGQqcCYekjHZ_3)
_Inference latency of MobileNet v2 between native execution in Node.js against converted execution and core Python TF on both CPU and GPU_
