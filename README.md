# Question Answering for Node.js

[![npm version](https://badge.fury.io/js/question-answering.svg)](https://www.npmjs.com/package/question-answering)

#### Run question answering locally, directly in Node.js: no Python or C++ code needed!

This package leverages the power of the [tokenizers](https://github.com/huggingface/tokenizers) library (built with Rust) to process the input text. It then uses [TensorFlow.js](https://www.tensorflow.org/js) to run the [DistilBERT](https://arxiv.org/abs/1910.01108)-cased model fine-tuned for Question Answering (87.1 F1 score on SQuAD v1.1 dev set, compared to 88.7 for BERT-base-cased).

## Installation

First download the package:
```bash
npm install question-answering
```

Then you need to download the model and vocabulary file that will be used:
```bash
npx question-answering download
```

By default, the model and vocabulary are downloaded inside a `.models` directory at the root of your project; you can provide a custom directory by using the `--dir` option of the CLI.

## Simple example

```typescript
import { QAClient } from "question-answering";

const text = `
  Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season.
  The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24–10 to earn their third Super Bowl title. The game was played on February 7, 2016, at Levi's Stadium in the San Francisco Bay Area at Santa Clara, California.
  As this was the 50th Super Bowl, the league emphasized the "golden anniversary" with various gold-themed initiatives, as well as temporarily suspending the tradition of naming each Super Bowl game with Roman numerals (under which the game would have been known as "Super Bowl L"), so that the logo could prominently feature the Arabic numerals 50.
`;

const question = "Who won the Super Bowl?";

const qaClient = await QAClient.fromOptions();
const answer = await qaClient.predict(question, text);

console.log(answer); // { text: 'Denver Broncos', score: 0.3 }
```

## Advanced

### Using a different model

You can choose to use the uncased version of DistilBERT instead.

First download the uncased model:
```bash
npx question-answering download distilbert-uncased
```

You can then instantiate a `QAClient` by specifying some options:
```typescript
const qaClient = await QAClient.fromOptions({
  model: { path: "./.models/distilbert-uncased", cased: false },
  vocabPath: "./.models/distilbert-uncased/vocab.txt"
});
```

You can also choose to use a custom model and pass it to `QAClient.fromOptions`, the same way than for DistilBERT-uncased. Check the [`QAOptions`](src/qa-options.ts) interface for the complete list of options.

### Using a custom tokenizer

You can provide your own tokenizer instance to `QAClient.fromOptions`, as long as it implements the [`BERTWordPieceTokenizer`](https://github.com/huggingface/tokenizers/blob/master/bindings/node/lib/tokenizers/bert-wordpiece.tokenizer.ts) methods.

## Performances

Thanks to [the native execution of SavedModel format](https://groups.google.com/a/tensorflow.org/d/msg/tfjs/Xtf6s1Bpkr0/7-Eqn8soAwAJ) in TFJS, the performance is similar to the one using TensorFlow in Python:

![Inference latency of MobileNet v2 between native execution in Node.js against converted execution and core Python TF on both CPU and GPU](https://lh4.googleusercontent.com/aTAHknwotexVqj_5sENZIKpsh-EsP8AuDaBupZEjuTBMzAcPbkuLP-LHuhvPoGpEmSCPpMr9MXj2up6GHbo0BNwzTY779GMzZx5EeljBNfkjQzUO-i5IO1XKMTuGQqcCYekjHZ_3)
_Inference latency of MobileNet v2 between native execution in Node.js against converted execution and core Python TF on both CPU and GPU_
