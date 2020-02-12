# Question Answering for Node.js

[![npm version](https://badge.fury.io/js/question-answering.svg)](https://www.npmjs.com/package/question-answering)

#### Question answering directly in Node.js, with only 3 lines of code!

This package leverages the power of the [tokenizers](https://github.com/huggingface/tokenizers) library (built with Rust) to process the input text. It then uses [TensorFlow.js](https://www.tensorflow.org/js) to run the [DistilBERT](https://arxiv.org/abs/1910.01108)-cased model fine-tuned for Question Answering (87.1 F1 score on SQuAD v1.1 dev set, compared to 88.7 for BERT-base-cased).

## Installation

```bash
npm install question-answering
```

## Simple example

This example is running the model locally. To do so, you first need to download the model and vocabulary file:
```bash
npx question-answering download
```

> By default, the model and vocabulary are downloaded inside a `.models` directory at the root of your project; you can provide a custom directory by using the `--dir` option of the CLI.

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

<a name="remote-model"></a>
### Using a remote model with [TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving)

You may prefer to host your model on a dedicated server. It's possible by simply passing the server endpoint as the `path` option and `remote` to `true`. Here is a simple example using [Docker](https://www.tensorflow.org/tfx/serving/docker) locally:

```bash
# Inside our project root, download DistilBERT-cased to its default `.models` location
npx question-answering download

# Download the TensorFlow Serving Docker image
docker pull tensorflow/serving

# Start TensorFlow Serving container and open the REST API port.
# Notice that in the `target` path we add a `/1`:
# this is required by TFX which is expecting the models to be "versioned"
docker run -t --rm -p 8501:8501 \
    --mount type=bind,source="$(pwd)/.models/distilbert-cased/",target="/models/cased/1" \
    -e MODEL_NAME=cased \
    tensorflow/serving &
```

Then in your code:

```typescript
const qaClient = await QAClient.fromOptions({
  model: { path: "http://localhost:8501/v1/models/cased", cased: true, remote: true }
});
```


### Using a different model

You can choose to use the uncased version of DistilBERT instead.

First, download the uncased model:
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

You can also choose to use a custom model and pass it to `QAClient.fromOptions`, the same way than for DistilBERT-uncased. Check the [`QAOptions`](src/qa-options.ts) interface for the complete list of options. And you can still [host it remotely](#remote-model).

### Using a custom tokenizer

You can provide your own tokenizer instance to `QAClient.fromOptions`, as long as it implements the [`BERTWordPieceTokenizer`](https://github.com/huggingface/tokenizers/blob/master/bindings/node/lib/tokenizers/bert-wordpiece.tokenizer.ts) methods.

## Performances

Thanks to [the native execution of SavedModel format](https://groups.google.com/a/tensorflow.org/d/msg/tfjs/Xtf6s1Bpkr0/7-Eqn8soAwAJ) in TFJS, the performance of such models is similar to the one using TensorFlow in Python. 

Specifically, here are the results of a benchmark using `question-answering` completely locally, with a (pseudo-)remote model server (i.e. local Docker), and using the Question Answering pipeline in the [`transformers`](https://github.com/huggingface/transformers) library.

![QA benchmark chart](https://docs.google.com/spreadsheets/d/e/2PACX-1vRCprbDB9T8nwdOpRv2pmlOXWKw3vVOx5P2jbn7hipjCyaGRuQS3u5KWpE7ux5Q0jbqT9HFVMivkI4x/pubchart?oid=2051609279&format=image)
_Shorts texts are texts between 500 and 1000 characters, long texts are between 4000 and 5000 characters. You can check the `question-answering` benchmark script [here](./scripts/benchmark.js) (the `transformers` one is equivalent)._
