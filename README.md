# Question Answering for Node.js

[![npm version](https://badge.fury.io/js/question-answering.svg)](https://www.npmjs.com/package/question-answering)

#### Production-ready Question Answering directly in Node.js, with only 3 lines of code!

This package leverages the power of the [🤗Tokenizers](https://github.com/huggingface/tokenizers) library (built with Rust) to process the input text. It then uses [TensorFlow.js](https://www.tensorflow.org/js) to run the [DistilBERT](https://arxiv.org/abs/1910.01108)-cased model fine-tuned for Question Answering (87.1 F1 score on SQuAD v1.1 dev set, compared to 88.7 for BERT-base-cased). DistilBERT is used by default, but you can use [other models](#models) available in the [🤗Transformers](https://github.com/huggingface/transformers) library in one additional line of code!

It can run models in SavedModel and TFJS formats locally, as well as [remote models](#remote-model) thanks to TensorFlow Serving.

## Installation

```bash
npm install question-answering@latest
```

## Quickstart

The following example will automatically download the default DistilBERT model in SavedModel format if not already present, along with the required vocabulary / tokenizer files. It will then run the model and return the answer to the `question`.

```typescript
import { QAClient } from "question-answering"; // When using Typescript or Babel
// const { QAClient } = require("question-answering"); // When using vanilla JS

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

> You can also download the model and vocabulary / tokenizer files separately by [using the CLI](#cli).

## Advanced

<a name="models"></a>
### Using another model

The above example internally makes use of the default DistilBERT-cased model in the SavedModel format. The library is also compatible with any other __DistilBERT__-based model, as well as any __BERT__-based and __RoBERTa__-based models, both in SavedModel and TFJS formats. The following models are available in SavedModel format from the [Hugging Face model hub](https://huggingface.co/models) thanks to the amazing NLP community 🤗:

* [`a-ware/mobilebert-squadv2`](https://huggingface.co/a-ware/mobilebert-squadv2)
* [`a-ware/roberta-large-squadv2`](https://huggingface.co/a-ware/roberta-large-squadv2)
* [`bert-large-cased-whole-word-masking-finetuned-squad`](https://huggingface.co/bert-large-cased-whole-word-masking-finetuned-squad)
* [`bert-large-uncased-whole-word-masking-finetuned-squad`](https://huggingface.co/bert-large-uncased-whole-word-masking-finetuned-squad)
* [`deepset/bert-base-cased-squad2`](https://huggingface.co/deepset/bert-base-cased-squad2)
* [`deepset/bert-large-uncased-whole-word-masking-squad2`](https://huggingface.co/deepset/bert-large-uncased-whole-word-masking-squad2)
* [`deepset/roberta-base-squad2`](https://huggingface.co/deepset/roberta-base-squad2)
* [`distilbert-base-cased-distilled-squad`](https://huggingface.co/distilbert-base-cased-distilled-squad) (default) (also available in TFJS format)
* [`distilbert-base-uncased-distilled-squad`](https://huggingface.co/distilbert-base-uncased-distilled-squad)
* [`henryk/bert-base-multilingual-cased-finetuned-dutch-squad2`](https://huggingface.co/henryk/bert-base-multilingual-cased-finetuned-dutch-squad2)
* [`ktrapeznikov/biobert_v1.1_pubmed_squad_v2`](https://huggingface.co/ktrapeznikov/biobert_v1.1_pubmed_squad_v2)
* [`ktrapeznikov/scibert_scivocab_uncased_squad_v2`](https://huggingface.co/ktrapeznikov/scibert_scivocab_uncased_squad_v2)
* [`mrm8488/bert-base-spanish-wwm-cased-finetuned-spa-squad2-es`](https://huggingface.co/mrm8488/bert-base-spanish-wwm-cased-finetuned-spa-squad2-es)
* [`mrm8488/distill-bert-base-spanish-wwm-cased-finetuned-spa-squad2-es`](https://huggingface.co/mrm8488/distill-bert-base-spanish-wwm-cased-finetuned-spa-squad2-es)
* [`mrm8488/spanbert-finetuned-squadv2`](https://huggingface.co/mrm8488/spanbert-finetuned-squadv2)
* [`NeuML/bert-small-cord19qa`](https://huggingface.co/NeuML/bert-small-cord19qa)
* [`twmkn9/bert-base-uncased-squad2`](https://huggingface.co/twmkn9/bert-base-uncased-squad2)

To specify a model to use with the library, you need to instantiate a model class that you'll then pass to the `QAClient`:

```typescript
import { initModel, QAClient } from "question-answering"; // When using Typescript or Babel
// const { initModel, QAClient } = require("question-answering"); // When using vanilla JS

const text = ...
const question = ...

const model = await initModel({ name: "deepset/roberta-base-squad2" });
const qaClient = await QAClient.fromOptions({ model });
const answer = await qaClient.predict(question, text);

console.log(answer); // { text: 'Denver Broncos', score: 0.46 }
```

> Note that using a model [hosted on Hugging Face](https://huggingface.co/models) is not a requirement: you can use any compatible model (including any from the HF hub not already available in SavedModel or TFJS format that you converted yourself) by passing the correct local path for the model and vocabulary files in the options.

#### Using models in TFJS format

To use a TFJS model, you simply need to pass `tfjs` to the `runtime` param of `initModel` (defaults to `saved_model`):

```typescript
const model = await initModel({ name: "distilbert-base-cased-distilled-squad", runtime: RuntimeType.TFJS });
```

As with any SavedModel hosted in the HF model hub, the required files for the TFJS models will be automatically downloaded the first time. You can also download them manually [using the CLI](#cli).

<a name="remote-model"></a>
#### Using remote models with [TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving)

If your model is in the SavedModel format, you may prefer to host it on a dedicated server. Here is a simple example using [Docker](https://www.tensorflow.org/tfx/serving/docker) locally:

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

In the code, you just have to pass `remote` as `runtime` and the server endpoint as `path`:

```typescript
const model = await initModel({
  name: "distilbert-base-cased-distilled-squad",
  path: "http://localhost:8501/v1/models/cased",
  runtime: RuntimeType.Remote
});
const qaClient = await QAClient.fromOptions({ model });
```

<a name="cli"></a>
### Downloading models with the CLI

You can choose to download the model and associated vocab file(s) manually using the CLI. For example to download the `deepset/roberta-base-squad2` model:
```bash
npx question-answering download deepset/roberta-base-squad2
```

> By default, the files are downloaded inside a `.models` directory at the root of your project; you can provide a custom directory by using the `--dir` option of the CLI. You can also use `--format tfjs` to download a model in TFJS format (if available). To check all the options of the CLI: `npx question-answering download --help`.

### Using a custom tokenizer

The `QAClient.fromOptions` params object has a `tokenizer` field which can either be a set of options relative to the tokenizer files, or an instance of a class extending the abstract [`Tokenizer`](./src/tokenizers/tokenizer.ts) class. To extend this class, you can create your own or, if you simply need to adjust some options, you can import and use the provided `initTokenizer` method, which will instantiate such a class for you.

## Performances

Thanks to [the native execution of SavedModel format](https://groups.google.com/a/tensorflow.org/d/msg/tfjs/Xtf6s1Bpkr0/7-Eqn8soAwAJ) in TFJS, the performance of such models is similar to the one using TensorFlow in Python.

Specifically, here are the results of a benchmark using `question-answering` with the default DistilBERT-cased model:

* Running entirely locally (both SavedModel and TFJS formats)
* Using a (pseudo) remote model server (i.e. local Docker with TensorFlow Serving running the SavedModel format)
* Using the Question Answering pipeline in the [🤗Transformers](https://github.com/huggingface/transformers) library.

![QA benchmark chart](https://docs.google.com/spreadsheets/d/e/2PACX-1vRCprbDB9T8nwdOpRv2pmlOXWKw3vVOx5P2jbn7hipjCyaGRuQS3u5KWpE7ux5Q0jbqT9HFVMivkI4x/pubchart?oid=2051609279&format=image)
_Shorts texts are texts between 500 and 1000 characters, long texts are between 4000 and 5000 characters. You can check the `question-answering` benchmark script [here](./scripts/benchmark.js) (the `transformers` one is equivalent). Benchmark run on a standard 2019 MacBook Pro running on macOS 10.15.2._

## Troubleshooting

### Errors when using Typescript

There is a known incompatibility in the TFJS library with some types. If you encounter errors when building your project, make sure to pass the `--skipLibCheck` flag when using the Typescript CLI, or to add `skipLibCheck: true` to your `tsconfig.json` file under `compilerOptions`. See [here](https://github.com/tensorflow/tfjs/issues/2007) for more information.

### `Tensor not referenced` when running SavedModel

Due to a [bug in TFJS](https://github.com/tensorflow/tfjs/issues/3463), the use of `@tensorflow/tfjs-node` to load or execute SavedModel models independently from the library is not recommended for now, since it could overwrite the TF backend used internally by the library. In the case where you would have to do so, make sure to require _both_ `question-answering` _and_ `@tensorflow/tfjs-node` in your code __before making any use of either of them__.
