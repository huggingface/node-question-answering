# [3.0.0](https://github.com/huggingface/node-question-answering/compare/v2.0.0...v3.0.0) (TBD)

This version introduces full support for any DistilBERT/BERT/RoBERTa based models from the [Hugging Face model hub](https://huggingface.co/models). It also simplifies the model instantiation by introducing a single `initModel` factory method (and its equivalent `initTokenizer` if needed).

### BREAKING CHANGES

* Nodejs v12 is now the minimum required version if using SavedModel format: for this model format, the library now makes use of a [worker thread](https://nodejs.org/docs/latest-v12.x/api/worker_threads.html) internally to provide full non-blocking processing of prediction requests, a feature that is only supported fully starting in version 12.
* The model-specific instantiation methods are removed and replaced by a single `initModel` method paired with a `runtime` field which can either be `tfjs`, `saved_model` or `remote`.
* When passing a tokenizer to `QAClient.fromOptions`, the tokenizer now needs to extends the abstract [`Tokenizer`](./src/tokenizers/tokenizer.ts) class, which itself is a wrapper around [🤗Tokenizers](https://github.com/huggingface/tokenizers).
* The `cased` option is moved from the model instantiation to the `QAClient.fromOptions` method.

### Features

* Added compatibility with BERT/RoBERTa based models
* [12 new additional models](./README.md#models) available thanks to the [Hugging Face model hub](https://huggingface.co/models) and the NLP community
* The model doesn't need to be downloaded through the CLI before running the code for the first time: if it's not present in the default (or specified) model directory, it will be automatically downloaded at runtime during initialization, along with vocabulary / tokenizer files.
* [🤗Tokenizers](https://github.com/huggingface/tokenizers) now requires version `0.6.2`.

### How to migrate

#### When using SavedModel format

Before:
```typescript
const model = await SavedModel.fromOptions({ path: "distilbert-uncased", cased: false });
const qaClient = await QAClient.fromOptions({ model });
```

After:
```typescript
const model = await initModel({ name: "distilbert-uncased" });
const qaClient = await QAClient.fromOptions({ model, cased: false });
// `cased` can be omitted: it will be based on the tokenizer configuration if possible, otherwise inferred from the model name
```

> ⚠️ Warning: due to a [bug in TFJS](https://github.com/tensorflow/tfjs/issues/3463), the use of `@tensorflow/tfjs-node` to load or execute SavedModel models independently from the library is not recommended for now, since it could overwrite the TF backend used internally by the library. In the case where you would have to do so, make sure to require _both_ `question-answering` _and_ `@tensorflow/tfjs-node` in your code __before making any use of either of them__.

#### When using TFJS format

Before:
```typescript
const model = await TFJS.fromOptions({ path: "distilbert-uncased", cased: false });
const qaClient = await QAClient.fromOptions({ model });
```

After:
```typescript
const model = await initModel({ name: "distilbert-uncased", runtime: RuntimeType.TFJS });
const qaClient = await QAClient.fromOptions({ model }); // `cased` can be omitted (see SavedModel migration)
```

#### When using a remote model

Before:
```typescript
const model = await RemoteModel.fromOptions({ path: "http://localhost:8501/v1/models/cased" cased: false });
const qaClient = await QAClient.fromOptions({ model });
```

After:
```typescript
const model = await initModel({
  name: "distilbert-uncased",
  path: "http://localhost:8501/v1/models/cased",
  runtime: RuntimeType.Remote
});
const qaClient = await QAClient.fromOptions({ model }); // `cased` can be omitted (see SavedModel migration)
```

# [2.0.0](https://github.com/huggingface/node-question-answering/compare/v1.4.0...v2.0.0) (2020-03-10)

This version introduces support for models in TFJS format.

### BREAKING CHANGES

- 3 new classes implementing an abstract `Model` are introduced: `SavedModel`, `TFJSModel` and `RemoteModel`. They can be instanciated using a `.fromOptions` method.
- The `model` field of the `QAClient.fromOptions` methods now expects a `Model` (sub)class instance.

### Features

- Upgrade [🤗Tokenizers](https://github.com/huggingface/tokenizers) to `0.6.0`

### How to migrate

#### When using a local SavedModel

Before:
```typescript
const qaClient = await QAClient.fromOptions({
  model: { path: "distilbert-uncased", cased: false }
});
```

After:
```typescript
const model = await SavedModel.fromOptions({ path: "distilbert-uncased", cased: false });
const qaClient = await QAClient.fromOptions({ model });
```

#### When using a remote model server

Before:
```typescript
const qaClient = await QAClient.fromOptions({
  model: { path: "distilbert-uncased", cased: false, remote: true }
});
```

After:
```typescript
const model = await RemoteModel.fromOptions({ path: "http://localhost:8501/v1/models/cased", cased: false });
const qaClient = await QAClient.fromOptions({ model });
```
