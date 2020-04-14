# [3.0.0](https://github.com/huggingface/node-question-answering/compare/v2.0.0...v3.0.0) (TBD)

This version introduces full support for any DistilBERT/BERT/RoBERTa based models from the [Hugging Face model hub](https://huggingface.co/models). It also simplifies the model instantiation by introducing a single `initModel` factory method (and its equivalent `initTokenizer` if needed).

### BREAKING CHANGES

* The model-specific instantiation methods are removed and replaced by a single `initModel` method paired with a `runtime` field which can either be `tfjs`, `saved_model` or `remote`.
* When passing a tokenizer to `QAClient.fromOptions`, the tokenizer now needs to extends the abstract [`Tokenizer`](https://github.com/huggingface/node-question-answering/blob/master/src/tokenizers/tokenizer.ts) class, which itself is a wrapper around [🤗Tokenizers](https://github.com/huggingface/tokenizers).
* The `cased` option is moved from the model instantiation to the `QAClient.fromOptions` method.

### Features

* Added compatibility with BERT/RoBERTa based models
* [12 new additional models](https://github.com/huggingface/node-question-answering/blob/master/README.md#models) available thanks to the [Hugging Face model hub](https://huggingface.co/models) and the NLP community
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
const qaClient = await QAClient.fromOptions({ model, cased: false }); // `cased` can be omitted: it will be inferred from the tokenizer configuration if possible, then from the name
```

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
