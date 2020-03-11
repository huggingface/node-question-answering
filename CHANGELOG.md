# [2.0.0](https://github.com/huggingface/node-question-answering/compare/v1.4.0...v2.0.0) (2020-03-10)

This version introduces support for models in TFJS format.

### BREAKING CHANGES

- 3 new classes implementing an abstract `Model` are introduced: `SavedModel`, `TFJSModel` and `RemoteModel`. They can be instanciated using a `.fromOptions` method.
- The `model` field of the `QAClient.fromOptions` methods now expects a `Model` (sub)class instance.

### Features

- Upgrade 🤗Tokenizers to `0.6.0`

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
const model = await RemoteModel.fromOptions({ path: "distilbert-uncased", cased: false });
const qaClient = await QAClient.fromOptions({ model });
```
