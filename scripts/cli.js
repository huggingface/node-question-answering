#!/usr/bin/env node

const fs = require("fs");
const path = require("path");
const shell = require("shelljs");
const https = require("https");
const ProgressBar = require("progress");
const tar = require("tar");
const yargs = require("yargs");
const fetch = require("node-fetch");

const utils = require("./utils");

const MODELS_PARAMS = {
  "distilbert-cased": {
    subDir: "distilbert-cased",
    modelUrl: {
      saved_model:
        "https://cdn.huggingface.co/distilbert-base-cased-distilled-squad-384-saved_model.tar.gz",
      tfjs:
        "https://cdn.huggingface.co/distilbert-base-cased-distilled-squad-384-tfjs.tar.gz"
    },
    vocabUrl:
      "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-vocab.txt"
  },
  "distilbert-uncased": {
    subDir: "distilbert-uncased",
    modelUrl: {
      saved_model:
        "https://cdn.huggingface.co/distilbert-base-uncased-distilled-squad-384-saved_model.tar.gz"
    },
    vocabUrl:
      "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt"
  }
};

const ROOT_DIR = process.cwd();
const DEFAULT_MODELS_DIR = "./.models";

// Fail this script if any of these commands fail
shell.set("-e");

yargs
  .command(
    "download [model]",
    "Download a model (defaults to distilbert-cased)",
    yargs => {
      yargs
        .positional("model", {
          default: "distilbert-cased",
          type: "string"
        })
        .option("dir", {
          default: DEFAULT_MODELS_DIR,
          type: "string",
          description: "The target directory to which download the model",
          requiresArg: true,
          normalize: true
        })
        .option("format", {
          type: "string",
          default: "saved_model",
          options: ["saved_model", "tfjs"],
          requiresArg: true,
          description: "Format to download"
        })
        .option("force", {
          type: "boolean",
          alias: "f",
          description:
            "Force download of model and vocab, erasing existing if already present"
        });
    },
    downloadModel
  )
  .demandCommand()
  .help().argv;

/**
 * Download a model with associated vocabulary
 * @param {yargs.Arguments<{ model: string, dir: string, format?: "saved_model" | "tfjs", force?: boolean }>} args
 */
async function downloadModel(args) {
  let modelParams = MODELS_PARAMS[args.model];
  const modelFormat = args.format || "saved_model";

  if (!modelParams) {
    const modelUrl = `https://cdn.huggingface.co/${args.model}/${modelFormat}.tar.gz`;
    const remoteModel = await fetch(modelUrl, { method: "HEAD" });
    if (!remoteModel.ok) {
      throw new Error("The requested model doesn't seem to exist");
    }

    modelParams = {
      subDir: args.model,
      modelFormat: modelFormat,
      modelUrl: modelUrl,
      vocabUrl: `https://cdn.huggingface.co/${args.model}/vocab.txt`
    };
  } else {
    modelParams = {
      subDir: modelParams.subDir,
      modelFormat: modelFormat,
      modelUrl: modelParams.modelUrl[modelFormat],
      vocabUrl: modelParams.vocabUrl
    };
  }

  const assetsDir = path.join(ROOT_DIR, args.dir);
  await utils.ensureDir(assetsDir);

  const modelDir = path.join(
    assetsDir,
    modelParams.subDir,
    modelParams.modelFormat === "tfjs" ? "tfjs" : ""
  );
  if (args.force) {
    shell.rm("-rf", modelDir);
  }

  if (!(await utils.exists(modelDir))) {
    await utils.ensureDir(modelDir);
    shell.echo("Downloading model...");

    await new Promise((resolve, reject) => {
      https.get(modelParams.modelUrl, res => {
        const bar = new ProgressBar("[:bar] :percent :etas", {
          width: 30,
          total: parseInt(res.headers["content-length"], 10)
        });

        res
          .on("data", chunk => bar.tick(chunk.length))
          .pipe(tar.x({ cwd: modelDir }))
          .on("close", resolve)
          .on("error", reject);
      });
    });
  } else {
    shell.echo(
      `Model ${modelParams.subDir} (format: ${modelParams.modelFormat}) already exists, doing nothing...`
    );
  }

  const vocabPath = path.join(assetsDir, modelParams.subDir, "vocab.txt");
  if (!(await utils.exists(vocabPath))) {
    shell.echo("Downloading vocab file...");

    await new Promise((resolve, reject) => {
      https.get(modelParams.vocabUrl, res => {
        const bar = new ProgressBar("[:bar] :percent :etas", {
          width: 30,
          total: parseInt(res.headers["content-length"], 10)
        });

        res
          .on("data", chunk => bar.tick(chunk.length))
          .pipe(fs.createWriteStream(vocabPath))
          .on("close", resolve)
          .on("error", reject);
      });
    });
  } else {
    shell.echo("Vocabulary already exists, doing nothing...");
  }

  shell.echo("\nModel successfully downloaded!");
}
