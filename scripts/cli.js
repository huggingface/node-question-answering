#!/usr/bin/env node
//@ts-check

const shell = require("shelljs");
const yargs = require("yargs");

const utils = require("../dist/utils");

// Fail this script if any of these commands fail
shell.set("-e");

yargs
  .command(
    "download [model]",
    "Download a model (defaults to distilbert-base-cased-distilled-squad)",
    yargs => {
      yargs
        .positional("model", {
          default: "distilbert-base-cased-distilled-squad",
          type: "string"
        })
        .option("dir", {
          default: "./.models",
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
    utils.downloadModelWithVocab
  )
  .demandCommand()
  .help().argv;
