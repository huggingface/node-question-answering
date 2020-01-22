#!/usr/bin/env node

const fs = require("fs");
const promisify = require("util").promisify;
const path = require("path");
const shell = require("shelljs");
const https = require("https");
const ProgressBar = require("progress");
const tar = require("tar");

const exists = promisify(fs.exists);

const distPath = "./dist";
const buildPath = "./build";

// Fail this script if any of these commands fail
shell.set("-e");

// Ensure that our directory is set to the root of the repo
const rootDirectory = path.join(path.dirname(process.argv[1]), "../");
shell.cd(rootDirectory);

const arg = process.argv.slice(2)[0];

run(arg);

/**************************************/

async function run(arg) {
  switch (arg) {
    case "--download-model":
      await downloadModel();
      break;

    case "--typescript":
      await buildTs();
      break;

    case "--npm-publish":
      await buildTs();
      await npmPublish();
      break;

    default:
      shell.echo("No arg provided, doing nothing...");
      break;
  }
}

async function downloadModel() {
  const assetsDir = "./assets";
  await ensureDir(assetsDir);

  if (!(await exists(`${assetsDir}/distilbert`))) {
    shell.echo("Downloading QA model (233MB)...");

    await new Promise((resolve, reject) => {
      https.get(
        "https://s3.amazonaws.com/models.huggingface.co/bert/distilbert-base-uncased-distilled-squad-384-saved_model.tar.gz",
        res => {
          const bar = new ProgressBar("[:bar] :percent :etas", {
            width: 30,
            total: parseInt(res.headers["content-length"], 10)
          });

          res
            .on("data", chunk => bar.tick(chunk.length))
            .pipe(tar.x({ cwd: assetsDir }))
            .on("close", resolve)
            .on("error", reject);
        }
      );
    });
  }

  const vocabFile = "vocab.txt";
  if (!(await exists(`${assetsDir}/${vocabFile}`))) {
    shell.echo("Downloading vocab file...");

    await new Promise((resolve, reject) => {
      https.get(
        "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt",
        res => {
          const bar = new ProgressBar("[:bar] :percent :etas", {
            width: 30,
            total: parseInt(res.headers["content-length"], 10)
          });

          res
            .on("data", chunk => bar.tick(chunk.length))
            .pipe(fs.createWriteStream(`${assetsDir}/${vocabFile}`))
            .on("close", resolve)
            .on("error", reject);
        }
      );
    });
  }
}

async function buildTs() {
  shell.echo("BUILDING TS...");

  // Cleanup the previous build, if it exists
  shell.rm("-rf", distPath);

  shell.exec("npm ci --ignore-scripts");
  await ensureDir(distPath);
  shell.exec("npx tsc -p tsconfig.prod.json");

  shell.echo("BUILDING TS COMPLETE...");
}

async function npmPublish() {
  shell.echo("PUBLISHING ON NPM...");

  shell.rm("-rf", buildPath);
  await ensureDir(buildPath);
  shell.cp(
    "-r",
    [distPath, "package.json", "README.md", "LICENSE", "./scripts"],
    buildPath
  );

  // Add a NPM install script to the package.json that we push to NPM so that when consumers pull it down it
  // runs the expected node-pre-gyp step.
  const npmPackageJson = require(`${rootDirectory}/${buildPath}/package.json`);
  npmPackageJson.scripts.postinstall = "node scripts/build.js --download-model";
  await fs.promises.writeFile(
    `${buildPath}/package.json`,
    JSON.stringify(npmPackageJson, null, 2)
  );

  // shell.exec(`npm pack ${buildPath}`);
  shell.exec(`npm publish ${buildPath} --access public`);

  shell.echo("PUBLISHING ON NPM COMPLETE...");
}

/**
 * Ensures a directory exists, creates as needed.
 */
async function ensureDir(dirPath) {
  if (!(await exists(dirPath))) {
    shell.mkdir(dirPath);
  }
}
