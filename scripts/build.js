#!/usr/bin/env node
//@ts-check

const path = require("path");
const shell = require("shelljs");
const fs = require("fs");
const promisify = require("util").promisify;

const distPath = "./dist";
const buildPath = "./build";

// Fail this script if any of these commands fail
shell.set("-e");

// Ensure that our directory is set to the root of the repo
const rootDirectory = path.join(path.dirname(process.argv[1]), "../");
shell.cd(rootDirectory);

const arg = process.argv.slice(2)[0];

run(arg)
  // Prevent "unhandledRejection" events, allowing to actually exit with error
  .catch(() => process.exit(1));

/**************************************/

async function run(arg) {
  switch (arg) {
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

  // shell.exec(`npm pack ${buildPath}`);
  shell.exec(`npm publish ${buildPath} --access public`);

  shell.echo("PUBLISHING ON NPM COMPLETE...");
}

/**
 * Ensures a directory exists, creates as needed.
 */
async function ensureDir(dirPath, recursive = true) {
  if (!(await promisify(fs.exists)(dirPath))) {
    recursive ? shell.mkdir("-p", dirPath) : shell.mkdir(dirPath);
  }
}
