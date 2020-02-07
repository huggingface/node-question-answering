#!/usr/bin/env node

const fs = require("fs");
const path = require("path");
const shell = require("shelljs");
const utils = require("./utils");

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
  await utils.ensureDir(distPath);
  shell.exec("npx tsc -p tsconfig.prod.json");

  shell.echo("BUILDING TS COMPLETE...");
}

async function npmPublish() {
  shell.echo("PUBLISHING ON NPM...");

  shell.rm("-rf", buildPath);
  await utils.ensureDir(buildPath);
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
