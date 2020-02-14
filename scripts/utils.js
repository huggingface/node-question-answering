const fs = require("fs");
const shell = require("shelljs");
const promisify = require("util").promisify;

const exists = promisify(fs.exists);

/**
 * Ensures a directory exists, creates as needed.
 */
async function ensureDir(dirPath, recursive = true) {
  if (!(await exists(dirPath))) {
    recursive ? shell.mkdir("-p", dirPath) : shell.mkdir(dirPath);
  }
}

module.exports = {
  ensureDir,
  exists
};
