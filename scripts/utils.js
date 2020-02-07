const fs = require("fs");
const shell = require("shelljs");
const promisify = require("util").promisify;

const exists = promisify(fs.exists);

/**
 * Ensures a directory exists, creates as needed.
 */
async function ensureDir(dirPath) {
  if (!(await exists(dirPath))) {
    shell.mkdir(dirPath);
  }
}

module.exports = {
  ensureDir,
  exists
};
