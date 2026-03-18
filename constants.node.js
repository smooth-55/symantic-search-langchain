import { spawnSync } from "node:child_process";

function tryLoadWith(bin) {
  const script = [
    "import json",
    "import constants",
    "payload = {",
    "  'Datasets': constants.Datasets,",
    "  'THRESHOLD': constants.THRESHOLD,",
    "  'Transcript': constants.Transcript,",
    "  'PROMPT_TO_SPLIT_INTO_CHUNKS': constants.PROMPT_TO_SPLIT_INTO_CHUNKS,",
    "  'REVIEW_PROMPT': constants.REVIEW_PROMPT,",
    "}",
    "print(json.dumps(payload, ensure_ascii=False))",
  ].join("\n");

  const result = spawnSync(bin, ["-c", script], {
    encoding: "utf-8",
  });

  if (result.status !== 0) {
    return null;
  }

  return result.stdout;
}

function loadConstantsFromPython() {
  const output =
    tryLoadWith(process.env.PYTHON_BIN || "python3") || tryLoadWith("python");

  if (!output) {
    throw new Error(
      "Failed to load constants from constants.py. Ensure python/python3 is installed and constants.py exists."
    );
  }

  return JSON.parse(output);
}

const data = loadConstantsFromPython();

export const Datasets = data.Datasets;
export const THRESHOLD = data.THRESHOLD;
export const Transcript = data.Transcript;
export const PROMPT_TO_SPLIT_INTO_CHUNKS = data.PROMPT_TO_SPLIT_INTO_CHUNKS;
export const REVIEW_PROMPT = data.REVIEW_PROMPT;
