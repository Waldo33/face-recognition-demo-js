import { defineConfig } from "vite";

function resolveBase() {
  const repository = process.env.GITHUB_REPOSITORY || "";
  const repoName = repository.split("/")[1];
  if (process.env.GITHUB_ACTIONS && repoName) {
    return `/${repoName}/`;
  }
  return "/";
}

export default defineConfig({
  base: resolveBase()
});
