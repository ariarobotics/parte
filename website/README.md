# PARTE website

From the PARTE repository root, build the exporter once:

```sh
git submodule update --init --recursive
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_EXECUTABLES=OFF -DBUILD_CACHE_EXPORTER=ON -DUSE_SYSTEM_EIGEN=ON
cmake --build build --target parte-cache -j 4
```

Omit `-DUSE_SYSTEM_EIGEN=ON` to download Eigen. The exporter uses Open3D for point-cloud loading; CMake downloads it unless `-DUSE_SYSTEM_OPEN3D=ON` is set.
It uses the same processing and registration pipeline as the command-line tools, including ordinary plane matching for ground.

Then generate data and run the site:

```sh
cd website
pnpm install --frozen-lockfile
pnpm cache:build --datasets-root /path/to/processed_datasets
pnpm dev
```

The generated data lives in `website/cache/viewer/`; private build state is in `website/cache/viewer.state/`.
Recompile the exporter after C++ changes, then use `pnpm cache:build --datasets-root /path/to/processed_datasets --rebuild` to regenerate.

Open http://localhost:4321.

`pnpm build` builds the production assets to `dist/`, including the viewer data.

## GitHub Pages

Set **Settings > Pages > Source** to **GitHub Actions**. The `github-pages`
deployment environment is created automatically if missing. If it has branch
restrictions, allow `web` under **Settings > Environments > github-pages**.
The workflow uses the published wPMC cache from R2.
The workflow downloads the cache, builds the site, and deploys it. Pushes to `web`
currently trigger deployment too. After the first successful deployment, remove
the `push` trigger from `.github/workflows/deploy.yml` to return to manual-only runs.

GitHub shows the Run workflow button when the workflow exists on the default branch.
The first push to `web` runs and registers the deployment workflow;
then remove that trigger and dispatch it through the CLI/API using `--ref web`.
See [GitHub workflow events](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#workflow_dispatch).
