# StaleyNET Dashboard

Live observability for StaleyNET training runs, in one small Rust binary.

It reads the `metrics.jsonl` and `config.json` that every StaleyNET trainer
writes, so it needs no change to training, no GPU libraries and no database.
It builds with plain `cargo` on any machine, including the AGX Thor, whose
GPU toolchain the main workspace needs.

For the run in focus it shows its status, progress and time left, and charts
of:

- the training and held-out loss, with resumes marked;
- the **generalization gap**: the held-out loss minus the mean training loss
  over the previous 1,000 steps;
- the held-out loss by residual level (language model) or by spectral
  resolution (vocoder);
- the learning rate, the gradient norm and the time per step;
- the held-out loss of several runs on one axis;
- the machine: GPU load, memory, CPU load and temperatures over the last hour.

A table lists every run, and each run is checked for overfitting. A run is
flagged when, over its last 5 validations, the held-out loss fell by less than
0.01 while the gap grew by more than 0.05; or when its held-out loss has not
improved for 4 validations and the gap has grown since its best.

## Running it

```bash
cargo build --release
target/release/staleynet-dashboard --address 0.0.0.0:8790 \
  --runs thor=~/Projects/stayleNET/runs \
  --runs orin=~/Projects/staleynet-dashboard-archive/orin
```

Each `--runs` folder gets a label, shown with its runs; a run's id is
`<label>/<name>`. The page refreshes itself every 10 seconds.

| Path | Purpose |
|---|---|
| `/` | the dashboard |
| `/api/runs` | every run at a glance |
| `/api/runs/<label>/<name>` | one run's summary, configuration and series |
| `/api/system` | the machine now and over the last hour |
