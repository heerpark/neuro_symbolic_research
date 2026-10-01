# third_party/ — vendored official baseline code

Both repos are git-cloned locally (gitignored; re-clone with the commands below rather than
committing vendored code). Local modifications are never made in place without a patch file
recorded here.

## DeepSoftLog

- Repo: https://github.com/jjcmoon/DeepSoftLog
- Pinned commit: `87765eda25dc12b5faf6dbda71aa9fea3a0b2ac0` (2024-12-13, HEAD at clone time)
- Clone: `git clone https://github.com/jjcmoon/DeepSoftLog third_party/deepsoftlog && cd third_party/deepsoftlog && git checkout 87765eda25dc12b5faf6dbda71aa9fea3a0b2ac0`
- Env: conda env `dsl_env` (python 3.10), built via:
  ```
  pip install cython==0.29.36
  python setup.py build_ext --inplace
  pip install -r requirements.txt
  ```
- **Environment fix (not a source patch): PySDD.** `requirements.txt` pins `pysdd==0.2.11`,
  whose prebuilt macOS `libsdd.a` (shipped inside the wheel/sdist, not rebuilt by
  `setup.py`) is missing symbols on Apple Silicon (`dlopen` fails with
  `symbol not found in flat namespace '_add_var_after_lca'`, confirmed by extracting the
  sdist and tracing it to `pysdd/lib/sdd_extra/include/sddapi_extra.h`'s declarations having
  no matching compiled object in the arm64 Darwin static lib). Building from source
  reproduces the same failure (the extra-API `.c` implementation isn't even shipped, only
  the prebuilt `.a`). Fix: install `pysdd==1.0.6` instead (native `macosx_11_0_arm64` wheel
  on PyPI) — verified import-compatible with this pinned DeepSoftLog commit. No DeepSoftLog
  source was changed for this.
- Also needed: `pip install "setuptools<81"` (wandb's vendored `pkg_resources` import breaks
  under setuptools>=81 in this env).
- **Config note:** the shipped `deepsoftlog/experiments/mnist_addition/config.yaml` has
  `nb_epochs: 3`, but the paper's Appendix D.1 (Table 5) documents 100 epochs as the setting
  used to obtain the reported 98.4±0.1 (N=1) / 96.6±0.3 (N=2) numbers. Our
  `configs/deepsoftlog_n1.yaml` / `configs/deepsoftlog_n2.yaml` override `nb_epochs: 100`
  accordingly (`configs/deepsoftlog_n2.yaml` is otherwise hand-authored with `digits: 2`,
  since no N=2 config ships in the repo). The repo also fixes `seed: 1337` in its shipped
  config; our runner (`scripts/run_deepsoftlog.py`) overrides `seed` per invocation.
- No source code modifications; `scripts/run_deepsoftlog.py` drives training by calling
  `deepsoftlog.experiments.mnist_addition.addition.train(cfg_dict)` directly (this function
  already accepts a plain dict, per `load_config`'s own `isinstance(cfg, dict)` branch) and
  swaps in a local `JsonlLogger` (same `log()`/`log_eval()`/`print()` interface as their
  `WandbLogger`) so no W&B account/network is needed.

## A-NeSI

- Repo: https://github.com/HEmile/a-nesi
- Pinned commit: `2c55d614132b40f25c69d09b4e140e80d21120b1` (2025-01-27, HEAD at clone time)
- Clone: `git clone https://github.com/HEmile/a-nesi third_party/anesi && cd third_party/anesi && git checkout 2c55d614132b40f25c69d09b4e140e80d21120b1`
- Env: conda env `anesi_env`, built via a locally-adapted copy of their `environment.yml`
  (`third_party/patches/anesi/environment.osx-arm64.yml`): identical except the `mkl=2022.1.0`
  pin is dropped (conda-forge doesn't publish `mkl` — an Intel-only package — for
  `osx-arm64`; the env already includes `libopenblas`/`libblas`/`liblapack`, so `pytorch`
  resolves its BLAS backend from those instead). Then `pip install -e .`, which pulls
  unpinned `torch`/`torchvision` from `setup.py`'s `install_requires` and silently upgrades
  past the paper-pinned versions — re-pin immediately after with
  `pip install --no-deps torch==1.12.1 torchvision==0.13.1` (both have native arm64 wheels
  on PyPI).
- **Source patch (`third_party/patches/anesi/0001-fix-train_all-TrainResult-unpacking.patch`,
  applied to the clone):** `anesi/experiments/mnist_op/mnistop.py` does
  `loss_nrm, loss_percept = model.train_all(x, label)`, but `ANeSIBase.train_all` (in
  `anesi/anesi.py`) returns a `TrainResult` dataclass, which isn't iterable/unpackable as a
  2-tuple. Confirmed this is a genuine upstream regression, not specific to our run: it
  reproduces identically even using the repo's own shipped defaults, and it already exists
  at the "Version 2" commit (`4366c29`, 2023-05-16) — i.e. `mnistop.py`'s direct
  `python mnistop.py --config ...` path for the MNIST-addition experiment appears to have
  never been fixed up after `TrainResult` was introduced, unlike the VisuDo experiment
  script (`a19409a fix: Visudo experiments not up to date with latest API`). Fix: unpack the
  two needed fields explicitly (`_train_result.q_loss`, `_train_result.percept_loss`) —
  behavior-preserving, no semantics changed.
- **Config note:** `mnistop.py`'s own default dict and every `repeat/*.yaml` in the repo
  (including `test_add_predict_only.yaml`, the "A-NeSI (predict)" config we target) use the
  key `amt_samples`, but `ANeSIBase.__init__` requires a parameter named `amount_samples`
  (no default) — confirmed by running the unmodified code, which raises
  `TypeError: __init__() missing 1 required positional argument: 'amount_samples'` even with
  the shipped defaults. This looks like the same "Version 2" naming drift, present in every
  sweep/repeat config across the whole repo (not just MNIST-addition), and is likely why
  `nrm_lr`/`nrm_loss` are also dead keys everywhere (see `configs/anesi_n1.yaml` for that
  detail). Fixed on our side only: `configs/anesi_n{1,2}.yaml` use `amount_samples` directly
  (no repo patch needed, since `ANeSIBase.__init__(**kwargs)` absorbs the unused key
  harmlessly if left in).
- `mnistop.py` is a `__main__`-only script with no importable `train()` entrypoint and no
  seeding hook. `scripts/run_anesi.py` runs it via `subprocess` (through a small shim that
  seeds `torch`/`numpy`/`random` before executing the script's own code — data order itself
  is already deterministic in this codebase: `shuffle=False`, no seed passed to the
  dataset), with `WANDB_MODE=disabled` (no account/network needed), and recovers metrics by
  parsing the script's own per-epoch stdout print (a wandb.log monkeypatch was tried first
  but silently discarded by `wandb.init()` rebinding `wandb.log` internally).
- A-NeSI variant used: **predict** (`experiments/mnist_op/repeat/test_add_predict_only.yaml`),
  per project decision — the four Table 1 variants (predict/explain/pruning/no-prior) don't
  cleanly match the project's originally-given reference range, and predict is the
  simplest/most commonly cited one.
