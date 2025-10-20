# Yggdrasil Decision Forests (Linux) — Python wheel build & usage

This README shows how to:

1. build a Python wheel from this repo with **Bazel** on Linux,
2. use the wheel from Python (with **kernel_method** exposed via `extra_training_config`),
3. run the **MIGHT** setting (bootstrapping + honest + kernel + OOB export), and

> **Status notes**
>
> * This branch **disables all fast_engine backends**. Call `model.predict(use_slow_engine=True)` or you will get an error.
> * `kernel_method` is plumbed through the learner via `extra_training_config` (see example below).

---

## 0) Prerequisites

* Linux (x86_64 or aarch64)
* Python 3.9–3.11 (recommended)
* Bazel = 6.5.0
* A virtual environment for installation (recommended)

```bash
# Example: create a venv named ydf-build
python3 -m venv ~/venvs/ydf-build
source ~/venvs/ydf-build/bin/activate
python -m pip install --upgrade pip wheel
```

---

## 1) Build the Python wheel with Bazel

From the repo root:

### A. Generic build (works on most Linux x86_64)

```bash
bazel build -- //ydf/...:all
./tools/package_linux.sh
pip install --force-reinstall dist/*.whl
```

### B. ARM/aarch64 build (e.g., Graviton; mirrors the current project setup)

```bash
# Uses a dedicated Bazel output base for Python builds and sets C++17 flags.
# Adjust the -march string for your target if needed.
bazel --output_base=$HOME/.cache/bazel/ydf_py \
  build \
  --copt=-march=armv8-a+crypto+crc \
  --host_copt=-march=armv8-a+crypto+crc \
  --cxxopt=-std=c++17 \
  --host_cxxopt=-std=c++17 \
  --jobs=1 \
  -- //ydf/...:all

./tools/package_linux.sh
pip install --force-reinstall dist/*.whl
```

After installation you can `python -c "import ydf, sys; print('YDF OK', ydf.__version__)"` to verify.

> If you’re debugging multiple local builds, consider `pip uninstall ydf` before reinstalling a new wheel.

---

## 2) Use from Python (enabling `kernel_method` )

The Python API follows the official docs for `RandomForestLearner` (see reference: [https://ydf.readthedocs.io/en/stable/py_api/RandomForestLearner/#ydf.RandomForestLearner](https://ydf.readthedocs.io/en/stable/py_api/RandomForestLearner/#ydf.RandomForestLearner)). This branch exposes a `kernel_method` flag in the random-forest training config via `extra_training_config`.

```python
from google.protobuf import text_format
import ydf

# Compose an embedded proto for the RF learner's extra training config.
extra_training_config = text_format.Parse(
    r"""
    [yggdrasil_decision_forests.model.random_forest.proto.random_forest_config] {
      # Toggle kernel_method here
      kernel_method: true

      # (Optional) Write OOB predictions to a CSV for later analysis
      export_oob_prediction_path: "csv:/tmp/oob_predictions.csv"
    }
    """,
    ydf.TrainingConfig(),
)

learner = ydf.RandomForestLearner(
    label="target",
    bootstrap_training_dataset=True,          
    bootstrap_size_ratio=1.6,                 
    num_trees=200,
    random_seed=42,
    winner_take_all=False,
    extra_training_config=extra_training_config,
)

model = learner.train(train_df)

# IMPORTANT: fast engines are disabled in this branch.
# Always set use_slow_engine=True.
y_pred = model.predict(test_df, use_slow_engine=True)
```


---

## 3) Run **MIGHT**

In this codebase, **MIGHT** means training with the following enabled:

* **Bootstrapping** (`bootstrap_training_dataset=True`, optional `bootstrap_size_ratio`)
* **Honest** trees (`honest=True`, with `honest_ratio_leaf_examples`, `honest_fixed_separation`)
* **Kernel** splits (`kernel_method: true` in `extra_training_config`)
* **OOB export** (`export_oob_prediction_path` in `extra_training_config`)

Minimal sketch:

```python
extra_training_config = text_format.Parse(
    r"""
    [yggdrasil_decision_forests.model.random_forest.proto.random_forest_config] {
      kernel_method: true
      export_oob_prediction_path: "csv:/tmp/might_oob.csv"
    }
    """,
    ydf.TrainingConfig(),
)

learner = ydf.RandomForestLearner(
    label="target",
    bootstrap_training_dataset=True,   # Bootstrapping
    honest=True,                       # Honest
    extra_training_config=extra_training_config,  # Kernel + OOB export
)
model = learner.train(train_df)

# Evaluate / predict
pred = model.predict(test_df, use_slow_engine=True)
```

---

## 6) Reference

* RandomForestLearner API: [https://ydf.readthedocs.io/en/stable/py_api/RandomForestLearner/#ydf.RandomForestLearner](https://ydf.readthedocs.io/en/stable/py_api/RandomForestLearner/#ydf.RandomForestLearner)

If anything fails, open an issue with: your CPU arch, Linux distro, Python version, Bazel version, full build command, and error log.
