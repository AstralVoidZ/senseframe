---
name: senseframe
description: >-
  Trains, evaluates, and exports ML models via a stage-based PyTorch Lightning pipeline.
  Use when the user wants to run supervised or self-supervised training, HPO, NAS
  (DARTS/ENAS), AutoAugment search, or comparison experiments. Supports custom strategy
  registration (loss/metric/task_type/model/scene/normalization), data-driven policy
  recommendation, resource-aware routing, ONNX/TorchScript export, pipeline resume from
  failed stages, and artifact integrity verification. Do not use for general PyTorch
  scripting, non-training ML tasks, or production serving orchestration.
---

# SenseFrame

Agent-driven AutoML training framework. Agent holds control over the training flow;
the framework provides composable primitives, an execution substrate, and safety guardrails.

## When to Invoke

- Train a model (supervised / self-supervised / HPO)
- Register custom strategies (loss / metric / task_type / model / scene / normalization)
- Profile data and let data characteristics drive strategy selection
- Compose a custom training flow via Stage Pipeline
- Run NAS, AutoAugment search, or meta-learning warm-start
- Run ε6 comparison experiments (Method vs Baseline)
- Export models to ONNX / TorchScript / state_dict
- Resume a failed pipeline from the failed stage
- Verify artifact integrity (SHA-256 manifest)
- Develop a new scene container

## Prerequisites

- Python 3.x + PyTorch + PyTorch Lightning installed
- Datasets placed in `CSI_DATASETS/` (WiFi CSI scene) or referenced via `data_root`
- Self-supervised mode supports NTU-Fi_HAR only; declare `num_classes: 14`
- For reproducibility: fix `trainer.seed`; strict mode `trainer.deterministic: true`
- **ONNX export** requires `pip install onnx` (optional dependency, not in requirements.txt)
- **Call `sf.activate_lazy_scenes()` before querying any registry** (CQS compliance:
  queries must not trigger registration side-effects). The CLI already does this; the
  imperative path must call it manually.

## Workflow

### Step 1: Identify the intent and path

- **Declarative path** — standard scenes: YAML config + `run_experiment`
- **Imperative path** — custom flow: `DataProfiler` + `register_*` + `Pipeline` + `load_extension`

Training modes: `supervised` / `self_supervised` (NTU-Fi_HAR only) / `hpo.enabled: true`.

### Step 2: Activate scenes and profile the data

```python
import senseframe as sf

sf.activate_lazy_scenes()        # REQUIRED before any registry query
sf.list_models()                  # now returns the full model list

profile = sf.DataProfiler(max_samples=500).profile_bundle(bundle, dataset_name="my_data")
# profile.recommended_task_type / recommended_loss / recommended_metrics / recommended_normalization

# Cache and reload (DataProfile is the data structure; DataProfiler is the probe)
profile.save("data_profile.json")
loaded = sf.DataProfile.load("data_profile.json")  # classmethod, returns DataProfile
```

### Step 3: Register custom strategies (optional)

When built-in strategies are insufficient, register instead of modifying framework source:

```python
sf.register_task_type("anomaly_detection", default_loss="bce_with_logits",
                      default_metrics=["accuracy"], description="异常检测")

@sf.register_loss("my_focal")
def _focal(alpha=0.25, gamma=2.0, **kw):
    import torch.nn as nn
    return nn.CrossEntropyLoss(**kw)
```

Registry API: `register_task_type` / `register_loss` / `register_metric` / `register_model` /
`register_dataset` / `register_normalization` / `register_scene`. All accept an `overwrite`
flag (default `True` for task_type/loss/metric, `False` for model/dataset/scene/normalization).

### Step 4: Compose the training flow

**Path A — Declarative YAML:**

```bash
python scripts/generate_config.py --dataset UT_HAR_data --model ResNet18 --mode supervised --output configs/exp.yaml
python scripts/validate_config.py --config configs/exp.yaml
python -m senseframe.cli experiment --config configs/exp.yaml
```

**Path B — Imperative Stage Pipeline:**

```python
pipeline = sf.Pipeline.default()
# 8 stages: validate → preflight → resolve → load → build → train → eval → export
pipeline.replace_stage("eval", my_custom_eval)
pipeline.before("train", data_check_hook)
pipeline.skip("export")

ctx = sf.PipelineContext(config=my_config)
result = pipeline.run(ctx)
```

**Path C — Factory injection:** pass `module_factory` / `datamodule_factory` /
`trainer_factory` / `extra_callbacks` to `ExperimentConfig`.

**Path D — Extension file:** `sf.load_extension("my_extension.py")` executes a file that
calls `register_*` directly (no `import senseframe` needed inside the file).

### Step 5: Probe resources and dry-run

```bash
python -m senseframe.cli probe
python -m senseframe.cli recommend --dataset UT_HAR_data --priority balanced
python -m senseframe.cli experiment --config configs/exp.yaml --dry-run
```

Dry-run checks: config validation, scene registration, dataset/model/learning-mode support,
resource detection, data existence, file-extension consistency, VRAM, and disk.

### Step 6: Execute training

```bash
python -m senseframe.cli experiment --config configs/exp.yaml
python -m senseframe.cli experiment --config configs/exp.yaml --retry            # OOM auto-halve batch_size
python -m senseframe.cli experiment --config configs/exp.yaml --export-formats onnx,torchscript
```

### Step 7: Verify artifacts and release resources

```python
from senseframe import verify_artifacts

report = verify_artifacts("runs/<exp>/")
# returns {artifact_name: bool}; False = missing or tampered

ctx.release_resources()  # REQUIRED after long runs / HPO / serial trials
```

`release_resources` cleans Trainer / DataModule / Loggers / GPU memory in order:
log_writer close → Logger finalize → Trainer `_teardown` → DataModule teardown →
model `.cpu()` → set None → CUDA empty_cache → gc.collect. HPO path calls it
automatically; the imperative path must call it manually.

### Step 8: Post-process and export

```bash
python scripts/postprocess.py --output-dir runs/<exp>
python -m senseframe.cli export \
    --metadata runs/<exp>/metadata.json \
    --checkpoint runs/<exp>/model.pth \
    --formats onnx,torchscript,state_dict \
    --output-dir exports/
```

`metadata.config` is a complete config snapshot (declarative `ExperimentConfig` +
resolved `ctx.resolved`, including `data_root` / `epochs` / `seed` / `learning_mode`)
written by `stage_export`; it is the source of truth for experiment reproduction and
inference-script generation.

## Output Contract

`TrainOutput` fields (use attribute access; `training` / `env_snapshot` / `feedback` are
typed dataclasses):

| Field | Type | Notes |
|-------|------|-------|
| `status` | `str` | `"success"` / `"error"` |
| `final_eval` | `Dict[str, Any]` | Final validation metrics |
| `training` | `Optional[TrainingSummary]` | `.epochs_trained` / `.early_stopped` / `.duration_s` / `.best_val_loss` |
| `env_snapshot` | `Optional[EnvSnapshot]` | `.torch` / `.pytorch_lightning` / `.cuda` / `.python` / `.deterministic` / `.seed` |
| `feedback` | `Optional[FeedbackResult]` | `.status` ∈ {converged, success, overfitting, underfitting, numerical_instability} / `.diagnosis` / `.suggestions` |
| `model_path` | `Optional[str]` | Best checkpoint path |
| `output_dir` | `Optional[str]` | Output directory |
| `error_code` | `Optional[str]` | Structured error code for programmatic branching |
| `error` / `error_traceback` | `Optional[str]` | Error message and traceback on failure |

Serialize via `output.to_dict()`; type validation runs at construction points via
`validate_feedback` / `validate_training_summary` / `validate_env_snapshot`.

## Error Handling

All training errors raise a `SenseFrameError` subclass carrying an `error_code` class
attribute. Branch on `error_code`, never on string matching.

| error_code | Exception | Action |
|------------|-----------|--------|
| `CONFIG_VALIDATION_ERROR` | `ConfigValidationError` | Fix config; do not retry |
| `CONFIG_PARSE_ERROR` | `ConfigValidationError` (from_dict stage) | Fix YAML structure / field types |
| `CONFIG_NOT_FOUND` | — (CLI/validate_config) | Check `--config` path |
| `MISSING_CONFIG` | — (CLI) | Provide `--config` argument |
| `INVALID_CONFIG_FORMAT` | — (CLI/validate_config) | YAML top-level must be a mapping |
| `UNSUPPORTED_FORMAT` | — (CLI export_formats) | Check `export_formats` values |
| `SCENE_NOT_FOUND` | `SceneNotRegisteredError` | Check `scene.name` |
| `DATASET_NOT_SUPPORTED` | `DatasetNotSupportedError` | Check dataset name |
| `MODEL_NOT_SUPPORTED` | `ModelNotSupportedError` | Check `model_id` |
| `DATA_NOT_FOUND` | `DataNotFoundError` | Check `data_root`; do not retry |
| `DATA_LOAD_ERROR` | `DataCorruptedError` | Check data integrity / format / permissions |
| `OOM_ERROR` | `OOMError` | Halve `batch_size` and retry |
| `CHECKPOINT_ERROR` | `CheckpointError` | Check checkpoint path / version / integrity |
| `PREFLIGHT_ERROR` | `PreflightError` | Upgrade hardware or pick a smaller model |
| `TRAINING_ERROR` | `TrainingError` | Inspect traceback |
| `MODEL_BUILD_ERROR` | `ModelBuildError` | Check `model_id` |
| `SAVE_ERROR` | `SaveError` | Check disk space / permissions |
| `METADATA_NOT_FOUND` | — (CLI predict) | Check `--metadata` path; produced by training |
| `METADATA_VERSION_ERROR` | `MetadataVersionError` | metadata.json schema_version incompatible; upgrade SenseFrame or use legacy metadata |
| `UNKNOWN_ERROR` | `SenseFrameError` (base) | Fallback classification; inspect traceback |

```python
from senseframe.engine.runner.errors import SenseFrameError, OOMError

try:
    output = sf.run_experiment(config)
except OOMError as e:
    config.trainer.batch_size //= 2
    output = sf.run_experiment(config)
except SenseFrameError as e:
    print(f"[{e.error_code}] {e}")
```

## Search Protocol (SP) — Ask/Tell

All search-driven capabilities (HPO / NAS / AutoAugment / ε6 Method / meta-learning) run
through the SP `ask`/`tell` interface. Build a `SearchSpace`, create a `Study`, then loop:

```python
from senseframe.search_protocol import ParameterSpec, SearchSpace, StudyManager

space = SearchSpace(parameters=[
    ParameterSpec(name="lr", type="float", low=1e-4, high=1e-2, log=True),
    ParameterSpec(name="batch_size", type="int", low=8, high=64, step=8),
])

sm = StudyManager()
study_id = sm.create_study(name="hpo_run", direction="maximize",
                           search_space=space, sampler="random")

for _ in range(n_trials):
    trial = sm.ask(study_id)
    config = apply_params(base_config, trial.params)
    output = sf.run_experiment(config)
    sm.tell(trial.trial_id, value=output.final_eval.get("val_accuracy", 0.0))

best = sm.best_trial(study_id)
```

Built-in samplers: `random` / `grid` / `tpe` / `asha` / `hyperband`. NAS samplers
(`darts` / `enas` / `evolutionary`) and `autoaugment` require explicit module import.
Built-in pruners: `asha` / `hyperband` (each also implements the Sampler protocol).

## Introspection

Query pipeline / context / data contracts before composing a pipeline; do not read source.

```python
sf.context_schema()              # PipelineContext field contract (fill_stage per field)
sf.stage_io("train")             # reads / writes of one stage (no `stage_` prefix)
sf.list_stages()                 # all stage names
sf.pipeline_graph()              # DAG of field producers/consumers
sf.data_bundle_schema()          # DatasetBundle filling rule per learning_mode
sf.data_profile_schema()         # DataProfile field contract
```

## Resume from a Failed Stage

```python
from senseframe.engine.runner.pipeline import Pipeline

pipeline, completed = Pipeline.resume("runs/<exp>")   # reads pipeline_checkpoint.json
ctx = sf.PipelineContext(config=my_config)
ctx.completed_stages = completed
ctx.stage_checkpoint_path = Path("runs/<exp>/pipeline_checkpoint.json")
result = pipeline.run(ctx)        # skips completed stages, resumes from the failed one
```

## Scene Development

Inherit `SceneContainer` and implement four abstract methods:
`meta` / `load_dataset` / `build_model_for_dataset` / `get_dataset_info`. Optional overrides:
`get_task_spec` / `get_feature_spec` / `get_scene_params` / `get_transforms` /
`get_search_space` / `get_default_config` / `get_model_info` / `normalize` / `postprocess`.

See `reference/scene_development.md` for the full guide and a minimal scene example.

## CLI

All commands emit structured JSON.

| Command | Purpose |
|---------|---------|
| `probe` | Probe hardware resources |
| `list-models` | List available models (filter by `--dataset`) |
| `list-datasets` | List available datasets |
| `list-scenes` | List scene containers |
| `paradigms` | List SOTA paradigms (filter by `--category`) |
| `recommend` | Recommend models by resource and dataset |
| `experiment` | Train from a YAML config (`--dry-run` to preflight only) |
| `export` | Export a model to ONNX / TorchScript / state_dict |

## Gotchas

- **Never query a registry without `activate_lazy_scenes()` first.** The CLI does this
  internally; the imperative path must call it explicitly. Symptom: empty `list_models()`.
- **Stage names in `sf.stage_io()` / `pipeline.check_readiness()` omit the `stage_` prefix**
  (`"train"`, `"eval"`). `ctx.filled_at()` uses the prefixed form (`"stage_load"`).
- **`TrainOutput.training` / `env_snapshot` / `feedback` are dataclass instances, not dicts.**
  Use attribute access (`output.training.epochs_trained`, `output.feedback.status`).
  Serialize via `output.to_dict()`.
- **`SceneConfig.params` is `Optional[SceneParams]`.** It supports dict-like access
  (`[]` / `in` / `.get()` / `.items()`) for backward compatibility, but new code should
  use attribute access. Before HPO assignment, check `is None` and create an empty instance.
- **Call `ctx.release_resources()` after long runs / HPO / serial trials.** HPO calls it
  automatically; the imperative path does not. Thread / handle / pipe / VRAM leaks result.
- **Call `orch.shutdown()` on every `Orchestrator`.** It closes the internal
  ThreadPoolExecutor. Wrap usage in try/finally; `subscribe` returns an unsubscribe
  function that should also be called in the finally block.
- **NAS resource management.** `DARTSPipelineRun.run()` uses try/finally to release the
  supernet / optimizer / iterator. `DARTSSampler.update()` uses `.detach().clone()` to
  cut the computation graph. Do not bypass these.
- **`postprocess.py` accepts only `--output-dir`.** All post-processing artifacts land
  inside `output_dir`; the manifest stores relative paths.
- **Catch `SenseFrameError` subclasses and branch on `error_code`.** Do not match on
  error message strings — messages may change across versions.
- **`verify_artifacts(dir)` takes a directory path** and returns `{artifact_name: bool}`.
  False means missing or tampered.

## References

Load on demand; do not read all at once.

| Document | When to read |
|----------|--------------|
| `reference/config_schema.md` | Writing or validating a YAML config |
| `reference/training_templates.md` | Generating a config from a template |
| `reference/datasets_and_models.md` | Choosing a dataset or model |
| `reference/resource_routing.md` | Probing resources, picking a model, or preflighting |
| `reference/self_supervised_paradigm.md` | Running self-supervised training |
| `reference/scene_development.md` | Developing a new scene container |
| `reference/troubleshooting.md` | Diagnosing a training failure or enumerating error codes |
| `reference/introspect.md` | Querying field contracts, exploration state, skills, or resume API |
