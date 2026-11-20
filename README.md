# HyperMSDE: Hyperspherical Mean Shift Density Enhancement

HyperMSDE is a training-free, unsupervised framework for zero-shot anomaly detection that exploits the hyperspherical geometry of pretrained image embeddings through mean-shift density enhancement.

## Setup

```bash
git clone https://github.com/nveshaan/hypermsde.git
cd hypermsde
uv sync
```

### Environment variables

Paths in `configs/` are read from environment variables (`${oc.env:NAME,default}`), so each machine sets its own. All are optional.

| Variable | Used for | Default |
|---|---|---|
| `HMSDE_DATA` | Raw images and embeddings (`<HMSDE_DATA>/<modality>/<dataset>/`). `generate_embeddings.py` reads the images and writes the embeddings and labels; `run_hypermsde.py` and `run_baseline.py` read them | `data` (relative to the repo) |
| `HMSDE_THIRD_PARTY` | Where UniMed-CLIP and the baseline repos (AnomalyCLIP, FAPrompt, AdaCLIP) are cloned | `.third_party` (relative to the repo) |

```bash
# e.g. in ~/.zshenv
export HMSDE_DATA="$HOME/Developer/delta/data"
export HMSDE_THIRD_PARTY="$HOME/Developer/delta/.third_party"
```

A variable can still be overridden for one run on the command line, e.g. `data_root=/path/to/data` for `run_baseline.py`.

## Data

The datasets have to be downloaded externally and are to be placed in the `data/` folder in the following way:

```bash
./data
├── chest
│   ├── cardiomegaly_set
│   ├── covid_affected_and_normal
│   ├── COVID-19_Radiography_Dataset
│   ├── emphysema
│   ├── RSNA
│   └── VinCXR
├── fundus
│   ├── AMDNet23 Dataset
│   ├── Diabetic Retinopathy and Retinal Diseases Fundus I
│   ├── Eye Disease Image Dataset
│   ├── G1020
│   ├── LAG
│   ├── ORIGA
│   └── Retinal Fundus Images
├── mri
│   ├── Alzheimer_s Dataset
│   ├── Br35H
│   ├── Brain Tumor MRI Dataset
│   ├── brain_tumor_dataset
│   ├── BrainTumor
│   ├── BraTS2021
│   ├── brisc2025
│   ├── Epic and CSCR hospital Dataset
│   └── OASIS Alzheimer's Detection
└── oct
    ├── ARMD OCT
    ├── OCT2017
    ├── Retinal OCT images
    ├── RetinalOCT_Dataset
    └── ZhangLabData OCT
```

## Generate embeddings

Generate embeddings with all configured encoders and modalities with:

```bash
uv run python scripts/generate_embeddings.py
```

Use `uv run python scripts/generate_embeddings.py --help` for all command-line options.

The generated files are written to
`data/<modality>/<dataset_name>/<encoder>_embeds.npy` and
`data/<modality>/<dataset_name>/labels.npy`.

> **MedSigLIP** is a gated model. Before running this, request access on its model page on the HF Hub, then authenticate locally with:
> `hf auth login`
> (or set the `HF_TOKEN` environment variable).

## Run HyperMSDE

`scripts/run_hypermsde.py` runs Mean Shift Density Enhancement on the embeddings of an encoder, using the settings in `configs/hypermsde.yaml`. The movement of each point is its anomaly score (label 0 is normal, any other label is anomalous).

```bash
# One dataset.
uv run python scripts/run_hypermsde.py modality=chest encoder=CLIP dataset=VinCXR

# Every dataset listed in configs/datasets/<modality>.yaml (one MLflow run each).
uv run python scripts/run_hypermsde.py modality=chest encoder=CLIP dataset=null

# Three runs; runs after the first use pseudo labels from the previous one (neighbours must be presumed normal).
uv run python scripts/run_hypermsde.py pseudo_labels.rounds=3 pseudo_labels.anomaly_fraction=0.2

# Override HyperMSDE hyperparameters.
uv run python scripts/run_hypermsde.py model.k=20 model.learning_rate=0.1
```

The density, score, weight and stability used are selected by the config groups in `configs/{density,score,weight,stability}/`.

Everything is logged with MLflow to `experiments/mlflow.db` (artifacts in `experiments/artifacts`), view it with `uv run mlflow ui --backend-store-uri sqlite:///experiments/mlflow.db`:

- **Params:** dataset, modality, encoder, density/score/weight/stability names, seed, distance, sample counts and the model hyperparameters.
- **Metrics:** AUROC, AUPRC and P@n (n = number of anomalies), each also computed on the negated score (`*_flipped`), since the direction of the movement score depends on the encoder. For datasets with several anomaly labels, the same metrics are logged per subtype as `subtype_<label>/<metric>` (each subtype against the normal samples). With several rounds, metrics are logged per round (`step`).
- **Artifacts:** the resolved config and the density, score, weight and stability configs under `configs/`.

## Run baselines

`scripts/run_baseline.py` runs an inference-only (zero-shot) baseline on the **raw images** of a dataset, on the same samples as `run_hypermsde.py` (same labels and cap). The method is a config in `configs/baselines/`: `anomalyclip`, `faprompt` or `adaclip`. Their official repos are cloned to `.third_party/` and the released checkpoints downloaded on first use. Raw images are read from `HMSDE_DATA` (see Environment variables), so run it where they are.

```bash
uv run python scripts/run_baseline.py baselines=adaclip modality=chest dataset=VinCXR

# Every dataset of a modality, another released checkpoint.
uv run python scripts/run_baseline.py baselines=anomalyclip baselines.checkpoint=visa modality=oct dataset=null
```

Checkpoints are chosen with `baselines.checkpoint`; selecting one the authors did not release (e.g. `medical` for AnomalyCLIP and FAPrompt) fails with the list of available ones. AdaCLIP's released weights were all trained on industrial and medical data (`mvtec_colondb`, `visa_clinicdb`, `all`).

**Encodings are never computed twice.** Each baseline splits its work into `encode` (the model on an image) and a cheap head on the result. A few-shot baseline is given lists, `supports=[1,2,4,8,16] support_seeds=[0,1,2,3,4]`, and runs every (K, seed) in one pass over the images: each image is encoded once and scored by all the supports (the support images are encoded once and kept in memory). The zero-shot baselines also keep each image's output on disk under `experiments/cache/` (`cache_dir`; AnomalyCLIP's is the image feature, so another checkpoint costs no image inference). The few-shot baselines' outputs are 1.5-9 MB per image, so `cache_encodings: false` in their configs; set it to `true` to keep them on disk too.

Runs go to the `baseline` MLflow experiment with the same metrics as HyperMSDE (including `*_flipped` and per-subtype). Params are only dataset, modality, method, seed and sample counts; the checkpoint is a tag, and the resolved configs are stored under `configs/` as artifacts.

## Acknowledgements

- Kar, P., Bordoloi, R., Wolkenhauer, O., & Bej, S. (2026). Anomaly Detection via Mean Shift Density Enhancement. arXiv:2602.03293. https://doi.org/10.48550/arXiv.2602.03293

- Kar, P., Lakshmi, G., & Bej, S. (2026). Improved Anomaly Detection in Medical Images via Mean Shift Density Enhancement. arXiv:2604.19191. https://doi.org/10.48550/arXiv.2604.19191

## License

This project is distributed under the MIT License. See the `LICENSE` file for details.
