# Fungal AMP Discovery
### Leveraging Machine Learning to Uncover Broad-Spectrum Antimicrobial Peptides from Pathogenic Fungal Genomes: A case study

## Environment Setup

### Create Conda Environment

```bash
conda create -n uniamp python=3.10
conda activate uniamp
```

### Install Dependencies

```bash
# Install PyTorch according to your GPU version
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128

# Install other dependencies
pip install scikit-learn pandas tape_proteins sentencepiece tqdm
```

### Install ESM-2 and ESM-C

> **Note:** The pip packages for ESM-2 and ESM-C have naming conflicts. Please follow the steps below carefully.

1. Install ESM-2:

```bash
pip install fair-esm
```

2. Rename the ESM-2 package to avoid conflicts:

Rename the `esm` directory under `~/miniconda3/envs/uniamp/lib/python3.10/site-packages/` to `esm2`, and replace all `import esm` statements in the files within that directory with `import esm2`.

3. Install ESM-C:

```bash
pip install esm
```

### Download Pretrained Models

- **ProtT5**: Download all files from [Rostlab/prot_t5_xl_uniref50](https://huggingface.co/Rostlab/prot_t5_xl_uniref50/tree/main) and place them in the `./prott5_pre_models` directory.
- **ESM-C**: Download `esmc_600m_2024_12_v0.pth` from [EvolutionaryScale/esmc-600m-2024-12](https://huggingface.co/EvolutionaryScale/esmc-600m-2024-12) and place it in the `./` root directory.

## Feature Extraction

Run `cal_features.py` to compute PLM features. The output files are saved in the same directory as the input file, with the `.fasta` extension replaced by `.pkl`.

```bash
python cal_features.py
```

## Usage

Use `run.py` as the entry point. It supports four modes: pretraining, training, testing, and inference.

### Parameters

#### General

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--device` | str | `cuda:0` | Device for computation, e.g., `cuda:0` for the first GPU or `cpu` for CPU only |
| `--model` | str | `uniamp` | Model type. Choices: `bert` (for pretraining), `uniamp` |
| `--mode` | str | `train` | Operation mode. Choices: `train`, `test`, `pretrain`, `infer` |
| `--dataset_path` | str | `./data/amp/training_dataset.fasta` | Path to the dataset file (`.fasta` format) |

#### Feature

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--feature` | str (multiple) | `bert unirep esm2 prott5` | PLM feature(s) to use. Choices: `bert`, `unirep`, `esm2`, `prott5`, `esmc`. Multiple features can be specified separated by spaces, e.g., `--feature unirep esmc` |

#### Save & Load

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--save` | str | `{timestamp}_model.pth` | Filename for saving the trained model (default: timestamp-based) |
| `--log` | str | `{timestamp}_train.log` | Filename for saving training logs (default: timestamp-based) |
| `--checkpoint` | str | `None` | Path to a checkpoint file. In `pretrain`/`train` mode, this is used to resume training from a previously saved checkpoint. In `test`/`infer` mode, this is used to specify the trained model path for evaluation or prediction |
| `--save_all` | flag | `False` | Save the model at every epoch. By default, only the best model is saved |

#### Training

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--lr` | float | `0.0001` | Learning rate for the optimizer |
| `--epochs` | int | `300` | Number of training epochs |
| `--patience` | int | `30` | Early stopping patience. Training stops after this many consecutive epochs without improvement |
| `--train_batch_size` | int | `128` | Batch size for training |
| `--val_batch_size` | int | `128` | Batch size for validation |
| `--val_pro` | float | `0.2` | Proportion of the dataset used for validation (default: 20%) |

#### Random Seed

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--data_random_seed` | int | `42` | Random seed for dataset splitting (train/validation) |
| `--model_random_seed` | int | `42` | Random seed for model parameter initialization |

#### Inference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--batch_infer` | flag | `False` | Only effective in `infer` mode. When enabled, pre-computed `.pkl` feature files will be used for batch inference, which is significantly faster. If not set, features will be computed on-the-fly in real time, which may be considerably slower |

### Examples

**Pretraining:**

```bash
python run.py --mode pretrain --dataset_path ./data/amp/pretraining_without_test_dataset.fasta
```

**Training:**

```bash
python run.py --mode train --dataset_path ./data/amp/training_dataset.fasta
```

**Testing:**

```bash
python run.py --mode test --dataset_path ./data/amp/test_dataset.fasta --checkpoint ./models/20260318_2152_model/20260318_2152_model_2.pth
```

**Inference:**

```bash
python run.py --mode infer --dataset_path ./data/amp/test_dataset.fasta --checkpoint ./models/20260318_2152_model/20260318_2152_model_2.pth
```