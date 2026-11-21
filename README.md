# Multi-Label Aerial Image Classification with Scene Labels on AID

CS415 course project, Sabancı University (Group 10).

In the multi-label AID dataset only 3,000 images have object labels (buildings, cars, trees, ...), but all 10,000
images of the original AID dataset have a scene label (airport, port, river, ...). We study whether these cheap
scene labels help a model recognize objects. The project proposal is in [`docs/proposal.pdf`](docs/proposal.pdf).

## Models

| Model | Description |
|---|---|
| B0 | ImageNet-pretrained ResNet-50 with 16 sigmoid outputs, trained on the multi-label images |
| B1 | B0 + a scene head trained on all scene-labeled images (multi-task) |
| B2 | B1 + the scene prior P(object \| scene) added to the object logits |
| B3 | Scene-routed experts (SR-MoE), planned for the final report |

Three control runs check where the gain of B1 comes from: **B1_noextra** (scene supervision on the multi-label
training images only, i.e. the images of B0), **B1_shuffled** (same images as B1, randomly permuted scene
labels) and **B0_long** (B0 trained for 40 epochs, so it sees as many images as B1).

A pilot reduced-label experiment trains B0, B0_long, B1 and B2 with only 25% of the object labels
(`*_labels25` runs).

## Repository

```
01_data_preparation.ipynb   download, scene matching, split, cached images, dataset figures
02_train_models.ipynb       trains B0, B1 and B2 for every seed
03_control_experiments.ipynb trains the control runs B1_noextra, B1_shuffled and B0_long
04_reduced_labels.ipynb     pilot: B0, B0_long, B1 and B2 with 25% of the object labels
05_results.ipynb            tables, significance tests and figures for the report
aidml.py                    shared code: data loading, model, training, metrics
environment.yml             conda environment
docs/                       proposal
results/                    metrics, training histories and test predictions (written by the notebooks)
figures/                    figures for the report (written by the notebooks)
data/, checkpoints/         downloaded data and model weights (not tracked)
```

## Setup

```bash
conda env create -f environment.yml
conda activate cs415-aid
```

Select the `cs415-aid` environment as the notebook kernel (VS Code or `jupyter notebook`).
Training uses Apple Silicon (MPS) or CUDA if available, otherwise the CPU.

## Running

Run the notebooks in order from the repository root:

1. `01_data_preparation.ipynb` downloads about 3 GB from Hugging Face
   ([multi-label AID](https://huggingface.co/datasets/jonathan-roberts1/AID_MultiLabel),
   [AID](https://huggingface.co/datasets/blanchon/AID)) and writes `data/processed/`.
2. `02_train_models.ipynb` trains B0, B1 and B2 (about 2.5 hours per seed on an M2).
3. `03_control_experiments.ipynb` trains the control runs (about 2.1 hours per seed).
4. `04_reduced_labels.ipynb` trains the reduced-label pilot (about 50 minutes per seed).
5. `05_results.ipynb` builds the tables and figures; it does not train anything.

Finished runs are skipped, so the training notebooks can be stopped and restarted.

## Data split

*mobile home* is removed (2 images in the whole dataset), leaving 16 labels. Inside each scene the multi-label
images are split 72 / 8 / 20 % into train / validation / test. The remaining 7,000 AID images are used only to
train the scene head. Validation and test images (and their scene labels) are never used for training, and
at test time every model uses its own predicted scene.

## Team

Öykü Demirel · Yağız Çetin · Erkan Ulaş Tepe · Bilge Kağan Yılmaz
