"""Shared code for the AID multi-label experiments: data loading, models, training and metrics."""

import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score
from torchvision.models import ResNet50_Weights, resnet50

PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("results")
CHECKPOINT_DIR = Path("checkpoints")

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------- data

def load_data():
    """Load the cached images and labels created by 01_data_preparation.ipynb."""
    meta = pd.read_csv(PROCESSED_DIR / "metadata.csv")
    with open(PROCESSED_DIR / "labels.json") as f:
        names = json.load(f)
    objects = meta[names["objects"]].to_numpy(dtype=np.float32)
    return {
        "images": np.load(PROCESSED_DIR / "images_224.npy"),
        "meta": meta,
        "object_names": names["objects"],
        "scene_names": names["scenes"],
        "objects": np.nan_to_num(objects),        # 0 for images without object labels
        "has_objects": ~np.isnan(objects[:, 0]),  # True for the 3,000 multi-label images
        "scenes": meta["scene_id"].to_numpy(),
        "split": meta["split"].to_numpy(),
    }


def split_indices(data, name):
    return np.where(data["split"] == name)[0]


def scene_object_table(data, smoothing=1.0):
    """P(object | scene) estimated on the training split, shape (num_scenes, num_objects)."""
    train = split_indices(data, "train")
    num_scenes = len(data["scene_names"])
    counts = np.zeros((num_scenes, len(data["object_names"])))
    totals = np.zeros(num_scenes)
    for i in train:
        counts[data["scenes"][i]] += data["objects"][i]
        totals[data["scenes"][i]] += 1
    return (counts + smoothing) / (totals[:, None] + 2 * smoothing)


def reduce_labels(data, fraction, seed=0):
    """Keep the object labels of a fraction of the training images, the same fraction in every scene.

    The other training images keep only their scene label and become scene-only images like the 7,000 extra
    images. Validation and test images do not change. The subset is fixed (seed 0), so all models and seeds use
    the same labeled images.
    """
    rng = np.random.default_rng(seed)
    split = data["split"].copy()
    objects = data["objects"].copy()
    has_objects = data["has_objects"].copy()
    for scene in np.unique(data["scenes"]):
        idx = np.where((split == "train") & (data["scenes"] == scene))[0]
        dropped = rng.permutation(idx)[round(fraction * len(idx)):]
        split[dropped] = "extra"
        objects[dropped] = 0
        has_objects[dropped] = False
    return {**data, "split": split, "objects": objects, "has_objects": has_objects}


def make_batch(images, idx, device, augment):
    """Convert uint8 images to a normalized float tensor. Augmentation: random 90-degree rotation and flip."""
    x = torch.from_numpy(images[idx]).to(device).permute(0, 3, 1, 2).float() / 255
    if augment:
        augmented = []
        for image in x:
            image = torch.rot90(image, random.randint(0, 3), dims=(1, 2))
            if random.random() < 0.5:
                image = image.flip(2)
            augmented.append(image)
        x = torch.stack(augmented)
    x = (x - IMAGENET_MEAN.to(device)) / IMAGENET_STD.to(device)
    return x.contiguous()


# ---------------------------------------------------------------- model

class MultiLabelNet(nn.Module):
    """ResNet-50 with an object head and, optionally, a scene head and a scene prior.

    B0: object head only.
    B1: object head + scene head (multi-task).
    B2: B1 + the scene prior added to the object logits.
    Control runs use the architecture of B0 (B0_long) or B1 (B1_noextra, B1_shuffled), see run_experiment.
    """

    def __init__(self, num_objects, num_scenes=0, prior_table=None, pretrained=True):
        super().__init__()
        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        self.backbone.fc = nn.Identity()
        self.object_head = nn.Linear(2048, num_objects)
        self.scene_head = nn.Linear(2048, num_scenes) if num_scenes > 0 else None
        if prior_table is not None:
            self.register_buffer("prior_table", torch.tensor(prior_table, dtype=torch.float32))
            self.prior_weight = nn.Parameter(torch.zeros(num_objects))
        else:
            self.prior_table = None

    def forward(self, x):
        features = self.backbone(x)
        object_logits = self.object_head(features)
        scene_logits = self.scene_head(features) if self.scene_head is not None else None
        if self.prior_table is not None:
            # The scene head is trained only by the scene loss, so its output is detached here.
            scene_probs = scene_logits.softmax(dim=1).detach()
            prior = scene_probs @ self.prior_table
            object_logits = object_logits + self.prior_weight * torch.logit(prior)
        return object_logits, scene_logits


def build_model(model_name, data, pretrained=True):
    num_objects = len(data["object_names"])
    num_scenes = len(data["scene_names"])
    if model_name in ["B0", "B0_long"]:
        return MultiLabelNet(num_objects, pretrained=pretrained)
    if model_name in ["B1", "B1_noextra", "B1_shuffled"]:
        return MultiLabelNet(num_objects, num_scenes, pretrained=pretrained)
    if model_name == "B2":
        return MultiLabelNet(num_objects, num_scenes, scene_object_table(data), pretrained=pretrained)
    raise ValueError(model_name)


# ---------------------------------------------------------------- metrics

def compute_metrics(probs, targets, threshold=0.5):
    """mAP and the example-based scores used by Hua et al. (2020)."""
    targets = targets.astype(bool)
    preds = probs >= threshold

    per_label_ap = [average_precision_score(targets[:, j], probs[:, j]) if targets[:, j].any() else np.nan
                    for j in range(targets.shape[1])]

    tp = (preds & targets).sum(1)
    precision = tp / np.maximum(preds.sum(1), 1)
    recall = tp / np.maximum(targets.sum(1), 1)

    def f_beta(beta):
        denom = beta**2 * precision + recall
        return np.where(denom > 0, (1 + beta**2) * precision * recall / np.where(denom > 0, denom, 1), 0)

    return {
        "mAP": float(np.nanmean(per_label_ap)),
        "example_F1": float(f_beta(1).mean()),
        "example_F2": float(f_beta(2).mean()),
        "example_precision": float(precision.mean()),
        "example_recall": float(recall.mean()),
        "macro_F1": float(f1_score(targets, preds, average="macro", zero_division=0)),
        "per_label_AP": [float(a) for a in per_label_ap],
    }


# ---------------------------------------------------------------- training

@torch.no_grad()
def predict(model, data, idx, device, batch_size=64):
    model.eval()
    object_probs, scene_probs = [], []
    for start in range(0, len(idx), batch_size):
        x = make_batch(data["images"], idx[start:start + batch_size], device, augment=False)
        object_logits, scene_logits = model(x)
        object_probs.append(torch.sigmoid(object_logits).cpu())
        if scene_logits is not None:
            scene_probs.append(scene_logits.softmax(dim=1).cpu())
    object_probs = torch.cat(object_probs).numpy()
    scene_probs = torch.cat(scene_probs).numpy() if scene_probs else None
    return object_probs, scene_probs


def train_model(model, data, config, use_scene_loss, device):
    """Train with BCE on object labels (+ cross-entropy on scene labels). Keeps the epoch with the best val mAP."""
    train_idx = split_indices(data, "train")
    extra_idx = split_indices(data, "extra")
    val_idx = split_indices(data, "val")
    num_extra = min(int(config["extra_ratio"] * len(train_idx)), len(extra_idx)) if use_scene_loss else 0

    head_params = [p for name, p in model.named_parameters() if not name.startswith("backbone.")]
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": config["lr"]},
        {"params": head_params, "lr": config["head_lr"]},
    ], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])

    objects = torch.from_numpy(data["objects"])
    has_objects = torch.from_numpy(data["has_objects"])
    scenes = torch.from_numpy(data["scenes"])

    history, best_map, best_state = [], -1.0, None
    for epoch in range(1, config["epochs"] + 1):
        start_time = time.time()
        model.train()
        # Every epoch uses all multi-label training images; scene-only images are resampled each epoch.
        epoch_idx = np.concatenate([train_idx, np.random.choice(extra_idx, num_extra, replace=False)])
        np.random.shuffle(epoch_idx)

        total_object_loss, total_scene_loss, num_batches = 0.0, 0.0, 0
        for start in range(0, len(epoch_idx), config["batch_size"]):
            idx = epoch_idx[start:start + config["batch_size"]]
            x = make_batch(data["images"], idx, device, augment=True)
            object_logits, scene_logits = model(x)

            mask = has_objects[idx].to(device)
            object_loss = torch.tensor(0.0, device=device)
            if mask.any():
                object_loss = F.binary_cross_entropy_with_logits(object_logits[mask], objects[idx].to(device)[mask])
            scene_loss = torch.tensor(0.0, device=device)
            if use_scene_loss:
                scene_loss = F.cross_entropy(scene_logits, scenes[idx].to(device))
            loss = object_loss + config["scene_loss_weight"] * scene_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_object_loss += object_loss.item()
            total_scene_loss += scene_loss.item()
            num_batches += 1
        scheduler.step()

        val_probs, _ = predict(model, data, val_idx, device)
        val_map = compute_metrics(val_probs, data["objects"][val_idx])["mAP"]
        history.append({
            "epoch": epoch,
            "object_loss": total_object_loss / num_batches,
            "scene_loss": total_scene_loss / num_batches,
            "val_mAP": val_map,
            "seconds": time.time() - start_time,
        })
        print(f"epoch {epoch:2d} | object loss {history[-1]['object_loss']:.4f} | "
              f"scene loss {history[-1]['scene_loss']:.4f} | val mAP {val_map:.4f} | {history[-1]['seconds']:.0f} s")
        if val_map > best_map:
            best_map = val_map
            best_state = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})

    model.load_state_dict(best_state)
    return history


def run_experiment(model_name, seed, data, config, device):
    """Train one model with one seed, evaluate it on the test split and save everything."""
    RESULTS_DIR.mkdir(exist_ok=True)
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    run_name = f"{model_name}_seed{seed}"
    print(f"--- {run_name} ---")

    set_seed(seed)

    # Reduced-label runs are named like "B1_labels25": B1 trained with 25% of the object labels.
    model_name, _, label_percent = model_name.partition("_labels")
    train_data = reduce_labels(data, int(label_percent) / 100) if label_percent else data

    # Control runs. B0_long: B0 trained for twice as many epochs, so it sees as many images as B1.
    # B1_noextra: no scene-only images, so B1 sees exactly the images of B0.
    # B1_shuffled: the scene labels of all training images are randomly permuted, so the scene head
    # gets the same amount of data and supervision as B1 but no real scene information.
    if model_name == "B0_long":
        config = {**config, "epochs": 2 * config["epochs"]}
    if model_name == "B1_noextra":
        config = {**config, "extra_ratio": 0.0}
    if model_name == "B1_shuffled":
        used = np.concatenate([split_indices(train_data, "train"), split_indices(train_data, "extra")])
        scenes = train_data["scenes"].copy()
        scenes[used] = np.random.permutation(scenes[used])
        train_data = {**train_data, "scenes": scenes}
        print(f"scene labels unchanged after shuffling: {(scenes[used] == data['scenes'][used]).mean():.1%}")

    # The model is built from train_data, so the scene prior of B2 only uses the labeled training images.
    model = build_model(model_name, train_data, pretrained=config["pretrained"]).to(device)
    history = train_model(model, train_data, config, use_scene_loss=not model_name.startswith("B0"),
                          device=device)

    test_idx = split_indices(data, "test")
    object_probs, scene_probs = predict(model, data, test_idx, device)
    metrics = compute_metrics(object_probs, data["objects"][test_idx])
    if scene_probs is not None:
        metrics["scene_accuracy"] = float((scene_probs.argmax(1) == data["scenes"][test_idx]).mean())
    if model.prior_table is not None:
        metrics["prior_weight"] = model.prior_weight.detach().cpu().tolist()

    np.save(RESULTS_DIR / f"{run_name}_test_object_probs.npy", object_probs)
    if scene_probs is not None:
        np.save(RESULTS_DIR / f"{run_name}_test_scene_probs.npy", scene_probs)
    torch.save(model.state_dict(), CHECKPOINT_DIR / f"{run_name}.pt")
    with open(RESULTS_DIR / f"{run_name}.json", "w") as f:
        json.dump({"model": run_name.rsplit("_seed", 1)[0], "seed": seed, "config": config,
                   "test_metrics": metrics, "history": history}, f, indent=2)
    print(f"test mAP {metrics['mAP']:.4f} | example F1 {metrics['example_F1']:.4f}")
    return metrics


# ---------------------------------------------------------------- notebook helpers

def run_all(model_name, seeds, data, config, device):
    """Run every seed of one model. Runs whose result file already exists are skipped."""
    for seed in seeds:
        if (RESULTS_DIR / f"{model_name}_seed{seed}.json").exists():
            print(f"{model_name}_seed{seed} already done, skipped")
            continue
        run_experiment(model_name, seed, data, config, device)


def load_runs(model_name):
    return [json.load(open(path)) for path in sorted(RESULTS_DIR.glob(f"{model_name}_seed*.json"))]


def plot_history(model_name):
    """Training curves of all seeds of one model."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.2))
    for run in load_runs(model_name):
        history = pd.DataFrame(run["history"])
        label = f"seed {run['seed']}"
        axes[0].plot(history["epoch"], history["object_loss"], label=label)
        axes[1].plot(history["epoch"], history["scene_loss"], label=label)
        axes[2].plot(history["epoch"], history["val_mAP"], label=label)
    for ax, title in zip(axes, ["object loss (train)", "scene loss (train)", "mAP (validation)"]):
        ax.set_title(f"{model_name}: {title}")
        ax.set_xlabel("epoch")
        ax.legend()
    plt.tight_layout()
    plt.show()
