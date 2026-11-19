"""Classifier routine extracted from the original experiment code."""
import argparse
from collections import Counter
from pathlib import Path
import os
import json
import time
from PIL import Image
ROOT = Path(os.environ.get("AQUARIUM_ROOT", str(Path(__file__).resolve().parents[1] / "data"))).resolve()
OUT = ROOT / "ablation"
def read_classes():
    return [s.strip() for s in (ROOT / "classes.txt").read_text(encoding="utf-8").splitlines() if s.strip()]

def train_classifier(args: argparse.Namespace) -> None:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    from torchvision import models, transforms

    classes = read_classes()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_tf = transforms.Compose([transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(), transforms.ColorJitter(0.2, 0.2, 0.2), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    eval_tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def samples(split):
        out = []
        for cid, name in enumerate(classes):
            d = OUT / "crops" / split / name
            if d.exists():
                out.extend((p, cid) for p in sorted(d.iterdir()) if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        return out

    class DS(Dataset):
        def __init__(self, split, tf):
            self.items = samples(split)
            self.tf = tf

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            p, cid = self.items[i]
            with Image.open(p).convert("RGB") as im:
                return self.tf(im), cid

    train_ds = DS("train", train_tf)
    val_ds = DS("val", eval_tf)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(classes))
    model.to(device)
    counts = Counter(cid for _, cid in train_ds.items)
    weights = torch.tensor([len(train_ds) / (len(classes) * counts[i]) for i in range(len(classes))], dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30)
    run_dir = OUT / "runs" / "classifier"
    run_dir.mkdir(parents=True, exist_ok=True)
    best = -1.0
    t0 = time.time()
    for epoch in range(1, 31):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            opt.step()
        sched.step()
        model.eval()
        targets, preds = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                targets.extend(labels.tolist())
                preds.extend(model(images).argmax(1).tolist())
        f1 = 0.0
        for c in range(len(classes)):
            tp = sum(t == c and p == c for t, p in zip(targets, preds))
            fp = sum(t != c and p == c for t, p in zip(targets, preds))
            fn = sum(t == c and p != c for t, p in zip(targets, preds))
            prec = tp / (tp + fp) if tp + fp else 0.0
            rec = tp / (tp + fn) if tp + fn else 0.0
            f1 += (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
        f1 /= len(classes)
        print(f"epoch {epoch} val_macro_f1 {f1:.4f}", flush=True)
        if f1 > best:
            best = f1
            torch.save({"state_dict": model.state_dict(), "classes": classes}, run_dir / "best.pt")
    (run_dir / "summary.json").write_text(json.dumps({"seconds": round(time.time() - t0), "parameters": sum(p.numel() for p in model.parameters()), "best_val_macro_f1": best}), encoding="utf-8")
    print(f"[done] classifier {time.time()-t0:.0f}s best_f1 {best:.4f}", flush=True)


if __name__ == "__main__":
    train_classifier(argparse.Namespace())
