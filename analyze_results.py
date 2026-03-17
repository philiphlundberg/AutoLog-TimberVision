"""
YOLO Training Results Analyzer
================================
How to interpret the results.csv:
- Each row = one epoch of training
- metrics/precision(B), recall(B): computed on the validation set at that epoch
- metrics/mAP50(B): mean Average Precision at IoU threshold 0.50 (lenient)
- metrics/mAP50-95(B): mean Average Precision averaged across IoU 0.50-0.95 (strict, primary metric)
- val/box_loss, val/cls_loss: validation losses (lower = better, rising = overfitting)
- train/* losses: training losses (should decrease steadily)
- lr/pg0: learning rate (decreases over time with cosine schedule)

The (B) suffix means Box/Detection metrics. For segmentation CSVs you also get (M) = Mask metrics.

How to find your model's reported performance:
- Use the MAXIMUM mAP50-95, not the final epoch or the average
- best.pt is saved at the epoch with highest fitness = 0.1*mAP50 + 0.9*mAP50-95
- Averaging across epochs is meaningless (mixes good and bad checkpoints)

F1 score = 2 * (Precision * Recall) / (Precision + Recall)
- Balances precision and recall into a single number
- Useful when you care equally about false positives and false negatives
- For robotics/grasping: precision matters more (false positive = wrong grasp)
"""

import csv
import sys

def analyze(filepath):
    epochs, precision, recall, map50, map5095 = [], [], [], [], []
    val_box, val_cls = [], []
    train_box = []

    with open(filepath) as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row['epoch']))
            precision.append(float(row['metrics/precision(B)']))
            recall.append(float(row['metrics/recall(B)']))
            map50.append(float(row['metrics/mAP50(B)']))
            map5095.append(float(row['metrics/mAP50-95(B)']))
            val_box.append(float(row['val/box_loss']))
            val_cls.append(float(row['val/cls_loss']))
            train_box.append(float(row['train/box_loss']))

    f1 = [2*p*r/(p+r) for p, r in zip(precision, recall)]

    # --- Best epoch by primary metric (mAP50-95) ---
    # This is what YOLO uses to save best.pt
    best_map_idx = map5095.index(max(map5095))
    print("=" * 55)
    print("BEST EPOCH (by mAP50-95) — matches best.pt")
    print("=" * 55)
    print(f"  Epoch:      {epochs[best_map_idx]}")
    print(f"  mAP50-95:   {map5095[best_map_idx]:.4f}  ← primary metric, use this for comparisons")
    print(f"  mAP50:      {map50[best_map_idx]:.4f}  ← lenient version, always higher")
    print(f"  Precision:  {precision[best_map_idx]:.4f}  ← of predicted detections, how many are correct")
    print(f"  Recall:     {recall[best_map_idx]:.4f}  ← of all true objects, how many were found")
    print(f"  F1:         {f1[best_map_idx]:.4f}  ← harmonic mean of precision and recall")

    # --- Best epoch by F1 ---
    # Useful if you want balanced precision/recall operating point
    best_f1_idx = f1.index(max(f1))
    print()
    print("=" * 55)
    print("BEST EPOCH (by F1) — useful for balanced P/R tradeoff")
    print("=" * 55)
    print(f"  Epoch:      {epochs[best_f1_idx]}")
    print(f"  F1:         {f1[best_f1_idx]:.4f}")
    print(f"  Precision:  {precision[best_f1_idx]:.4f}")
    print(f"  Recall:     {recall[best_f1_idx]:.4f}")
    print(f"  mAP50-95:   {map5095[best_f1_idx]:.4f}")

    # --- Overfitting check ---
    # If val loss starts rising while train loss keeps falling, the model is overfitting
    min_val_box_epoch = epochs[val_box.index(min(val_box))]
    final_val_box = val_box[-1]
    min_val_box = min(val_box)
    print()
    print("=" * 55)
    print("OVERFITTING CHECK")
    print("=" * 55)
    print(f"  Min val/box_loss:   {min_val_box:.4f} at epoch {min_val_box_epoch}")
    print(f"  Final val/box_loss: {final_val_box:.4f}")
    drift = (final_val_box - min_val_box) / min_val_box * 100
    if drift > 5:
        print(f"  ⚠️  Val loss drifted +{drift:.1f}% from minimum — possible overfitting")
    else:
        print(f"  ✅ Val loss stable (only +{drift:.1f}% from minimum) — no significant overfitting")

    # --- Training progression ---
    # Shows how quickly the model learned and where it plateaued
    print()
    print("=" * 55)
    print("TRAINING PROGRESSION (mAP50-95 at key epochs)")
    print("  Interpretation: fast early gain, slow late gain is normal")
    print("=" * 55)
    checkpoints = [0, 49, 99, 149, 199, 249, 299, 349, 399, 449, len(epochs)-1]
    checkpoints = [i for i in checkpoints if i < len(epochs)]
    print(f"  {'Epoch':>6} | {'mAP50-95':>9} | {'F1':>6} | {'Precision':>9} | {'Recall':>7}")
    print(f"  {'-'*6}-+-{'-'*9}-+-{'-'*6}-+-{'-'*9}-+-{'-'*7}")
    for i in checkpoints:
        marker = " ← best" if i == best_map_idx else ""
        print(f"  {epochs[i]:>6} | {map5095[i]:>9.4f} | {f1[i]:>6.4f} | {precision[i]:>9.4f} | {recall[i]:>7.4f}{marker}")

    # --- Precision vs Recall tradeoff note ---
    print()
    print("=" * 55)
    print("PRECISION vs RECALL INTERPRETATION")
    print("=" * 55)
    p = precision[best_map_idx]
    r = recall[best_map_idx]
    if p > r + 0.05:
        print(f"  Model is PRECISION-BIASED (P={p:.3f} > R={r:.3f})")
        print("  → Conservative: misses some objects but rarely predicts wrong ones")
        print("  → Good for robotics where false positives (wrong grasps) are costly")
    elif r > p + 0.05:
        print(f"  Model is RECALL-BIASED (R={r:.3f} > P={p:.3f})")
        print("  → Aggressive: finds most objects but some predictions are wrong")
    else:
        print(f"  Model is BALANCED (P={p:.3f}, R={r:.3f})")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "results.csv"
    analyze(path)
