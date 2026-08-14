"""
KVCIS-3 Step 3: Train BOTH probes.

  Probe A (regression)  -- recreation of the ORIGINAL KVCIS probe, unchanged:
                           Ridge regression, activation -> cumulative importance.
                           Saved exactly like the parent repo's step3 so step4
                           can use it the same way (weights [H], bias [1]).

  Probe B (evictability) -- binary logistic probe, activation -> "will go quiet
                           after the grace window" (1) vs "stays relevant" (0).
                           Consulted at runtime ONLY for int8-tier tokens; the
                           fp16 tier is never evicted regardless of probe B.

Outputs (in --output-dir):
  regression/weights.npy   [hidden]   probe A (original KVCIS)
  regression/bias.npy      [1]
  regression/metrics.json             R^2, correlation
  regression/probe.joblib
  evict/weights.npy        [hidden]   probe B (evictability logit)
  evict/bias.npy           [1]
  evict/metrics.json                  accuracy, precision/recall/F1 for evict
  evict/probe.joblib

Example:
  python step3_train_probe.py --data-dir ../data --output-dir ../data/probe2
"""

import argparse
import json
from pathlib import Path

import numpy as np
import joblib
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (r2_score, mean_squared_error, accuracy_score,
                             balanced_accuracy_score, precision_recall_fscore_support,
                             confusion_matrix)


def train_probe_a(X, y, out_dir, alpha, test_size):
    """Original KVCIS regression probe, recreated (parent repo step3 semantics)."""
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size,
                                              random_state=42)
    print(f"[probe A] train {X_tr.shape[0]}  test {X_te.shape[0]}  dim {X_tr.shape[1]}")
    probe = Ridge(alpha=alpha)
    probe.fit(X_tr, y_tr)
    pred_tr, pred_te = probe.predict(X_tr), probe.predict(X_te)
    metrics = {
        "train_r2": float(r2_score(y_tr, pred_tr)),
        "test_r2": float(r2_score(y_te, pred_te)),
        "train_mse": float(mean_squared_error(y_tr, pred_tr)),
        "test_mse": float(mean_squared_error(y_te, pred_te)),
        "train_corr": float(np.corrcoef(y_tr, pred_tr)[0, 1]),
        "test_corr": float(np.corrcoef(y_te, pred_te)[0, 1]),
        "alpha": alpha,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "weights.npy", probe.coef_.astype(np.float32))
    np.save(out_dir / "bias.npy", np.array([probe.intercept_], dtype=np.float32))
    joblib.dump(probe, out_dir / "probe.joblib")
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # verify manual prediction path used at runtime
    manual = X_te @ np.load(out_dir / "weights.npy") + np.load(out_dir / "bias.npy")[0]
    diff = float(np.abs(manual - pred_te).max())
    print(f"[probe A] test R^2 {metrics['test_r2']:.4f}  corr {metrics['test_corr']:.4f}"
          f"  (manual-vs-sklearn max diff {diff:.2e})")
    return metrics


def train_probe_b(X, y, out_dir, C, test_size):
    """Binary evictability probe (trained on all tokens; the fp16-never-evict
    rule is enforced at runtime as a hard exemption, not learned)."""
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size,
                                              random_state=42, stratify=y)
    print(f"[probe B] train {X_tr.shape[0]}  test {X_te.shape[0]}  "
          f"evictable {int(y.sum())}/{len(y)} ({100 * y.mean():.1f}%)")
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=2000,
                             solver="lbfgs")
    clf.fit(X_tr, y_tr)
    assert list(clf.classes_) == [0, 1]
    y_pred = clf.predict(X_te)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_te, y_pred, labels=[1], zero_division=0)
    cm = confusion_matrix(y_te, y_pred, labels=[0, 1])
    metrics = {
        "test_accuracy": float(accuracy_score(y_te, y_pred)),
        "test_balanced_accuracy": float(balanced_accuracy_score(y_te, y_pred)),
        "evict_precision": float(prec[0]),
        "evict_recall": float(rec[0]),
        "evict_f1": float(f1[0]),
        "confusion_matrix_rows_true_cols_pred": cm.tolist(),
        "C": C,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "weights.npy", clf.coef_[0].astype(np.float32))
    np.save(out_dir / "bias.npy", clf.intercept_.astype(np.float32))
    joblib.dump(clf, out_dir / "probe.joblib")
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[probe B] acc {metrics['test_accuracy']:.4f} "
          f"(balanced {metrics['test_balanced_accuracy']:.4f})  "
          f"evict P/R/F1 {metrics['evict_precision']:.3f}/"
          f"{metrics['evict_recall']:.3f}/{metrics['evict_f1']:.3f}")
    print("  NOTE: evict PRECISION is the safety metric -- a false positive "
          "evicts a token that was still needed.")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="KVCIS-3 Step 3: train probes A and B")
    parser.add_argument("--data-dir", type=str, default="../data")
    parser.add_argument("--output-dir", type=str, default="../data/probe2")
    parser.add_argument("--alpha", type=float, default=1.0, help="Probe A Ridge strength")
    parser.add_argument("--C", type=float, default=1.0, help="Probe B inverse L2 strength")
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    X = np.load(data_dir / "activations.npy")
    importance = np.load(data_dir / "importance.npy")
    evictable = np.load(data_dir / "evictable.npy")
    if not (0 < evictable.mean() < 1):
        raise SystemExit("evictable.npy has a single class -- re-run step2 with "
                         "adjusted --evict-max-ratio / --evict-window")

    out = Path(args.output_dir)
    print("=== Probe A: original KVCIS importance regression ===")
    ma = train_probe_a(X, importance, out / "regression", args.alpha, args.test_size)
    print("\n=== Probe B: deferred-evictability classifier ===")
    mb = train_probe_b(X, evictable, out / "evict", args.C, args.test_size)

    with open(out / "summary.json", "w") as f:
        json.dump({"probe_a": ma, "probe_b": mb}, f, indent=2)
    print(f"\nProbes saved under {out}  (regression/ = A, evict/ = B)")
    print("\nStep 3 complete - both probes trained")


if __name__ == "__main__":
    main()
