# ============================================================
#  train_all.py
#  Run: python train_all.py
#  Output: models + graphs/interval_name/ folder mein PNG files
# ============================================================

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import joblib
import warnings
import os
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')   # VS Code mein popup nahi aayega — file save hogi
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.preprocessing   import RobustScaler
from sklearn.ensemble        import RandomForestClassifier
from sklearn.metrics         import (accuracy_score, roc_auc_score,
                                     classification_report,
                                     confusion_matrix, roc_curve,
                                     precision_recall_curve)

# ════════════════════════════════════════════════════════
#  CONFIG
# ════════════════════════════════════════════════════════

CSV_FILE   = "BTCUSDT_ALL_INTERVALS_4yr.csv"
INTERVALS  = ["15m", "30m", "2h", "4h", "6h", "8h", "12h"]
TARGET     = "target_1"
GRAPH_DIR  = "graphs"   # graphs yahan save honge

DROP_COLS  = [
    "open_time", "close_time", "coin", "interval",
    "target_1", "target_3", "target_7", "target_14",
    "target_pct_1", "target_pct_3", "target_pct_7", "target_pct_14",
    "market_state",
]

os.makedirs(GRAPH_DIR, exist_ok=True)


# ════════════════════════════════════════════════════════
#  GRAPH FUNCTION — sab plots ek hi image mein
# ════════════════════════════════════════════════════════

def save_graphs(interval, y_test, y_prob, y_pred, feat_imp_df,
                lgb_model, X_val_sc, y_val):

    save_dir = os.path.join(GRAPH_DIR, interval)
    os.makedirs(save_dir, exist_ok=True)

    # ── 1. Confusion Matrix ───────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(y_test, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    im = ax.imshow(cm_pct, cmap='Blues', vmin=0, vmax=100)
    plt.colorbar(im, ax=ax, label='Percentage %')

    labels = ['BEARISH (0)', 'BULLISH (1)']
    ax.set_xticks([0, 1]); ax.set_xticklabels(labels)
    ax.set_yticks([0, 1]); ax.set_yticklabels(labels)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title(f'Confusion Matrix — {interval}', fontsize=14, fontweight='bold')

    for i in range(2):
        for j in range(2):
            ax.text(j, i,
                    f'{cm[i,j]:,}\n({cm_pct[i,j]:.1f}%)',
                    ha='center', va='center', fontsize=12,
                    color='white' if cm_pct[i,j] > 50 else 'black')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '1_confusion_matrix.png'), dpi=120)
    plt.close()
    print(f"    Saved: confusion matrix")

    # ── 2. ROC Curve ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 5))
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_val     = roc_auc_score(y_test, y_prob)

    ax.plot(fpr, tpr, color='#2196F3', lw=2,
            label=f'ROC AUC = {auc_val:.4f}')
    ax.plot([0,1],[0,1], 'k--', lw=1, alpha=0.5, label='Random (0.5)')
    ax.fill_between(fpr, tpr, alpha=0.1, color='#2196F3')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'ROC Curve — {interval}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '2_roc_curve.png'), dpi=120)
    plt.close()
    print(f"    Saved: ROC curve (AUC={auc_val:.4f})")

    # ── 3. Precision-Recall Curve ─────────────────────────
    fig, ax = plt.subplots(figsize=(6, 5))
    prec, rec, _ = precision_recall_curve(y_test, y_prob)
    baseline     = y_test.mean()

    ax.plot(rec, prec, color='#4CAF50', lw=2, label='PR Curve')
    ax.axhline(baseline, color='red', lw=1, linestyle='--',
               label=f'Baseline = {baseline:.2f}')
    ax.fill_between(rec, prec, alpha=0.1, color='#4CAF50')
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(f'Precision-Recall Curve — {interval}',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '3_precision_recall.png'), dpi=120)
    plt.close()
    print(f"    Saved: Precision-Recall curve")

    # ── 4. Feature Importance (Top 20) ────────────────────
    fig, ax = plt.subplots(figsize=(8, 7))
    top20 = feat_imp_df.head(20)

    colors = ['#2196F3' if i < 5 else '#90CAF9' for i in range(len(top20))]
    bars   = ax.barh(range(len(top20)), top20['importance'], color=colors)
    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels(top20['feature'], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Importance Score', fontsize=12)
    ax.set_title(f'Top 20 Feature Importance — {interval}',
                 fontsize=14, fontweight='bold')
    ax.grid(True, axis='x', alpha=0.3)

    # Value labels
    for bar, val in zip(bars, top20['importance']):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{val:.0f}', va='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '4_feature_importance.png'), dpi=120)
    plt.close()
    print(f"    Saved: Feature importance")

    # ── 5. Confidence Distribution ────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: histogram
    bull_probs = y_prob[y_test == 1]
    bear_probs = y_prob[y_test == 0]

    axes[0].hist(bear_probs, bins=40, alpha=0.6, color='#F44336',
                 label='Actual BEARISH', density=True)
    axes[0].hist(bull_probs, bins=40, alpha=0.6, color='#4CAF50',
                 label='Actual BULLISH', density=True)
    axes[0].axvline(0.5, color='black', lw=1.5, linestyle='--',
                    label='Decision boundary (0.5)')
    axes[0].set_xlabel('Predicted Probability (BULLISH)', fontsize=11)
    axes[0].set_ylabel('Density', fontsize=11)
    axes[0].set_title('Confidence Distribution', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # Right: calibration — actual vs predicted
    from sklearn.calibration import calibration_curve
    prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)
    axes[1].plot(prob_pred, prob_true, 'o-', color='#2196F3',
                 lw=2, label='Model calibration')
    axes[1].plot([0,1],[0,1], 'k--', lw=1, label='Perfect calibration')
    axes[1].set_xlabel('Mean Predicted Probability', fontsize=11)
    axes[1].set_ylabel('Fraction of Positives', fontsize=11)
    axes[1].set_title('Calibration Curve', fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(f'Confidence Analysis — {interval}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '5_confidence_analysis.png'), dpi=120)
    plt.close()
    print(f"    Saved: Confidence distribution + Calibration")

    # ── 6. Prediction Over Time ────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

    x = np.arange(len(y_test))

    # Panel 1: Actual vs Predicted
    axes[0].fill_between(x, y_test.values, alpha=0.4, color='#2196F3',
                         label='Actual')
    axes[0].plot(x, y_prob, color='#FF9800', lw=0.8, alpha=0.8,
                 label='Predicted prob')
    axes[0].axhline(0.5, color='red', lw=1, linestyle='--')
    axes[0].set_ylabel('Signal', fontsize=10)
    axes[0].set_title(f'Prediction vs Actual Over Time — {interval}',
                      fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=9, loc='upper right')
    axes[0].set_ylim(-0.1, 1.1)
    axes[0].grid(True, alpha=0.3)

    # Panel 2: Correct / Wrong predictions
    correct = (y_pred == y_test.values).astype(int)
    rolling_acc = pd.Series(correct).rolling(50).mean()
    axes[1].plot(x, rolling_acc, color='#4CAF50', lw=1.2)
    axes[1].axhline(0.5, color='red', lw=1, linestyle='--', label='50% baseline')
    axes[1].fill_between(x, rolling_acc, 0.5,
                         where=(rolling_acc >= 0.5),
                         alpha=0.3, color='#4CAF50', label='Above baseline')
    axes[1].fill_between(x, rolling_acc, 0.5,
                         where=(rolling_acc < 0.5),
                         alpha=0.3, color='#F44336', label='Below baseline')
    axes[1].set_ylabel('Rolling Acc (50)', fontsize=10)
    axes[1].legend(fontsize=9, loc='upper right')
    axes[1].set_ylim(0.2, 0.8)
    axes[1].grid(True, alpha=0.3)

    # Panel 3: Confidence over time
    axes[2].plot(x, y_prob, color='#9C27B0', lw=0.8, alpha=0.7)
    axes[2].axhline(0.65, color='green',  lw=1, linestyle='--',
                    label='Strong bullish (0.65)')
    axes[2].axhline(0.35, color='red',    lw=1, linestyle='--',
                    label='Strong bearish (0.35)')
    axes[2].axhline(0.5,  color='black',  lw=0.5, linestyle=':')
    axes[2].set_ylabel('Confidence', fontsize=10)
    axes[2].set_xlabel('Test samples (time order)', fontsize=10)
    axes[2].legend(fontsize=9, loc='upper right')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '6_prediction_over_time.png'), dpi=120)
    plt.close()
    print(f"    Saved: Prediction over time")

    # ── 7. Summary Card ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.axis('off')

    cm      = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    acc     = accuracy_score(y_test, y_pred)
    auc     = roc_auc_score(y_test, y_prob)
    prec    = tp / (tp + fp) if (tp+fp) > 0 else 0
    rec     = tp / (tp + fn) if (tp+fn) > 0 else 0
    f1      = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0

    data = [
        ['Metric',              'Value',        'Status'],
        ['Accuracy',            f'{acc:.4f}',   '✅' if acc > 0.52 else '❌'],
        ['ROC-AUC',             f'{auc:.4f}',   '✅' if auc > 0.53 else '❌'],
        ['Precision (Bullish)', f'{prec:.4f}',  '✅' if prec > 0.52 else '❌'],
        ['Recall (Bullish)',    f'{rec:.4f}',   '✅' if rec  > 0.50 else '❌'],
        ['F1 Score',            f'{f1:.4f}',    '✅' if f1   > 0.50 else '❌'],
        ['True Positives',      f'{tp:,}',      ''],
        ['True Negatives',      f'{tn:,}',      ''],
        ['False Positives',     f'{fp:,}',      ''],
        ['False Negatives',     f'{fn:,}',      ''],
    ]

    table = ax.table(cellText=data[1:], colLabels=data[0],
                     cellLoc='center', loc='center',
                     colWidths=[0.45, 0.3, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.0)

    # Header color
    for j in range(3):
        table[0, j].set_facecolor('#1565C0')
        table[0, j].set_text_props(color='white', fontweight='bold')

    # Row colors
    for i in range(1, len(data)):
        for j in range(3):
            if i % 2 == 0:
                table[i, j].set_facecolor('#E3F2FD')

    ax.set_title(f'Model Performance Summary — {interval}',
                 fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '7_summary_card.png'), dpi=120)
    plt.close()
    print(f"    Saved: Summary card")


# ════════════════════════════════════════════════════════
#  HELPER — evaluate
# ════════════════════════════════════════════════════════

def evaluate(model, X_val_sc, y_val, X_test_sc, y_test, name):
    y_prob_val  = model.predict_proba(X_val_sc)[:, 1]
    y_prob_test = model.predict_proba(X_test_sc)[:, 1]
    y_pred_test = model.predict(X_test_sc)

    val_auc  = roc_auc_score(y_val,  y_prob_val)
    test_auc = roc_auc_score(y_test, y_prob_test)
    test_acc = accuracy_score(y_test, y_pred_test)

    print(f"    {name:<20} Val AUC: {val_auc:.4f} | "
          f"Test AUC: {test_auc:.4f} | Acc: {test_acc:.4f}")
    return val_auc, test_auc, test_acc, y_prob_test, y_pred_test


# ════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════

def main():
    print("Loading CSV...")
    df_all = pd.read_csv(CSV_FILE)
    df_all["open_time"] = pd.to_datetime(df_all["open_time"], utc=True)
    print(f"Total rows: {len(df_all):,}\n")

    summary = {}

    for interval in INTERVALS:
        print(f"\n{'═'*55}")
        print(f"  INTERVAL: {interval}")
        print(f"{'═'*55}")

        # ── 1. Filter ─────────────────────────────────────
        df = df_all[df_all["interval"] == interval].copy()
        df = df.sort_values("open_time").reset_index(drop=True)
        print(f"  Rows: {len(df):,}")

        # ── 2. Features & Target ──────────────────────────
        FEATURES = [c for c in df.columns if c not in DROP_COLS]
        X = df[FEATURES]
        y = df[TARGET]

        bull_pct = y.mean() * 100
        print(f"  Bullish: {bull_pct:.1f}% | Bearish: {100-bull_pct:.1f}%")

        # ── 3. Split ──────────────────────────────────────
        n         = len(df)
        train_end = int(n * 0.70)
        val_end   = int(n * 0.85)

        X_train = X.iloc[:train_end]
        X_val   = X.iloc[train_end:val_end]
        X_test  = X.iloc[val_end:]
        y_train = y.iloc[:train_end]
        y_val   = y.iloc[train_end:val_end]
        y_test  = y.iloc[val_end:]

        print(f"  Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

        # ── 4. Scale ──────────────────────────────────────
        scaler     = RobustScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_val_sc   = scaler.transform(X_val)
        X_test_sc  = scaler.transform(X_test)

        print(f"\n  Training 3 models...")

        # ── 5. LightGBM ───────────────────────────────────
        lgb_model = lgb.LGBMClassifier(
            n_estimators=1000, learning_rate=0.05,
            num_leaves=63, min_child_samples=50,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=0.1,
            class_weight="balanced",
            random_state=42, n_jobs=-1, verbose=-1,
        )
        lgb_model.fit(
            X_train_sc, y_train,
            eval_set=[(X_val_sc, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False),
                       lgb.log_evaluation(-1)],
        )
        lgb_val_auc, lgb_test_auc, lgb_acc, lgb_prob, lgb_pred = evaluate(
            lgb_model, X_val_sc, y_val, X_test_sc, y_test, "LightGBM")

        # ── 6. XGBoost ────────────────────────────────────
        scale_pos = (y_train == 0).sum() / (y_train == 1).sum()
        xgb_model = xgb.XGBClassifier(
            n_estimators=1000, learning_rate=0.05,
            max_depth=6, min_child_weight=5,
            subsample=0.8, colsample_bytree=0.8,
            gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
            scale_pos_weight=scale_pos,
            eval_metric="auc", early_stopping_rounds=50,
            random_state=42, n_jobs=-1, verbosity=0,
        )
        xgb_model.fit(
            X_train_sc, y_train,
            eval_set=[(X_val_sc, y_val)],
            verbose=False,
        )
        xgb_val_auc, xgb_test_auc, xgb_acc, xgb_prob, xgb_pred = evaluate(
            xgb_model, X_val_sc, y_val, X_test_sc, y_test, "XGBoost")

        # ── 7. Random Forest ──────────────────────────────
        max_rf = 30_000
        if len(X_train_sc) > max_rf:
            idx  = np.sort(np.random.choice(len(X_train_sc), max_rf, replace=False))
            X_rf = X_train_sc[idx]
            y_rf = y_train.iloc[idx]
        else:
            X_rf = X_train_sc
            y_rf = y_train

        rf_model = RandomForestClassifier(
            n_estimators=300, max_depth=10,
            min_samples_leaf=20, max_features="sqrt",
            class_weight="balanced",
            random_state=42, n_jobs=-1,
        )
        rf_model.fit(X_rf, y_rf)
        rf_val_auc, rf_test_auc, rf_acc, rf_prob, rf_pred = evaluate(
            rf_model, X_val_sc, y_val, X_test_sc, y_test, "Random Forest")

        # ── 8. Best model select ──────────────────────────
        scores = {"lgb": lgb_val_auc,
                  "xgb": xgb_val_auc,
                  "rf" : rf_val_auc}
        best_name  = max(scores, key=scores.get)
        models_map = {"lgb": (lgb_model, lgb_prob, lgb_pred, lgb_val_auc, lgb_test_auc, lgb_acc),
                      "xgb": (xgb_model, xgb_prob, xgb_pred, xgb_val_auc, xgb_test_auc, xgb_acc),
                      "rf" : (rf_model,  rf_prob,  rf_pred,  rf_val_auc,  rf_test_auc,  rf_acc)}

        best_model, best_prob, best_pred, bv, bt, ba = models_map[best_name]
        print(f"\n  ✅ Best: {best_name.upper()} (Val AUC: {bv:.4f})")

        # ── 9. Feature importance ─────────────────────────
        feat_imp = pd.DataFrame({
            "feature"   : FEATURES,
            "importance": lgb_model.feature_importances_,
        }).sort_values("importance", ascending=False)

        # ── 10. Graphs save karo ──────────────────────────
        print(f"\n  Saving graphs for [{interval}]...")
        save_graphs(
            interval, y_test, best_prob, best_pred,
            feat_imp, lgb_model, X_val_sc, y_val
        )

        # ── 11. Models save ───────────────────────────────
        joblib.dump(best_model, f"model_{interval}.pkl")
        joblib.dump(scaler,     f"scaler_{interval}.pkl")
        joblib.dump(FEATURES,   f"features_{interval}.pkl")
        joblib.dump(best_name,  f"modeltype_{interval}.pkl")

        summary[interval] = {
            "best_model": best_name.upper(),
            "val_auc"   : bv,
            "test_auc"  : bt,
            "test_acc"  : ba,
            "rows"      : len(df),
        }

    # ── Final Summary ─────────────────────────────────────
    print(f"\n\n{'═'*62}")
    print(f"  FINAL SUMMARY")
    print(f"{'═'*62}")
    print(f"  {'TF':<6} {'Model':<8} {'Rows':>8} "
          f"{'Val AUC':>9} {'Test AUC':>9} {'Test ACC':>9}")
    print(f"  {'-'*60}")
    for iv, r in summary.items():
        status = "✅" if r["test_auc"] > 0.53 else "❌"
        print(f"  {iv:<6} {r['best_model']:<8} {r['rows']:>8,} "
              f"{r['val_auc']:>9.4f} {r['test_auc']:>9.4f} "
              f"{r['test_acc']:>9.4f}  {status}")

    print(f"\n  Graphs saved in: ./{GRAPH_DIR}/")
    print(f"  Ab predict_multi.py chalao.")


if __name__ == "__main__":
    main()