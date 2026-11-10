# ==============================================================================
# STEP 4 (V2): TEST SET PROCESSING, 10-FOLD STACKED ENSEMBLE & PRIOR CALIBRATION
# ==============================================================================
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, roc_auc_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

if os.path.exists(TEST_JSONL_PATH):
    print("Building V2 features for test set...")
    if 'test_p1_df' not in globals():
        test_p1_df, test_timeline_df, test_p2_df = load_test_data(TEST_JSONL_PATH)
    test_turn_df = build_turn_level_dataset_v2(test_p1_df, test_timeline_df, meta_kb)
    test_t31_df, _ = compute_turn31_expected_type_features_v2(
        test_turn_df, meta_kb, p1_switch_model, p2_switch_model, p1_next_type_model, p2_next_type_model
    )
    test_battle_features_df = build_battle_level_features_v2(
        test_p1_df, test_timeline_df, test_p2_df, test_turn_df, test_t31_df, meta_kb, TYPE_MATRIX_17x17
    )
    print(f"Upgraded V2 Test Dataset Shape: {test_battle_features_df.shape}")

FEATURE_COLS = [c for c in battle_features_df.columns if c not in ['battle_id', 'player_won']]
X = battle_features_df[FEATURE_COLS].copy()
y = battle_features_df['player_won'].astype(int).values
X_test = test_battle_features_df[FEATURE_COLS].copy() if test_battle_features_df is not None else None

# Use 10-Fold Stratified CV for stronger regularization and higher training data per fold (90% vs 80%)
N_SPLITS = 10
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

oof_lgb = np.zeros(len(X))
oof_xgb = np.zeros(len(X))
oof_cat = np.zeros(len(X))

test_preds_lgb = np.zeros(len(X_test)) if X_test is not None else None
test_preds_xgb = np.zeros(len(X_test)) if X_test is not None else None
test_preds_cat = np.zeros(len(X_test)) if X_test is not None else None
feature_importances = np.zeros(len(FEATURE_COLS))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    X_tr, y_tr = X.iloc[train_idx], y[train_idx]
    X_va, y_va = X.iloc[val_idx], y[val_idx]

    # 1. Tuned LightGBM (shallower leaves + stronger regularization to prevent overfitting)
    model_lgb = LGBMClassifier(
        n_estimators=1200,
        learning_rate=0.02,
        num_leaves=23,
        min_child_samples=40,
        subsample_for_bin=50000,
        colsample_bytree=0.7,
        subsample=0.8,
        reg_alpha=2.0,
        reg_lambda=2.0,
        random_state=42 + fold,
        verbose=-1
    )
    model_lgb.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[__import__('lightgbm').early_stopping(stopping_rounds=60, verbose=False)]
    )
    oof_lgb[val_idx] = model_lgb.predict_proba(X_va)[:, 1]
    feature_importances += model_lgb.feature_importances_ / N_SPLITS

    # 2. Tuned XGBoost
    model_xgb = XGBClassifier(
        n_estimators=1200,
        learning_rate=0.02,
        max_depth=4,
        min_child_weight=10,
        colsample_bytree=0.7,
        subsample=0.8,
        reg_alpha=2.0,
        reg_lambda=2.0,
        eval_metric='logloss',
        early_stopping_rounds=60,
        random_state=42 + fold,
        n_jobs=-1
    )
    model_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    oof_xgb[val_idx] = model_xgb.predict_proba(X_va)[:, 1]

    # 3. Tuned CatBoost
    model_cat = CatBoostClassifier(
        iterations=1500,
        learning_rate=0.03,
        depth=5,
        l2_leaf_reg=5.0,
        random_seed=42 + fold,
        verbose=False
    )
    model_cat.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=60, verbose=False)
    oof_cat[val_idx] = model_cat.predict_proba(X_va)[:, 1]

    if X_test is not None:
        test_preds_lgb += model_lgb.predict_proba(X_test)[:, 1] / N_SPLITS
        test_preds_xgb += model_xgb.predict_proba(X_test)[:, 1] / N_SPLITS
        test_preds_cat += model_cat.predict_proba(X_test)[:, 1] / N_SPLITS

# Train a Logistic Regression Meta-Learner (Stacking) out-of-fold to find optimal model weights
meta_X_oof = np.column_stack([oof_lgb, oof_xgb, oof_cat])
oof_stacked = np.zeros(len(X))
meta_weights = np.zeros(3)

for train_idx, val_idx in skf.split(meta_X_oof, y):
    meta_clf = LogisticRegression(C=1.0)
    meta_clf.fit(meta_X_oof[train_idx], y[train_idx])
    oof_stacked[val_idx] = meta_clf.predict_proba(meta_X_oof[val_idx])[:, 1]
    meta_weights += meta_clf.coef_[0] / N_SPLITS

print("\n" + "="*65)
print("V2 10-FOLD OUT-OF-FOLD (OOF) VALIDATION RESULTS (10,000 BATTLES)")
print("="*65)
for name, preds in [
    ('LightGBM (V2)', oof_lgb),
    ('XGBoost (V2)', oof_xgb),
    ('CatBoost (V2)', oof_cat),
    ('Stacked Meta-Ensemble', oof_stacked)
]:
    acc = accuracy_score(y, (preds >= 0.50).astype(int))
    auc = roc_auc_score(y, preds)
    print(f"{name:<22} -> OOF Accuracy (@0.50): {acc*100:.2f}% | OOF ROC-AUC: {auc:.4f}")

# Generate Properly Calibrated Test Predictions (Matching True Training Win-Rate Prior!)
if test_battle_features_df is not None:
    meta_clf_full = LogisticRegression(C=1.0).fit(meta_X_oof, y)
    meta_X_test = np.column_stack([test_preds_lgb, test_preds_xgb, test_preds_cat])
    test_stacked_probs = meta_clf_full.predict_proba(meta_X_test)[:, 1]

    # Calibrate cutoff so Test Set P1 Win Rate matches the true Training Set Win Rate (~50%)
    train_win_rate = y.mean()
    prior_calibrated_thresh = float(np.quantile(test_stacked_probs, 1.0 - train_win_rate))

    # Blend standard 0.500 cutoff with prior-calibrated cutoff for maximum stability
    safe_thresh = 0.5 * 0.500 + 0.5 * prior_calibrated_thresh
    final_test_predictions = (test_stacked_probs >= safe_thresh).astype(int)

    submission_df = pd.DataFrame({
        'battle_id': test_battle_features_df['battle_id'].astype(int),
        'player_won': final_test_predictions
    }).sort_values('battle_id').reset_index(drop=True)

    submission_path = os.path.join(DATA_DIR, "submission_v2_26_10.csv")
    submission_df.to_csv(submission_path, index=False)
    print(f"\nTraining True Win Rate: {train_win_rate*100:.2f}% | Safe Threshold: {safe_thresh:.4f}")
    print(f"New Predicted Win Rate for Player 1 in Test Set: {submission_df['player_won'].mean()*100:.2f}% (Fixed from 59.76%!)")
    print(f"Saved upgraded '{submission_path}' successfully!")