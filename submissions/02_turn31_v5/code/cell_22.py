# ==============================================================================
# STEP 5 (INSTANT V5 OPTIMIZER): CATBOOST-DOMINANT BLEND & TOP V5 FEATURES
# ==============================================================================
# 1. Inspect Top 25 Features in CatBoost V5
v5_imp = pd.DataFrame({
    'Feature': V5_FEATURE_COLS,
    'Importance': model_cat.get_feature_importance()
}).sort_values('Importance', ascending=False).reset_index(drop=True)

print("=== TOP 25 MOST IMPORTANT FEATURES IN V5 (CATBOOST 0.7597 AUC) ===")
for idx, r in v5_imp.head(25).iterrows():
    print(f"{idx+1:2d}. {r['Feature']:<32} : {r['Importance']:.2f}")

# 2. Compare Pure CatBoost vs CatBoost-Dominant Blend (80% Cat + 12% XGB + 8% LGB)
oof_cat_dom = 0.80 * oof_cat_v5 + 0.12 * oof_xgb_v5 + 0.08 * oof_lgb_v5

for name, preds in [
    ('Old 45/30/25 Blend', oof_v5_blend),
    ('Pure CatBoost (V5)', oof_cat_v5),
    ('CatBoost-Dominant (80/12/8)', oof_cat_dom)
]:
    acc_50 = accuracy_score(y_v5, (preds >= 0.500).astype(int))
    # Also check median-calibrated accuracy (exact 50.0% win rate prior)
    med_th = np.median(preds)
    acc_med = accuracy_score(y_v5, (preds >= med_th).astype(int))
    auc = roc_auc_score(y_v5, preds)
    print(f"\n{name:<28} -> ROC-AUC: {auc:.4f} | Acc (@0.500): {acc_50*100:.2f}% | Acc (@50% Prior): {acc_med*100:.2f}%")

# 3. Save the CatBoost-Dominant (80/12/8) Submission (0.7598+ AUC!)
if X_test_v5 is not None:
    test_cat_dom_probs = 0.80 * test_cat_v5 + 0.12 * test_xgb_v5 + 0.08 * test_lgb_v5
    sub_cat_dom = pd.DataFrame({
        'battle_id': test_battle_v5_df['battle_id'].astype(int),
        'player_won': (test_cat_dom_probs >= 0.500).astype(int)
    }).sort_values('battle_id').reset_index(drop=True)

    cat_dom_path = os.path.join(DATA_DIR, "submission_v5_catboost_dominant.csv")
    sub_cat_dom.to_csv(cat_dom_path, index=False)
    print(f"\nSaved '{cat_dom_path}'! (Predicted P1 Win Rate: {sub_cat_dom['player_won'].mean()*100:.2f}%)")