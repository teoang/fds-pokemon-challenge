# ==============================================================================
# STEP 4 (V5 - SELF-CONTAINED): OOF META PRIORS, 10-FOLD GBDT & SUBMISSION
# ==============================================================================
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

def load_test_data(test_jsonl_path):
    p1_rows, timeline_rows, p2_rows = [], [], []
    with open(test_jsonl_path, 'r') as f:
        for line in f:
            b = json.loads(line)
            b_id = b['battle_id']
            p2 = b['p2_lead_details'].copy()
            p2['battle_id'] = b_id
            p2_rows.append(p2)
            for pkmn in b['p1_team_details']:
                p1 = pkmn.copy()
                p1['battle_id'] = b_id
                p1_rows.append(p1)
            for turn_data in b['battle_timeline']:
                row = {'battle_id': b_id, 'turn': turn_data['turn']}
                for side in ['p1', 'p2']:
                    state = turn_data.get(f'{side}_pokemon_state', {}) or {}
                    row[f'{side}_pokemon_state.name'] = state.get('name')
                    row[f'{side}_pokemon_state.hp_pct'] = state.get('hp_pct')
                    row[f'{side}_pokemon_state.status'] = state.get('status')
                    row[f'{side}_pokemon_state.effects'] = str(state.get('effects', 'noeffect'))
                    boosts = state.get('boosts', {}) or {}
                    for stat in ['atk', 'def', 'spa', 'spd', 'spe']:
                        row[f'{side}_pokemon_state.boosts.{stat}'] = boosts.get(stat, 0)
                    move = turn_data.get(f'{side}_move_details', {}) or {}
                    row[f'{side}_move_details.name'] = move.get('name')
                    row[f'{side}_move_details.type'] = move.get('type')
                    row[f'{side}_move_details.category'] = move.get('category')
                    row[f'{side}_move_details.base_power'] = move.get('base_power')
                    row[f'{side}_move_details.accuracy'] = move.get('accuracy')
                    row[f'{side}_move_details.priority'] = move.get('priority')
                timeline_rows.append(row)
    return pd.DataFrame(p1_rows), pd.DataFrame(timeline_rows), pd.DataFrame(p2_rows)

# 1. Process Test Set through the exact same Step 3 (V5) pipeline
if os.path.exists(TEST_JSONL_PATH):
    print("Building V5 features for Test Set...")
    if 'test_p1_df' not in globals():
        test_p1_df, test_timeline_df, test_p2_df = load_test_data(TEST_JSONL_PATH)
    if 'test_turn_df' not in globals() or 'net_duel_threat' not in test_turn_df.columns:
        test_turn_df = build_turn_level_dataset_v2(test_p1_df, test_timeline_df, meta_kb)
    test_t31_df, _ = compute_turn31_expected_type_features_v2(
        test_turn_df, meta_kb, p1_switch_model, p2_switch_model, p1_next_type_model, p2_next_type_model
    )
    test_battle_v5_base_df, test_entities = build_full_v5_battle_dataset(
        test_p1_df, test_timeline_df, test_p2_df, test_turn_df, test_t31_df, meta_kb
    )
else:
    test_battle_v5_base_df, test_entities = None, None

# 2. Compute Leak-Free Out-of-Fold (OOF) Smoothed Bayesian Win-Rate Priors
N_SPLITS = 10
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

def compute_oof_target_encoded_meta_features(train_ids, y_train, train_ent, test_ids, test_ent, skf_splitter):
    n_train = len(train_ids)
    oof_feats = np.zeros((n_train, 5))
    PRIOR_WEIGHT = 25.0

    def fit_entity_rates(idx_list, y_sub):
        global_mean = float(np.mean(y_sub))
        p1_pk_sum, p1_pk_cnt = Counter(), Counter()
        p2_pk_sum, p2_pk_cnt = Counter(), Counter()
        p1_mv_sum, p1_mv_cnt = Counter(), Counter()
        p2_mv_sum, p2_mv_cnt = Counter(), Counter()

        for idx, target in zip(idx_list, y_sub):
            b_id = train_ids[idx]
            ent = train_ent[b_id]
            for pk in ent['p1_alive']: p1_pk_sum[pk] += target; p1_pk_cnt[pk] += 1
            for pk in ent['p2_alive']: p2_pk_sum[pk] += target; p2_pk_cnt[pk] += 1
            for mv in ent['p1_moves']: p1_mv_sum[mv] += target; p1_mv_cnt[mv] += 1
            for mv in ent['p2_moves']: p2_mv_sum[mv] += target; p2_mv_cnt[mv] += 1

        p1_pk_rate = {k: (p1_pk_sum[k] + PRIOR_WEIGHT * global_mean) / (p1_pk_cnt[k] + PRIOR_WEIGHT) for k in p1_pk_cnt}
        p2_pk_rate = {k: (p2_pk_sum[k] + PRIOR_WEIGHT * global_mean) / (p2_pk_cnt[k] + PRIOR_WEIGHT) for k in p2_pk_cnt}
        p1_mv_rate = {k: (p1_mv_sum[k] + PRIOR_WEIGHT * global_mean) / (p1_mv_cnt[k] + PRIOR_WEIGHT) for k in p1_mv_cnt}
        p2_mv_rate = {k: (p2_mv_sum[k] + PRIOR_WEIGHT * global_mean) / (p2_mv_cnt[k] + PRIOR_WEIGHT) for k in p2_mv_cnt}
        return global_mean, p1_pk_rate, p2_pk_rate, p1_mv_rate, p2_mv_rate

    def transform_entities(id_list, ent_dict, rates):
        g_mean, p1_pk_r, p2_pk_r, p1_mv_r, p2_mv_r = rates
        out = np.zeros((len(id_list), 5))
        for i, b_id in enumerate(id_list):
            ent = ent_dict[b_id]
            r_p1_pk = np.mean([p1_pk_r.get(k, g_mean) for k in ent['p1_alive']] or [g_mean])
            r_p2_pk = np.mean([p2_pk_r.get(k, g_mean) for k in ent['p2_alive']] or [g_mean])
            r_p1_mv = np.mean([p1_mv_r.get(k, g_mean) for k in ent['p1_moves']] or [g_mean])
            r_p2_mv = np.mean([p2_mv_r.get(k, g_mean) for k in ent['p2_moves']] or [g_mean])
            out[i, 0] = r_p1_pk
            out[i, 1] = r_p2_pk
            out[i, 2] = r_p1_pk - (1.0 - r_p2_pk)
            out[i, 3] = r_p1_mv
            out[i, 4] = r_p2_mv
        return out

    for tr_idx, va_idx in skf_splitter.split(train_ids, y_train):
        rates = fit_entity_rates(tr_idx, y_train[tr_idx])
        oof_feats[va_idx] = transform_entities(train_ids[va_idx], train_ent, rates)

    cols = ['oof_p1_alive_winrate', 'oof_p2_alive_p1winrate', 'oof_alive_roster_edge', 'oof_p1_moves_winrate', 'oof_p2_moves_p1winrate']
    train_te_df = pd.DataFrame(oof_feats, columns=cols)
    train_te_df['battle_id'] = train_ids

    test_te_df = None
    if test_ids is not None:
        full_rates = fit_entity_rates(np.arange(n_train), y_train)
        test_feats = transform_entities(test_ids, test_ent, full_rates)
        test_te_df = pd.DataFrame(test_feats, columns=cols)
        test_te_df['battle_id'] = test_ids

    return train_te_df, test_te_df

train_ids_arr = battle_v5_base_df['battle_id'].values
y_v5 = battle_v5_base_df['player_won'].astype(int).values
test_ids_arr = test_battle_v5_base_df['battle_id'].values if test_battle_v5_base_df is not None else None

train_te_df, test_te_df = compute_oof_target_encoded_meta_features(
    train_ids_arr, y_v5, train_entities, test_ids_arr, test_entities, skf
)

battle_v5_df = battle_v5_base_df.merge(train_te_df, on='battle_id', how='left')
test_battle_v5_df = test_battle_v5_base_df.merge(test_te_df, on='battle_id', how='left') if test_battle_v5_base_df is not None else None

V5_FEATURE_COLS = [c for c in battle_v5_df.columns if c not in ['battle_id', 'player_won']]
print(f"Final V5 Feature Count: {len(V5_FEATURE_COLS)} features.")

X_v5 = battle_v5_df[V5_FEATURE_COLS].replace([np.inf, -np.inf], np.nan).fillna(0.0).copy()
X_test_v5 = test_battle_v5_df[V5_FEATURE_COLS].replace([np.inf, -np.inf], np.nan).fillna(0.0).copy() if test_battle_v5_df is not None else None

# 3. 10-Fold Stratified CV Training (CatBoost + XGBoost + LightGBM)
oof_lgb_v5 = np.zeros(len(X_v5))
oof_xgb_v5 = np.zeros(len(X_v5))
oof_cat_v5 = np.zeros(len(X_v5))

test_lgb_v5 = np.zeros(len(X_test_v5)) if X_test_v5 is not None else None
test_xgb_v5 = np.zeros(len(X_test_v5)) if X_test_v5 is not None else None
test_cat_v5 = np.zeros(len(X_test_v5)) if X_test_v5 is not None else None

for fold, (train_idx, val_idx) in enumerate(skf.split(X_v5, y_v5), 1):
    X_tr, y_tr = X_v5.iloc[train_idx], y_v5[train_idx]
    X_va, y_va = X_v5.iloc[val_idx], y_v5[val_idx]

    model_lgb = LGBMClassifier(
        n_estimators=1500, learning_rate=0.015, num_leaves=23, min_child_samples=45,
        colsample_bytree=0.65, subsample=0.8, reg_alpha=2.5, reg_lambda=2.5,
        random_state=42 + fold, verbose=-1
    )
    model_lgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[__import__('lightgbm').early_stopping(60, verbose=False)])
    oof_lgb_v5[val_idx] = model_lgb.predict_proba(X_va)[:, 1]

    model_xgb = XGBClassifier(
        n_estimators=1500, learning_rate=0.015, max_depth=4, min_child_weight=12,
        colsample_bytree=0.65, subsample=0.8, reg_alpha=2.5, reg_lambda=2.5,
        eval_metric='logloss', early_stopping_rounds=60, random_state=42 + fold, n_jobs=-1
    )
    model_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    oof_xgb_v5[val_idx] = model_xgb.predict_proba(X_va)[:, 1]

    model_cat = CatBoostClassifier(
        iterations=1800, learning_rate=0.025, depth=5, l2_leaf_reg=6.0,
        random_seed=42 + fold, verbose=False
    )
    model_cat.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=60, verbose=False)
    oof_cat_v5[val_idx] = model_cat.predict_proba(X_va)[:, 1]

    if X_test_v5 is not None:
        test_lgb_v5 += model_lgb.predict_proba(X_test_v5)[:, 1] / N_SPLITS
        test_xgb_v5 += model_xgb.predict_proba(X_test_v5)[:, 1] / N_SPLITS
        test_cat_v5 += model_cat.predict_proba(X_test_v5)[:, 1] / N_SPLITS

oof_v5_blend = 0.45 * oof_cat_v5 + 0.30 * oof_xgb_v5 + 0.25 * oof_lgb_v5

print("\n" + "="*68)
print("V5 FULL ENSEMBLE RESULTS (10-FOLD OOF ON 10,000 BATTLES)")
print("="*68)
for name, preds in [('LightGBM (V5)', oof_lgb_v5), ('XGBoost (V5)', oof_xgb_v5), ('CatBoost (V5)', oof_cat_v5), ('Top-3 GBDT Blend (V5)', oof_v5_blend)]:
    print(f"{name:<22} -> OOF Accuracy: {accuracy_score(y_v5, (preds >= 0.50).astype(int))*100:.2f}% | ROC-AUC: {roc_auc_score(y_v5, preds):.4f}")

if X_test_v5 is not None:
    test_v5_probs = 0.45 * test_cat_v5 + 0.30 * test_xgb_v5 + 0.25 * test_lgb_v5
    sub_v5 = pd.DataFrame({
        'battle_id': test_battle_v5_df['battle_id'].astype(int),
        'player_won': (test_v5_probs >= 0.500).astype(int)
    }).sort_values('battle_id').reset_index(drop=True)
    sub_v5_path = os.path.join(DATA_DIR, "submission_v5_full.csv")
    sub_v5.to_csv(sub_v5_path, index=False)
    print(f"\nSaved '{sub_v5_path}' successfully! (Win rate: {sub_v5['player_won'].mean()*100:.2f}%)")