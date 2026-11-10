# ==============================================================================
# STEP 2B (V2): TURN ENGINE, STAGE-1 SWITCH/NEXT-TYPE MODELS & DATASET BUILDER
# ==============================================================================
from sklearn.multioutput import MultiOutputClassifier
from lightgbm import LGBMClassifier

P1_NEXT_TYPE_TARGETS = [f'target_p1_next_type_{t}' for t in ALL_TYPES]
P2_NEXT_TYPE_TARGETS = [f'target_p2_next_type_{t}' for t in ALL_TYPES]

def build_turn_level_dataset_v2(p1_team_df, timeline_df, meta_kb):
    p1_rosters = defaultdict(dict)
    for _, row in p1_team_df.iterrows():
        b_id = row['battle_id']
        name = row['name']
        p_types = extract_clean_types(row['types']) if 'types' in row else extract_clean_types([row.get('type_1'), row.get('type_2')])
        p1_rosters[b_id][name] = p_types

    timeline_sorted = timeline_df.sort_values(['battle_id', 'turn']).reset_index(drop=True)
    turn_records = []

    for b_id, group in timeline_sorted.groupby('battle_id', sort=False):
        rows = group.to_dict('records')
        n_turns = len(rows)
        p1_roster = p1_rosters[b_id]
        p1_hp_tracker = {name: 1.0 for name in p1_roster}
        p2_revealed_set = set()
        p2_hp_tracker = {}

        for i, row in enumerate(rows):
            p1_name, p2_name = row['p1_pokemon_state.name'], row['p2_pokemon_state.name']
            p1_hp, p2_hp = float(row['p1_pokemon_state.hp_pct']), float(row['p2_pokemon_state.hp_pct'])
            p1_hp_tracker[p1_name], p2_hp_tracker[p2_name] = p1_hp, p2_hp
            p2_revealed_set.add(p2_name)

            p1_types = meta_kb.pokedex.get(p1_name, {}).get('types', p1_roster.get(p1_name, []))
            p2_types = meta_kb.pokedex.get(p2_name, {}).get('types', [])

            p1_boosts = {s: int(row.get(f'p1_pokemon_state.boosts.{s}', 0)) for s in ['atk', 'def', 'spa', 'spd', 'spe']}
            p2_boosts = {s: int(row.get(f'p2_pokemon_state.boosts.{s}', 0)) for s in ['atk', 'def', 'spa', 'spd', 'spe']}

            p1_move_type, p2_move_type = row.get('p1_move_details.type'), row.get('p2_move_details.type')
            p1_bp = float(row.get('p1_move_details.base_power') or 0.0)
            p2_bp = float(row.get('p2_move_details.base_power') or 0.0)

            p1_move_eff_on_p2 = get_type_multiplier_from_list(meta_kb, p1_move_type, p2_types)
            p2_move_eff_on_p1 = get_type_multiplier_from_list(meta_kb, p2_move_type, p1_types)

            p1_expected_threat_on_p2 = estimate_gen2_duel_threat(meta_kb, p1_name, p1_types, p2_name, p2_types, p1_boosts, p2_boosts)
            p2_expected_threat_on_p1 = estimate_gen2_duel_threat(meta_kb, p2_name, p2_types, p1_name, p1_types, p2_boosts, p1_boosts)

            p1_boost_sum, p2_boost_sum = sum(p1_boosts.values()), sum(p2_boosts.values())
            p1_setup_role = meta_kb.pokemon_setup_ratio.get(p1_name, 0.0)
            p2_setup_role = meta_kb.pokemon_setup_ratio.get(p2_name, 0.0)

            p1_resists_p2 = int(p2_move_eff_on_p1 <= 0.5 or p2_expected_threat_on_p1 <= 0.75)
            p2_resists_p1 = int(p1_move_eff_on_p2 <= 0.5 or p1_expected_threat_on_p2 <= 0.75)

            p1_can_tank_and_stay = int((p1_hp >= 0.55 and p2_expected_threat_on_p1 <= 1.2) or p1_resists_p2 == 1 or p1_boost_sum > 0)
            p2_can_tank_and_stay = int((p2_hp >= 0.55 and p1_expected_threat_on_p2 <= 1.2) or p2_resists_p1 == 1 or p2_boost_sum > 0)

            p1_bench_resist_count = 0
            p1_alive_bench_types = set()
            for b_name, b_types in p1_roster.items():
                if b_name != p1_name and p1_hp_tracker.get(b_name, 1.0) > 0.0:
                    p1_alive_bench_types.update(b_types)
                    if estimate_gen2_duel_threat(meta_kb, p2_name, p2_types, b_name, b_types) <= 0.8:
                        p1_bench_resist_count += 1

            p2_rem_type_probs = meta_kb.predict_remaining_team_type_probs(p2_revealed_set)
            p2_known_bench_types = set()
            for rev_name in p2_revealed_set:
                if rev_name != p2_name and p2_hp_tracker.get(rev_name, 1.0) > 0.0:
                    p2_known_bench_types.update(meta_kb.pokedex.get(rev_name, {}).get('types', []))

            p1_spe = meta_kb.pokedex.get(p1_name, {}).get('base_spe', 80) * get_boost_multiplier(p1_boosts['spe'])
            p2_spe = meta_kb.pokedex.get(p2_name, {}).get('base_spe', 80) * get_boost_multiplier(p2_boosts['spe'])
            if str(row.get('p1_pokemon_state.status')) == 'par': p1_spe *= 0.25
            if str(row.get('p2_pokemon_state.status')) == 'par': p2_spe *= 0.25
            spe_advantage = p1_spe - p2_spe

            rec = {
                'battle_id': b_id, 'turn': row['turn'],
                'p1_hp': p1_hp, 'p2_hp': p2_hp, 'p1_bp': p1_bp, 'p2_bp': p2_bp,
                'p1_move_eff_on_p2': p1_move_eff_on_p2, 'p2_move_eff_on_p1': p2_move_eff_on_p1,
                'p1_expected_threat_on_p2': p1_expected_threat_on_p2, 'p2_expected_threat_on_p1': p2_expected_threat_on_p1,
                'net_duel_threat': p1_expected_threat_on_p2 - p2_expected_threat_on_p1,
                'p1_resists_p2': p1_resists_p2, 'p2_resists_p1': p2_resists_p1,
                'p1_boost_sum': p1_boost_sum, 'p2_boost_sum': p2_boost_sum,
                'p1_setup_role': p1_setup_role, 'p2_setup_role': p2_setup_role,
                'p1_can_tank_and_stay': p1_can_tank_and_stay, 'p2_can_tank_and_stay': p2_can_tank_and_stay,
                'p1_bench_resist_count': p1_bench_resist_count, 'p2_revealed_count': len(p2_revealed_set),
                'spe_advantage': spe_advantage, 'p1_is_faster': int(spe_advantage > 0),
            }

            for t in ALL_TYPES:
                rec[f'p1_active_type_{t}'] = int(t in p1_types)
                rec[f'p2_active_type_{t}'] = int(t in p2_types)
                rec[f'p1_bench_has_{t}'] = int(t in p1_alive_bench_types)
                rec[f'p2_rem_prob_{t}'] = 1.0 if (t in p2_known_bench_types) else p2_rem_type_probs.get(t, 0.0)

            if i < n_turns - 1:
                next_p1_name, next_p2_name = rows[i + 1]['p1_pokemon_state.name'], rows[i + 1]['p2_pokemon_state.name']
                next_p1_types = meta_kb.pokedex.get(next_p1_name, {}).get('types', p1_roster.get(next_p1_name, []))
                next_p2_types = meta_kb.pokedex.get(next_p2_name, {}).get('types', [])
                rec['has_next_turn'] = 1
                rec['target_p1_switched'] = int(next_p1_name != p1_name)
                rec['target_p2_switched'] = int(next_p2_name != p2_name)
                for t in ALL_TYPES:
                    rec[f'target_p1_next_type_{t}'] = int(t in next_p1_types)
                    rec[f'target_p2_next_type_{t}'] = int(t in next_p2_types)
            else:
                rec['has_next_turn'] = 0

            turn_records.append(rec)

    return pd.DataFrame(turn_records)

TURN_FEATURE_COLS = [
    'turn', 'p1_hp', 'p2_hp', 'p1_bp', 'p2_bp',
    'p1_move_eff_on_p2', 'p2_move_eff_on_p1',
    'p1_expected_threat_on_p2', 'p2_expected_threat_on_p1', 'net_duel_threat',
    'p1_resists_p2', 'p2_resists_p1',
    'p1_boost_sum', 'p2_boost_sum',
    'p1_setup_role', 'p2_setup_role',
    'p1_can_tank_and_stay', 'p2_can_tank_and_stay',
    'p1_bench_resist_count', 'p2_revealed_count', 'spe_advantage', 'p1_is_faster'
] + [f'p1_active_type_{t}' for t in ALL_TYPES] \
  + [f'p2_active_type_{t}' for t in ALL_TYPES] \
  + [f'p1_bench_has_{t}' for t in ALL_TYPES] \
  + [f'p2_rem_prob_{t}' for t in ALL_TYPES]

# Build turn_df and train Stage-1 models if not already in memory
if 'turn_df' not in globals() or 'net_duel_threat' not in turn_df.columns:
    print("Building turn_df and training Stage-1 Switch & Next-Type models...")
    turn_df = build_turn_level_dataset_v2(p1_team_df, timeline_df, meta_kb)
    train_turns = turn_df[turn_df['has_next_turn'] == 1].reset_index(drop=True)
    X_turns = train_turns[TURN_FEATURE_COLS]

    p1_switch_model = LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31, subsample_for_bin=50000, random_state=42, verbose=-1)
    p2_switch_model = LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31, subsample_for_bin=50000, random_state=42, verbose=-1)
    p1_switch_model.fit(X_turns, train_turns['target_p1_switched'])
    p2_switch_model.fit(X_turns, train_turns['target_p2_switched'])

    p1_switch_mask = train_turns['target_p1_switched'] == 1
    p2_switch_mask = train_turns['target_p2_switched'] == 1
    p1_next_type_model = MultiOutputClassifier(LGBMClassifier(n_estimators=120, learning_rate=0.05, num_leaves=15, random_state=42, verbose=-1))
    p2_next_type_model = MultiOutputClassifier(LGBMClassifier(n_estimators=120, learning_rate=0.05, num_leaves=15, random_state=42, verbose=-1))
    p1_next_type_model.fit(X_turns[p1_switch_mask], train_turns.loc[p1_switch_mask, P1_NEXT_TYPE_TARGETS])
    p2_next_type_model.fit(X_turns[p2_switch_mask], train_turns.loc[p2_switch_mask, P2_NEXT_TYPE_TARGETS])

def extract_multioutput_probs(multi_model, X_df):
    preds_list = multi_model.predict_proba(X_df)
    prob_matrix = np.zeros((len(X_df), len(ALL_TYPES)))
    for idx, p in enumerate(preds_list):
        prob_matrix[:, idx] = p[:, 1] if p.shape[1] > 1 else 0.0
    return prob_matrix

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
