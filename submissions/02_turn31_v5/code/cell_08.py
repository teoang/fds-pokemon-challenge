# ==============================================================================
# STEP 3 (V5 - SELF-CONTAINED): TURN ENGINE, TURN-31 FORECAST & FULL V5 FEATURES
# ==============================================================================
from sklearn.multioutput import MultiOutputClassifier
from lightgbm import LGBMClassifier

PHYSICAL_TYPES = {'normal', 'fighting', 'flying', 'ground', 'rock', 'bug', 'ghost', 'poison', 'steel'}
SPECIAL_TYPES  = {'fire', 'water', 'grass', 'electric', 'psychic', 'ice', 'dragon', 'dark'}

P1_NEXT_TYPE_TARGETS = [f'target_p1_next_type_{t}' for t in ALL_TYPES]
P2_NEXT_TYPE_TARGETS = [f'target_p2_next_type_{t}' for t in ALL_TYPES]

def get_boost_multiplier(stage):
    stage = max(-6, min(6, int(stage)))
    return (2.0 + stage) / 2.0 if stage >= 0 else 2.0 / (2.0 - stage)

def extract_clean_types(types_val):
    if isinstance(types_val, (list, tuple, set)):
        return [str(t).lower().strip() for t in types_val if str(t).lower().strip() not in ['nan', 'none', 'notype', '']]
    elif pd.notna(types_val):
        t_str = str(types_val).lower().strip()
        if t_str not in ['nan', 'none', 'notype', '']:
            return [t_str]
    return []

def get_type_multiplier_from_list(meta_kb, move_type, def_types):
    if pd.isna(move_type) or not def_types:
        return 1.0
    m_type = str(move_type).lower().strip()
    if m_type in ['nan', 'none', 'threequestionmarks', '']:
        return 1.0
    mult = 1.0
    for dt in def_types:
        mult *= meta_kb.type_matrix[m_type][dt]
    return mult

def estimate_gen2_duel_threat(meta_kb, atk_name, atk_types, def_name, def_types, atk_boosts=None, def_boosts=None):
    if not def_types:
        return 1.0
    atk_info = meta_kb.pokedex.get(atk_name, {})
    def_info = meta_kb.pokedex.get(def_name, {})
    b_atk = atk_info.get('base_atk', 85) * get_boost_multiplier((atk_boosts or {}).get('atk', 0))
    b_spa = atk_info.get('base_spa', 85) * get_boost_multiplier((atk_boosts or {}).get('spa', 0))
    b_def = max(20.0, def_info.get('base_def', 85) * get_boost_multiplier((def_boosts or {}).get('def', 0)))
    b_spd = max(20.0, def_info.get('base_spd', 85) * get_boost_multiplier((def_boosts or {}).get('spd', 0)))
    candidate_types = set(atk_types) if atk_types else {'normal'}
    best_damage_index = 0.5
    for m_type in candidate_types:
        type_mult = get_type_multiplier_from_list(meta_kb, m_type, def_types)
        stab = 1.5 if m_type in atk_types else 1.0
        stat_ratio = (b_atk / b_def) if m_type in PHYSICAL_TYPES else (b_spa / b_spd)
        best_damage_index = max(best_damage_index, type_mult * stab * stat_ratio)
    return best_damage_index

# Ensure Pokedex in meta_kb has clean types and combat stats
for _, row in pd.concat([p1_team_df, p2_lead_df], ignore_index=True).drop_duplicates(subset=['name']).iterrows():
    p_types = extract_clean_types(row['types']) if 'types' in row else extract_clean_types([row.get('type_1'), row.get('type_2')])
    meta_kb.pokedex[row['name']] = {
        'types': p_types,
        'base_hp': float(row['base_hp']),
        'base_atk': float(row['base_atk']),
        'base_def': float(row['base_def']),
        'base_spa': float(row['base_spa']),
        'base_spd': float(row['base_spd']),
        'base_spe': float(row['base_spe']),
    }

# 1. Turn-Level Feature Builder (Defines turn_df!)
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

def compute_turn31_expected_type_features_v2(turn_df, meta_kb, p1_sw_mod, p2_sw_mod, p1_nt_mod, p2_nt_mod):
    last_turns = turn_df.sort_values(['battle_id', 'turn']).groupby('battle_id', as_index=False).last()
    X_last = last_turns[TURN_FEATURE_COLS]

    p1_sw_prob = p1_sw_mod.predict_proba(X_last)[:, 1]
    p2_sw_prob = p2_sw_mod.predict_proba(X_last)[:, 1]
    p1_switch_type_probs = extract_multioutput_probs(p1_nt_mod, X_last)
    p2_switch_type_probs = extract_multioutput_probs(p2_nt_mod, X_last)

    p1_curr_types = last_turns[[f'p1_active_type_{t}' for t in ALL_TYPES]].values
    p2_curr_types = last_turns[[f'p2_active_type_{t}' for t in ALL_TYPES]].values

    p1_t31_type_probs = (1.0 - p1_sw_prob[:, None]) * p1_curr_types + p1_sw_prob[:, None] * p1_switch_type_probs
    p2_t31_type_probs = (1.0 - p2_sw_prob[:, None]) * p2_curr_types + p2_sw_prob[:, None] * p2_switch_type_probs

    M = np.ones((len(ALL_TYPES), len(ALL_TYPES)))
    for i, atk_t in enumerate(ALL_TYPES):
        for j, def_t in enumerate(ALL_TYPES):
            M[i, j] = meta_kb.type_matrix[atk_t][def_t]

    p1_t31_expected_offense = np.einsum('bi,ij,bj->b', p1_t31_type_probs, M, p2_t31_type_probs)
    p2_t31_expected_offense = np.einsum('bi,ij,bj->b', p2_t31_type_probs, M, p1_t31_type_probs)
    p1_vs_p2_switch_mult = np.einsum('bi,ij,bj->b', p1_curr_types, M, p2_switch_type_probs)
    p2_vs_p1_switch_mult = np.einsum('bi,ij,bj->b', p2_curr_types, M, p1_switch_type_probs)

    t31_df = pd.DataFrame({
        'battle_id': last_turns['battle_id'].values,
        't31_p1_switch_prob': p1_sw_prob,
        't31_p2_switch_prob': p2_sw_prob,
        't31_switch_prob_diff': p2_sw_prob - p1_sw_prob,
        't31_p1_expected_type_mult': p1_t31_expected_offense,
        't31_p2_expected_type_mult': p2_t31_expected_offense,
        't31_net_type_advantage': p1_t31_expected_offense - p2_t31_expected_offense,
        't31_p1_punish_switch_mult': p1_vs_p2_switch_mult,
        't31_p2_punish_switch_mult': p2_vs_p1_switch_mult,
        't30_active_net_duel_threat': last_turns['net_duel_threat'].values,
        't30_active_spe_advantage': last_turns['spe_advantage'].values,
        't30_p1_hp': last_turns['p1_hp'].values,
        't30_p2_hp': last_turns['p2_hp'].values,
        't30_active_hp_diff': last_turns['p1_hp'].values - last_turns['p2_hp'].values,
    })

    KEY_META_TYPES = ['normal', 'water', 'electric', 'steel', 'ground', 'rock', 'fighting', 'psychic', 'ghost', 'fire', 'ice']
    for t in KEY_META_TYPES:
        idx = ALL_TYPES.index(t)
        t31_df[f't31_p2_pred_type_{t}'] = p2_t31_type_probs[:, idx]
        t31_df[f't31_p1_pred_type_{t}'] = p1_t31_type_probs[:, idx]

    return t31_df, M

# 2. Full Battle-Level Feature Extractor (Combines V3 + V4 Multiplicative Type + V5 Unlogged P2 KO Detector!)
def build_full_v5_battle_dataset(p1_team_df, timeline_df, p2_lead_df, turn_df, t31_df, meta_kb):
    p1_team_info = defaultdict(dict)
    for _, row in p1_team_df.iterrows():
        b_id = row['battle_id']
        name = row['name']
        lvl = float(row.get('level', 100.0))
        item = str(row.get('item', '')).lower().strip()
        p_types = extract_clean_types(row['types']) if 'types' in row else extract_clean_types([row.get('type_1'), row.get('type_2')])
        eff_atk = float(row['base_atk']) * (2.0 if item == 'thickclub' else 1.0)
        bst = (float(row['base_hp']) + eff_atk + float(row['base_def']) +
               float(row['base_spa']) + float(row['base_spd']) + float(row['base_spe'])) * (lvl / 100.0)
        p1_team_info[b_id][name] = {
            'types': p_types, 'level': lvl, 'item': item, 'bst': bst,
            'usage': meta_kb.pokemon_usage_prob.get(name, 0.0),
            'setup_role': meta_kb.pokemon_setup_ratio.get(name, 0.0),
            'base_spe': float(row['base_spe']) * (lvl / 100.0)
        }

    turn_grp = turn_df.groupby('battle_id').agg(
        p1_mean_move_eff=('p1_move_eff_on_p2', 'mean'),
        p2_mean_move_eff=('p2_move_eff_on_p1', 'mean'),
        p1_se_turns=('p1_move_eff_on_p2', lambda s: (s >= 2.0).sum()),
        p2_se_turns=('p2_move_eff_on_p1', lambda s: (s >= 2.0).sum()),
        p1_resist_turns=('p1_resists_p2', 'sum'),
        p2_resist_turns=('p2_resists_p1', 'sum'),
        p1_tank_stay_turns=('p1_can_tank_and_stay', 'sum'),
        p2_tank_stay_turns=('p2_can_tank_and_stay', 'sum'),
        p1_mean_exp_threat=('p1_expected_threat_on_p2', 'mean'),
        p2_mean_exp_threat=('p2_expected_threat_on_p1', 'mean'),
        mean_net_duel_threat=('net_duel_threat', 'mean'),
        mean_spe_advantage=('spe_advantage', 'mean'),
        p1_faster_turns=('p1_is_faster', 'sum'),
        last_p1_bench_resist=('p1_bench_resist_count', 'last'),
    ).reset_index()

    late_turns = turn_df[turn_df['turn'] >= 21].groupby('battle_id').agg(
        late_net_duel_threat=('net_duel_threat', 'mean'),
        late_p1_se_turns=('p1_move_eff_on_p2', lambda s: (s >= 2.0).sum()),
        late_p2_se_turns=('p2_move_eff_on_p1', lambda s: (s >= 2.0).sum()),
        late_spe_advantage=('spe_advantage', 'mean'),
    ).reset_index()

    very_late_turns = turn_df[turn_df['turn'] >= 26].groupby('battle_id').agg(
        vlate_net_duel_threat=('net_duel_threat', 'mean'),
        vlate_p1_faster=('p1_is_faster', 'mean'),
    ).reset_index()

    STATUS_SEVERITY = {'fnt': 1.0, 'frz': 0.85, 'slp': 0.65, 'tox': 0.45, 'brn': 0.40, 'par': 0.35, 'psn': 0.25, 'nostatus': 0.0}
    RECOVERY_MOVES = {'rest', 'recover', 'milkdrink', 'softboiled', 'moonlight', 'synthesis', 'morningsun', 'painsplit'}
    PHAZING_MOVES = {'roar', 'whirlwind'}
    BOOM_MOVES = {'explosion', 'selfdestruct', 'destinybond'}
    CLERIC_MOVES = {'healbell'}
    SPIN_MOVES = {'rapidspin'}
    TRAP_AND_LOCK_MOVES = {'meanlook', 'spiderweb', 'perishsong', 'encore', 'pursuit', 'batonpass', 'bellydrum'}

    battle_rows = []
    battle_entities = {}
    timeline_sorted = timeline_df.sort_values(['battle_id', 'turn']).reset_index(drop=True)

    for b_id, group in timeline_sorted.groupby('battle_id', sort=False):
        rows = group.to_dict('records')
        n_turns = len(rows)
        roster = p1_team_info[b_id]

        p1_hp = {name: 1.0 for name in roster}
        p1_status = {name: 'nostatus' for name in roster}
        p1_has_rest_talk = defaultdict(bool)

        p2_hp, p2_status, p2_last_turn_seen = {}, {}, {}
        p2_last_hit_taken = defaultdict(bool)
        p2_has_rest_talk = defaultdict(bool)
        p1_revealed = set()
        p1_seen_move_types, p2_seen_move_types = defaultdict(set), defaultdict(set)
        p1_moves_set, p2_moves_set = set(), set()

        p1_cum_dmg_taken, p2_cum_dmg_taken = 0.0, 0.0
        p1_cum_healed, p2_cum_healed = 0.0, 0.0
        p1_late_dmg_taken, p2_late_dmg_taken = 0.0, 0.0

        p1_spikes_active, p2_spikes_active = 0, 0
        p1_setup_moves_used, p2_setup_moves_used = 0, 0
        p1_recovery_used, p2_recovery_used = 0, 0
        p1_phaze_used, p2_phaze_used = 0, 0
        p1_cleric_used, p2_cleric_used = 0, 0
        p1_trap_lock_used, p2_trap_lock_used = 0, 0
        p1_boom_net_gain = 0.0

        p1_switches, p2_switches = 0, 0
        p1_late_switches, p2_late_switches = 0, 0
        p1_spikes_chip_taken, p2_spikes_chip_taken = 0, 0
        p1_missed_or_no_move, p2_missed_or_no_move = 0, 0

        prev_p1, prev_p2 = None, None
        hp_diff_turn_10, hp_diff_turn_20, hp_diff_turn_30 = 0.0, 0.0, 0.0

        for r in rows:
            t = int(r['turn'])
            p1_n, p2_n = r['p1_pokemon_state.name'], r['p2_pokemon_state.name']
            p1_revealed.add(p1_n)
            p2_last_turn_seen[p2_n] = t

            c_p1_hp, c_p2_hp = float(r['p1_pokemon_state.hp_pct']), float(r['p2_pokemon_state.hp_pct'])
            old_p1_hp, old_p2_hp = p1_hp.get(p1_n, 1.0), p2_hp.get(p2_n, 1.0)

            if c_p1_hp < old_p1_hp:
                dmg1 = old_p1_hp - c_p1_hp
                p1_cum_dmg_taken += dmg1
                if t >= 21: p1_late_dmg_taken += dmg1
            elif c_p1_hp > old_p1_hp:
                p1_cum_healed += (c_p1_hp - old_p1_hp)

            if c_p2_hp < old_p2_hp:
                dmg2 = old_p2_hp - c_p2_hp
                p2_cum_dmg_taken += dmg2
                if t >= 21: p2_late_dmg_taken += dmg2
            elif c_p2_hp > old_p2_hp:
                p2_cum_healed += (c_p2_hp - old_p2_hp)

            if prev_p1 is not None and p1_n != prev_p1:
                p1_switches += 1
                if t >= 21: p1_late_switches += 1
                if p2_spikes_active > 0 and 'flying' not in roster.get(p1_n, {}).get('types', []):
                    p1_spikes_chip_taken += 1
            if prev_p2 is not None and p2_n != prev_p2:
                p2_switches += 1
                if t >= 21: p2_late_switches += 1
                if p1_spikes_active > 0 and 'flying' not in meta_kb.pokedex.get(p2_n, {}).get('types', []):
                    p2_spikes_chip_taken += 1
            prev_p1, prev_p2 = p1_n, p2_n

            m1 = str(r.get('p1_move_details.name') or '').lower().strip()
            m2 = str(r.get('p2_move_details.name') or '').lower().strip()
            m1_t = str(r.get('p1_move_details.type') or '').lower().strip()
            m2_t = str(r.get('p2_move_details.type') or '').lower().strip()
            cat1 = str(r.get('p1_move_details.category') or '').lower().strip()
            cat2 = str(r.get('p2_move_details.category') or '').lower().strip()

            if m1 not in ['', 'nan', 'none']: p1_moves_set.add(m1)
            if m2 not in ['', 'nan', 'none']: p2_moves_set.add(m2)
            if cat1 in ['physical', 'special'] and m1_t in ALL_TYPES: p1_seen_move_types[p1_n].add(m1_t)
            if cat2 in ['physical', 'special'] and m2_t in ALL_TYPES: p2_seen_move_types[p2_n].add(m2_t)

            p2_last_hit_taken[p2_n] = (cat1 in ['physical', 'special'] or m1 in {'pursuit', 'seismictoss', 'nightshade', 'superfang', 'counter', 'mirrorcoat'})

            if m1 in BOOM_MOVES or m2 in BOOM_MOVES:
                if c_p1_hp == 0.0 and c_p2_hp == 0.0:
                    p1_boom_net_gain += (old_p2_hp - old_p1_hp)
                elif m1 in BOOM_MOVES and c_p1_hp == 0.0 and c_p2_hp > 0.0:
                    p1_boom_net_gain -= old_p1_hp
                elif m2 in BOOM_MOVES and c_p2_hp == 0.0 and c_p1_hp > 0.0:
                    p1_boom_net_gain += old_p2_hp

            p1_hp[p1_n], p2_hp[p2_n] = c_p1_hp, c_p2_hp
            p1_status[p1_n] = str(r['p1_pokemon_state.status']).lower().strip()
            p2_status[p2_n] = str(r['p2_pokemon_state.status']).lower().strip()

            if m1 in ['', 'nan', 'none']: p1_missed_or_no_move += 1
            if m2 in ['', 'nan', 'none']: p2_missed_or_no_move += 1

            if m1 == 'spikes': p1_spikes_active = 1
            if m2 == 'spikes': p2_spikes_active = 1
            if m1 in SPIN_MOVES and 'ghost' not in meta_kb.pokedex.get(p2_n, {}).get('types', []):
                p2_spikes_active = 0
            if m2 in SPIN_MOVES and 'ghost' not in roster.get(p1_n, {}).get('types', []):
                p1_spikes_active = 0

            if m1 in meta_kb.setup_and_support_moves: p1_setup_moves_used += 1
            if m2 in meta_kb.setup_and_support_moves: p2_setup_moves_used += 1
            if m1 in RECOVERY_MOVES: p1_recovery_used += 1
            if m2 in RECOVERY_MOVES: p2_recovery_used += 1
            if m1 in PHAZING_MOVES: p1_phaze_used += 1
            if m2 in PHAZING_MOVES: p2_phaze_used += 1
            if m1 in CLERIC_MOVES: p1_cleric_used += 1
            if m2 in CLERIC_MOVES: p2_cleric_used += 1
            if m1 in TRAP_AND_LOCK_MOVES: p1_trap_lock_used += 1
            if m2 in TRAP_AND_LOCK_MOVES: p2_trap_lock_used += 1
            if m1 in {'rest', 'sleeptalk'}: p1_has_rest_talk[p1_n] = True
            if m2 in {'rest', 'sleeptalk'}: p2_has_rest_talk[p2_n] = True

            if t in [10, 20, 30]:
                p1_tot_t = sum(p1_hp.values())
                p2_unrev_t = max(0, 6 - len(p2_hp))
                p2_tot_t = sum(p2_hp.values()) + p2_unrev_t * 1.0
                if t == 10: hp_diff_turn_10 = p1_tot_t - p2_tot_t
                elif t == 20: hp_diff_turn_20 = p1_tot_t - p2_tot_t
                elif t == 30: hp_diff_turn_30 = p1_tot_t - p2_tot_t

        # V5 Unlogged P2 KO Detector (Turns 1..N-1)
        p2_unlogged_ko_score = 0.0
        p2_swept_by_boosted_p1 = 0
        for i in range(n_turns - 1):
            r = rows[i]
            next_r = rows[i + 1]
            t = int(r['turn'])
            p1_n, p2_n = r['p1_pokemon_state.name'], r['p2_pokemon_state.name']
            next_p2_n = next_r['p2_pokemon_state.name']
            if next_p2_n != p2_n and p2_last_turn_seen[p2_n] == t and float(r['p2_pokemon_state.hp_pct']) > 0.0:
                cat1 = str(r.get('p1_move_details.category') or '').lower().strip()
                m1_type = str(r.get('p1_move_details.type') or '').lower().strip()
                bp1 = float(r.get('p1_move_details.base_power') or 0.0)
                p1_atk_boost = max(int(r.get('p1_pokemon_state.boosts.atk', 0)), int(r.get('p1_pokemon_state.boosts.spa', 0)))
                if cat1 in ['physical', 'special'] and bp1 > 0:
                    p2_types = meta_kb.pokedex.get(p2_n, {}).get('types', [])
                    eff = get_type_multiplier_from_list(meta_kb, m1_type, p2_types)
                    cur_p2_hp = float(r['p2_pokemon_state.hp_pct'])
                    lethal_power = (bp1 / 100.0) * eff * get_boost_multiplier(p1_atk_boost)
                    if cur_p2_hp <= 0.45 or lethal_power >= 1.5:
                        p2_unlogged_ko_score += min(1.0, lethal_power / max(0.25, cur_p2_hp))
                    if p1_atk_boost >= 1 and lethal_power >= 1.2:
                        p2_swept_by_boosted_p1 += 1

        p1_fainted = sum(1 for name, hp in p1_hp.items() if hp <= 0.0 or p1_status[name] == 'fnt')
        p2_logged_fainted = sum(1 for name, hp in p2_hp.items() if hp <= 0.0 or p2_status[name] == 'fnt')
        p2_probable_unlogged_kos = sum(1 for name, hp in p2_hp.items() if 0.0 < hp <= 0.38 and p2_last_turn_seen[name] <= 25 and p2_last_hit_taken[name])

        p1_critical_hp_count = sum(1 for name, hp in p1_hp.items() if 0.0 < hp <= 0.25)
        p2_critical_hp_count = sum(1 for name, hp in p2_hp.items() if 0.0 < hp <= 0.25)

        p1_unrevealed = max(0, 6 - len(p1_revealed))
        p2_unrevealed = max(0, 6 - len(p2_hp))

        p1_total_hp = sum(p1_hp.values())
        p2_total_hp = sum(p2_hp.values()) + p2_unrevealed * 1.0
        p2_adjusted_total_hp = p2_total_hp - 0.25 * p2_probable_unlogged_kos

        p1_status_penalty, p2_status_penalty = 0.0, 0.0
        p1_statused_alive, p2_statused_alive = 0, 0
        for n, s in p1_status.items():
            if p1_hp[n] > 0 and s not in ['nostatus', 'fnt']:
                p1_statused_alive += 1
                p1_status_penalty += (0.15 if (s == 'slp' and p1_has_rest_talk[n]) else STATUS_SEVERITY.get(s, 0.3))
        for n, s in p2_status.items():
            if p2_hp[n] > 0 and s not in ['nostatus', 'fnt']:
                p2_statused_alive += 1
                p2_status_penalty += (0.15 if (s == 'slp' and p2_has_rest_talk[n]) else STATUS_SEVERITY.get(s, 0.3))

        p1_effective_hp = p1_total_hp - 0.5 * p1_status_penalty
        p2_effective_hp = p2_adjusted_total_hp - 0.5 * p2_status_penalty

        p1_snorlax_hp = p1_hp.get('snorlax', 0.0)
        p2_snorlax_hp = p2_hp.get('snorlax', 0.822 * 1.0 if p2_unrevealed > 0 else 0.0)

        # Best Bench Counter-Matchup Score & V4 Exact Multiplicative Type Matchups
        p1_alive_mons = [(n, hp, roster[n]['types']) for n, hp in p1_hp.items() if hp > 0]
        p2_alive_mons = [(n, hp, meta_kb.pokedex.get(n, {}).get('types', [])) for n, hp in p2_hp.items() if hp > 0]

        p1_best_counter_scores = []
        pair_adv_sum, total_w = 0.0, 0.0
        p1_se_coverage_count, p2_se_coverage_count = 0, 0
        p1_immunity_count, p2_immunity_count = 0, 0

        for p2_n, p2_h, p2_t in p2_alive_mons:
            atk2 = set(p2_t) | p2_seen_move_types[p2_n]
            best_score = -5.0
            for p1_n, p1_h, p1_t in p1_alive_mons:
                atk1 = set(p1_t) | p1_seen_move_types[p1_n]
                off_threat = estimate_gen2_duel_threat(meta_kb, p1_n, p1_t, p2_n, p2_t)
                def_threat = estimate_gen2_duel_threat(meta_kb, p2_n, p2_t, p1_n, p1_t)
                best_score = max(best_score, (off_threat - def_threat) * p1_h)

                w = p1_h * p2_h
                eff_1_on_2 = max([get_type_multiplier_from_list(meta_kb, m_t, p2_t) for m_t in atk1] or [1.0])
                eff_2_on_1 = max([get_type_multiplier_from_list(meta_kb, m_t, p1_t) for m_t in atk2] or [1.0])
                pair_adv_sum += w * (eff_1_on_2 - eff_2_on_1)
                total_w += w
                if eff_1_on_2 >= 2.0: p1_se_coverage_count += 1
                if eff_2_on_1 >= 2.0: p2_se_coverage_count += 1
                if any(get_type_multiplier_from_list(meta_kb, m_t, p1_t) == 0.0 for m_t in atk2): p1_immunity_count += 1
                if any(get_type_multiplier_from_list(meta_kb, m_t, p2_t) == 0.0 for m_t in atk1): p2_immunity_count += 1

            if best_score > -5.0:
                p1_best_counter_scores.append(best_score)

        p1_counter_coverage_mean = float(np.mean(p1_best_counter_scores)) if p1_best_counter_scores else 0.0
        p1_counter_coverage_min = float(np.min(p1_best_counter_scores)) if p1_best_counter_scores else 0.0
        n_pairs = max(1, len(p1_alive_mons) * len(p2_alive_mons))

        p1_alive_bst = sum(roster[n]['bst'] * hp for n, hp in p1_hp.items() if hp > 0)
        p2_alive_bst = sum(
            sum(meta_kb.pokedex.get(n, {}).get(s, 85) for s in ['base_hp', 'base_atk', 'base_def', 'base_spa', 'base_spd', 'base_spe']) * hp
            for n, hp in p2_hp.items() if hp > 0
        ) + p2_unrevealed * 535.0

        p1_max_alive_spe = max([roster[n]['base_spe'] for n, hp in p1_hp.items() if hp > 0] or [0.0])
        p2_max_alive_spe = max([meta_kb.pokedex.get(n, {}).get('base_spe', 85) for n, hp in p2_hp.items() if hp > 0] or [100.0])

        p1_alive_meta_score = sum(roster[n]['usage'] * hp for n, hp in p1_hp.items() if hp > 0)
        p2_alive_meta_score = sum(meta_kb.pokemon_usage_prob.get(n, 0.1) * hp for n, hp in p2_hp.items() if hp > 0) + p2_unrevealed * 0.25

        last_r = rows[-1]
        lp1, lp2 = last_r['p1_pokemon_state.name'], last_r['p2_pokemon_state.name']
        lt1, lt2 = roster.get(lp1, {}).get('types', []), meta_kb.pokedex.get(lp2, {}).get('types', [])
        latk1, latk2 = set(lt1) | p1_seen_move_types[lp1], set(lt2) | p2_seen_move_types[lp2]
        t30_eff1 = max([get_type_multiplier_from_list(meta_kb, m_t, lt2) for m_t in latk1] or [1.0])
        t30_eff2 = max([get_type_multiplier_from_list(meta_kb, m_t, lt1) for m_t in latk2] or [1.0])

        p1_final_boosts = sum(int(last_r.get(f'p1_pokemon_state.boosts.{s}', 0)) for s in ['atk', 'def', 'spa', 'spd', 'spe'])
        p2_final_boosts = sum(int(last_r.get(f'p2_pokemon_state.boosts.{s}', 0)) for s in ['atk', 'def', 'spa', 'spd', 'spe'])
        p1_final_off_boosts = int(last_r.get('p1_pokemon_state.boosts.atk', 0)) + int(last_r.get('p1_pokemon_state.boosts.spa', 0))
        p2_final_off_boosts = int(last_r.get('p2_pokemon_state.boosts.atk', 0)) + int(last_r.get('p2_pokemon_state.boosts.spa', 0))

        p1_eff_str = str(last_r.get('p1_pokemon_state.effects', '')).lower()
        p2_eff_str = str(last_r.get('p2_pokemon_state.effects', '')).lower()
        p1_has_sub = int('substitute' in p1_eff_str)
        p2_has_sub = int('substitute' in p2_eff_str)
        p1_trapped = int('trapped' in p1_eff_str or 'perish' in p1_eff_str)
        p2_trapped = int('trapped' in p2_eff_str or 'perish' in p2_eff_str)

        t30_p1_fnt = int(str(last_r['p1_pokemon_state.status']) == 'fnt' or float(last_r['p1_pokemon_state.hp_pct']) == 0.0)
        t30_p2_fnt = int(str(last_r['p2_pokemon_state.status']) == 'fnt' or float(last_r['p2_pokemon_state.hp_pct']) == 0.0)

        battle_rows.append({
            'battle_id': b_id,
            'p1_total_hp': p1_total_hp,
            'p2_total_hp': p2_total_hp,
            'total_hp_diff': p1_total_hp - p2_total_hp,
            'adjusted_hp_diff': p1_total_hp - p2_adjusted_total_hp,
            'total_hp_ratio': (p1_total_hp + 0.1) / (p2_adjusted_total_hp + 0.1),
            'effective_hp_diff': p1_effective_hp - p2_effective_hp,
            'net_cum_dmg_dealt': p2_cum_dmg_taken - p1_cum_dmg_taken,
            'net_late_dmg_dealt': p2_late_dmg_taken - p1_late_dmg_taken,
            'net_cum_healed': p1_cum_healed - p2_cum_healed,
            'p1_fainted': p1_fainted,
            'p2_fainted': p2_logged_fainted,
            'p2_probable_unlogged_kos': p2_probable_unlogged_kos,
            'fainted_diff': (p2_logged_fainted + p2_probable_unlogged_kos) - p1_fainted,
            'p2_unlogged_ko_score': p2_unlogged_ko_score,
            'p2_swept_by_boosted_p1': p2_swept_by_boosted_p1,
            'true_estimated_ko_diff': (p2_logged_fainted + p2_unlogged_ko_score) - p1_fainted,
            't30_p1_fnt': t30_p1_fnt,
            't30_p2_fnt': t30_p2_fnt,
            't30_fnt_diff': t30_p2_fnt - t30_p1_fnt,
            'critical_hp_diff': p2_critical_hp_count - p1_critical_hp_count,
            'p1_unrevealed_count': p1_unrevealed,
            'p2_unrevealed_count': p2_unrevealed,
            'unrevealed_diff': p1_unrevealed - p2_unrevealed,
            'p1_statused_alive': p1_statused_alive,
            'p2_statused_alive': p2_statused_alive,
            'status_diff': p2_statused_alive - p1_statused_alive,
            'status_penalty_diff': p2_status_penalty - p1_status_penalty,
            'p1_snorlax_hp': p1_snorlax_hp,
            'p2_snorlax_hp': p2_snorlax_hp,
            'snorlax_hp_diff': p1_snorlax_hp - p2_snorlax_hp,
            'p1_spikes_active': p1_spikes_active,
            'p2_spikes_active': p2_spikes_active,
            'spikes_chip_diff': p2_spikes_chip_taken - p1_spikes_chip_taken,
            'p1_setup_moves_used': p1_setup_moves_used,
            'p2_setup_moves_used': p2_setup_moves_used,
            'setup_moves_diff': p1_setup_moves_used - p2_setup_moves_used,
            'recovery_moves_diff': p1_recovery_used - p2_recovery_used,
            'phaze_moves_diff': p1_phaze_used - p2_phaze_used,
            'cleric_moves_diff': p1_cleric_used - p2_cleric_used,
            'trap_lock_moves_diff': p1_trap_lock_used - p2_trap_lock_used,
            'p1_boom_net_gain': p1_boom_net_gain,
            'p1_switches': p1_switches,
            'p2_switches': p2_switches,
            'switch_diff': p2_switches - p1_switches,
            'late_switch_diff': p2_late_switches - p1_late_switches,
            'action_efficiency_diff': p2_missed_or_no_move - p1_missed_or_no_move,
            'hp_diff_turn_10': hp_diff_turn_10,
            'hp_diff_turn_20': hp_diff_turn_20,
            'hp_momentum_20_to_30': hp_diff_turn_30 - hp_diff_turn_20,
            'hp_momentum_10_to_30': hp_diff_turn_30 - hp_diff_turn_10,
            'exact_mult_team_type_adv': pair_adv_sum / max(0.01, total_w),
            'se_coverage_ratio_diff': (p1_se_coverage_count - p2_se_coverage_count) / n_pairs,
            'immunity_matchup_diff': (p1_immunity_count - p2_immunity_count) / n_pairs,
            't30_revealed_move_type_adv': t30_eff1 - t30_eff2,
            'p1_counter_coverage_mean': p1_counter_coverage_mean,
            'p1_counter_coverage_min': p1_counter_coverage_min,
            'alive_bst_diff': p1_alive_bst - p2_alive_bst,
            'max_alive_spe_diff': p1_max_alive_spe - p2_max_alive_spe,
            'alive_meta_score_diff': p1_alive_meta_score - p2_alive_meta_score,
            'final_boost_diff': p1_final_boosts - p2_final_boosts,
            'final_off_boost_diff': p1_final_off_boosts - p2_final_off_boosts,
            'sub_active_diff': p1_has_sub - p2_has_sub,
            'trapped_diff': p2_trapped - p1_trapped,
        })

        battle_entities[b_id] = {
            'p1_alive': [n for n, h in p1_hp.items() if h > 0],
            'p2_alive': [n for n, h in p2_hp.items() if h > 0],
            'p1_moves': list(p1_moves_set),
            'p2_moves': list(p2_moves_set),
        }

    battle_df = pd.DataFrame(battle_rows)
    battle_df = battle_df.merge(turn_grp, on='battle_id', how='left')
    battle_df = battle_df.merge(late_turns, on='battle_id', how='left').fillna(0.0)
    battle_df = battle_df.merge(very_late_turns, on='battle_id', how='left').fillna(0.0)
    battle_df = battle_df.merge(t31_df, on='battle_id', how='left')

    if 'player_won' in p2_lead_df.columns:
        battle_df = battle_df.merge(p2_lead_df[['battle_id', 'player_won']], on='battle_id', how='left')

    return battle_df, battle_entities

print("Computing Turn-31 Forecast & Full V5 Battle Features for Training Set...")
t31_features_df, TYPE_MATRIX_17x17 = compute_turn31_expected_type_features_v2(
    turn_df, meta_kb, p1_switch_model, p2_switch_model, p1_next_type_model, p2_next_type_model
)
battle_v5_base_df, train_entities = build_full_v5_battle_dataset(
    p1_team_df, timeline_df, p2_lead_df, turn_df, t31_features_df, meta_kb
)
print(f"Step 3 (V5) Complete! Base Battle Dataset Shape: {battle_v5_base_df.shape}")