# ==============================================================================
# STEP 3 (V2): DEEP COMPETITIVE BATTLE FEATURE AGGREGATION
# ==============================================================================
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

    # Expected Turn-31 offensive multipliers
    p1_t31_expected_offense = np.einsum('bi,ij,bj->b', p1_t31_type_probs, M, p2_t31_type_probs)
    p2_t31_expected_offense = np.einsum('bi,ij,bj->b', p2_t31_type_probs, M, p1_t31_type_probs)

    # Also compute: If P1 stays in vs P2 switches (and vice versa)
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

    # Keep only the top high-impact competitive types on Turn 31 to prevent tree overfitting on rare types
    KEY_META_TYPES = ['normal', 'water', 'electric', 'steel', 'ground', 'rock', 'fighting', 'psychic', 'ghost', 'fire', 'ice']
    for t in KEY_META_TYPES:
        idx = ALL_TYPES.index(t)
        t31_df[f't31_p2_pred_type_{t}'] = p2_t31_type_probs[:, idx]
        t31_df[f't31_p1_pred_type_{t}'] = p1_t31_type_probs[:, idx]

    return t31_df, M


def build_battle_level_features_v2(p1_team_df, timeline_df, p2_lead_df, turn_df, t31_df, meta_kb, M_17x17):
    p1_team_info = defaultdict(dict)
    for _, row in p1_team_df.iterrows():
        b_id = row['battle_id']
        name = row['name']
        p_types = extract_clean_types(row['types']) if 'types' in row else extract_clean_types([row.get('type_1'), row.get('type_2')])
        p1_team_info[b_id][name] = {
            'types': p_types,
            'bst': float(row['base_hp'] + row['base_atk'] + row['base_def'] + row['base_spa'] + row['base_spd'] + row['base_spe']),
            'usage': meta_kb.pokemon_usage_prob.get(name, 0.0),
            'setup_role': meta_kb.pokemon_setup_ratio.get(name, 0.0),
            'base_spe': float(row['base_spe'])
        }

    # Global 30-turn aggregations
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

    # Late-game window (Turns 21 to 30) aggregations
    late_turns = turn_df[turn_df['turn'] >= 21].groupby('battle_id').agg(
        late_net_duel_threat=('net_duel_threat', 'mean'),
        late_p1_se_turns=('p1_move_eff_on_p2', lambda s: (s >= 2.0).sum()),
        late_p2_se_turns=('p2_move_eff_on_p1', lambda s: (s >= 2.0).sum()),
        late_spe_advantage=('spe_advantage', 'mean'),
    ).reset_index()

    # Status severity weights (when NOT a Rest/SleepTalk user)
    STATUS_SEVERITY = {'fnt': 1.0, 'frz': 0.85, 'slp': 0.65, 'tox': 0.45, 'brn': 0.40, 'par': 0.35, 'psn': 0.25, 'nostatus': 0.0}
    RECOVERY_MOVES = {'rest', 'recover', 'milkdrink', 'softboiled', 'moonlight', 'synthesis'}
    PHAZING_MOVES = {'roar', 'whirlwind'}

    battle_rows = []
    timeline_sorted = timeline_df.sort_values(['battle_id', 'turn']).reset_index(drop=True)

    for b_id, group in timeline_sorted.groupby('battle_id', sort=False):
        rows = group.to_dict('records')
        roster = p1_team_info[b_id]

        p1_hp = {name: 1.0 for name in roster}
        p1_status = {name: 'nostatus' for name in roster}
        p1_min_hp = {name: 1.0 for name in roster}
        p1_has_rest_talk = defaultdict(bool)

        p2_hp = {}
        p2_status = {}
        p2_min_hp = {}
        p2_has_rest_talk = defaultdict(bool)
        p1_revealed = set()

        p1_spikes_used, p2_spikes_used = 0, 0
        p1_setup_moves_used, p2_setup_moves_used = 0, 0
        p1_recovery_used, p2_recovery_used = 0, 0
        p1_phaze_used, p2_phaze_used = 0, 0
        p1_switches, p2_switches = 0, 0
        p1_late_switches, p2_late_switches = 0, 0
        p1_spikes_chip_taken, p2_spikes_chip_taken = 0, 0
        p1_missed_or_no_move, p2_missed_or_no_move = 0, 0

        prev_p1, prev_p2 = None, None
        hp_diff_turn_10, hp_diff_turn_20, hp_diff_turn_30 = 0.0, 0.0, 0.0

        for r in rows:
            t = int(r['turn'])
            p1_n = r['p1_pokemon_state.name']
            p2_n = r['p2_pokemon_state.name']
            p1_revealed.add(p1_n)

            c_p1_hp = float(r['p1_pokemon_state.hp_pct'])
            c_p2_hp = float(r['p2_pokemon_state.hp_pct'])

            # Detect switches & grounded Spikes chip damage
            if prev_p1 is not None and p1_n != prev_p1:
                p1_switches += 1
                if t >= 21: p1_late_switches += 1
                if p2_spikes_used > 0 and 'flying' not in roster.get(p1_n, {}).get('types', []):
                    p1_spikes_chip_taken += 1
            if prev_p2 is not None and p2_n != prev_p2:
                p2_switches += 1
                if t >= 21: p2_late_switches += 1
                if p1_spikes_used > 0 and 'flying' not in meta_kb.pokedex.get(p2_n, {}).get('types', []):
                    p2_spikes_chip_taken += 1
            prev_p1, prev_p2 = p1_n, p2_n

            p1_hp[p1_n] = c_p1_hp
            p2_hp[p2_n] = c_p2_hp
            p1_min_hp[p1_n] = min(p1_min_hp.get(p1_n, 1.0), c_p1_hp)
            p2_min_hp[p2_n] = min(p2_min_hp.get(p2_n, 1.0), c_p2_hp)

            p1_status[p1_n] = str(r['p1_pokemon_state.status']).lower().strip()
            p2_status[p2_n] = str(r['p2_pokemon_state.status']).lower().strip()

            m1 = str(r.get('p1_move_details.name') or '').lower().strip()
            m2 = str(r.get('p2_move_details.name') or '').lower().strip()

            if m1 in ['', 'nan', 'none']: p1_missed_or_no_move += 1
            if m2 in ['', 'nan', 'none']: p2_missed_or_no_move += 1

            if m1 == 'spikes': p1_spikes_used += 1
            if m2 == 'spikes': p2_spikes_used += 1
            if m1 in meta_kb.setup_and_support_moves: p1_setup_moves_used += 1
            if m2 in meta_kb.setup_and_support_moves: p2_setup_moves_used += 1
            if m1 in RECOVERY_MOVES: p1_recovery_used += 1
            if m2 in RECOVERY_MOVES: p2_recovery_used += 1
            if m1 in PHAZING_MOVES: p1_phaze_used += 1
            if m2 in PHAZING_MOVES: p2_phaze_used += 1
            if m1 in {'rest', 'sleeptalk'}: p1_has_rest_talk[p1_n] = True
            if m2 in {'rest', 'sleeptalk'}: p2_has_rest_talk[p2_n] = True

            if t in [10, 20, 30]:
                p1_tot_t = sum(p1_hp.values())
                p2_unrev_t = max(0, 6 - len(p2_hp))
                p2_tot_t = sum(p2_hp.values()) + p2_unrev_t * 1.0
                if t == 10: hp_diff_turn_10 = p1_tot_t - p2_tot_t
                elif t == 20: hp_diff_turn_20 = p1_tot_t - p2_tot_t
                elif t == 30: hp_diff_turn_30 = p1_tot_t - p2_tot_t

        # 1. Survival, Fainted & Weighted Status Severity
        p1_fainted = sum(1 for name, hp in p1_hp.items() if hp <= 0.0 or p1_status[name] == 'fnt')
        p2_fainted = sum(1 for name, hp in p2_hp.items() if hp <= 0.0 or p2_status[name] == 'fnt')
        p1_unrevealed = max(0, 6 - len(p1_revealed))
        p2_unrevealed = max(0, 6 - len(p2_hp))

        p1_total_hp = sum(p1_hp.values())
        p2_total_hp = sum(p2_hp.values()) + p2_unrevealed * 1.0

        # Calculate smart status severity penalty on ALIVE Pokemon
        p1_status_penalty, p2_status_penalty = 0.0, 0.0
        p1_statused_alive, p2_statused_alive = 0, 0
        for n, s in p1_status.items():
            if p1_hp[n] > 0 and s not in ['nostatus', 'fnt']:
                p1_statused_alive += 1
                if s == 'slp' and p1_has_rest_talk[n]:
                    p1_status_penalty += 0.15  # RestTalk sleep is much less punishing!
                else:
                    p1_status_penalty += STATUS_SEVERITY.get(s, 0.3)

        for n, s in p2_status.items():
            if p2_hp[n] > 0 and s not in ['nostatus', 'fnt']:
                p2_statused_alive += 1
                if s == 'slp' and p2_has_rest_talk[n]:
                    p2_status_penalty += 0.15
                else:
                    p2_status_penalty += STATUS_SEVERITY.get(s, 0.3)

        # Effective Health Pool (HP adjusted for status crippling!)
        p1_effective_hp = p1_total_hp - 0.5 * p1_status_penalty
        p2_effective_hp = p2_total_hp - 0.5 * p2_status_penalty

        # Snorlax & Top OU Tier Tracking
        p1_snorlax_hp = p1_hp.get('snorlax', 0.0)
        p2_snorlax_hp = p2_hp.get('snorlax', 0.822 * 1.0 if p2_unrevealed > 0 else 0.0)

        # 2. Best Bench Counter-Matchup Score (Min-Max Competitive Coverage!)
        # For each alive revealed P2 Pokemon, what is P1's BEST counter score among P1's alive Pokemon?
        p1_alive_mons = [(n, hp) for n, hp in p1_hp.items() if hp > 0]
        p2_alive_mons = [(n, hp) for n, hp in p2_hp.items() if hp > 0]

        p1_best_counter_scores = []
        for p2_n, p2_h in p2_alive_mons:
            p2_t = meta_kb.pokedex.get(p2_n, {}).get('types', [])
            best_score = -5.0
            for p1_n, p1_h in p1_alive_mons:
                p1_t = roster[p1_n]['types']
                off_threat = estimate_gen2_duel_threat(meta_kb, p1_n, p1_t, p2_n, p2_t)
                def_threat = estimate_gen2_duel_threat(meta_kb, p2_n, p2_t, p1_n, p1_t)
                best_score = max(best_score, (off_threat - def_threat) * p1_h)
            if best_score > -5.0:
                p1_best_counter_scores.append(best_score)

        p1_counter_coverage_mean = float(np.mean(p1_best_counter_scores)) if p1_best_counter_scores else 0.0
        p1_counter_coverage_min = float(np.min(p1_best_counter_scores)) if p1_best_counter_scores else 0.0

        # Whole-Team Remaining Type Matrix Score
        p1_rem_type_vec = np.zeros(len(ALL_TYPES))
        for name, hp in p1_hp.items():
            if hp > 0:
                for t in roster[name]['types']:
                    if t in ALL_TYPES:
                        p1_rem_type_vec[ALL_TYPES.index(t)] += hp

        p2_rem_type_vec = np.zeros(len(ALL_TYPES))
        for name, hp in p2_hp.items():
            if hp > 0:
                for t in meta_kb.pokedex.get(name, {}).get('types', []):
                    if t in ALL_TYPES:
                        p2_rem_type_vec[ALL_TYPES.index(t)] += hp

        p2_hidden_probs = meta_kb.predict_remaining_team_type_probs(set(p2_hp.keys()))
        for idx, t in enumerate(ALL_TYPES):
            p2_rem_type_vec[idx] += (p2_unrevealed / 6.0) * p2_hidden_probs.get(t, 0.0)

        p1_team_type_offense = float(p1_rem_type_vec @ M_17x17 @ p2_rem_type_vec)
        p2_team_type_offense = float(p2_rem_type_vec @ M_17x17 @ p1_rem_type_vec)

        # Surviving Base Stat Totals (BST) & Speed Tiers
        p1_alive_bst = sum(roster[n]['bst'] * hp for n, hp in p1_hp.items() if hp > 0)
        p2_alive_bst = sum(
            sum(meta_kb.pokedex.get(n, {}).get(s, 85) for s in ['base_hp', 'base_atk', 'base_def', 'base_spa', 'base_spd', 'base_spe']) * hp
            for n, hp in p2_hp.items() if hp > 0
        ) + p2_unrevealed * 535.0  # 535 is average OU BST

        p1_max_alive_spe = max([roster[n]['base_spe'] for n, hp in p1_hp.items() if hp > 0] or [0.0])
        p2_max_alive_spe = max([meta_kb.pokedex.get(n, {}).get('base_spe', 85) for n, hp in p2_hp.items() if hp > 0] or [100.0])

        p1_alive_meta_score = sum(roster[n]['usage'] * hp for n, hp in p1_hp.items() if hp > 0)
        p2_alive_meta_score = sum(meta_kb.pokemon_usage_prob.get(n, 0.1) * hp for n, hp in p2_hp.items() if hp > 0) + p2_unrevealed * 0.25

        # Final turn (Turn 30) boosts & volatile effects (Substitute, Leech Seed, Screens)
        last_r = rows[-1]
        p1_final_boosts = sum(int(last_r.get(f'p1_pokemon_state.boosts.{s}', 0)) for s in ['atk', 'def', 'spa', 'spd', 'spe'])
        p2_final_boosts = sum(int(last_r.get(f'p2_pokemon_state.boosts.{s}', 0)) for s in ['atk', 'def', 'spa', 'spd', 'spe'])
        p1_final_off_boosts = int(last_r.get('p1_pokemon_state.boosts.atk', 0)) + int(last_r.get('p1_pokemon_state.boosts.spa', 0))
        p2_final_off_boosts = int(last_r.get('p2_pokemon_state.boosts.atk', 0)) + int(last_r.get('p2_pokemon_state.boosts.spa', 0))

        p1_eff_str = str(last_r.get('p1_pokemon_state.effects', '')).lower()
        p2_eff_str = str(last_r.get('p2_pokemon_state.effects', '')).lower()
        p1_has_sub = int('substitute' in p1_eff_str)
        p2_has_sub = int('substitute' in p2_eff_str)

        battle_rows.append({
            'battle_id': b_id,
            'p1_total_hp': p1_total_hp,
            'p2_total_hp': p2_total_hp,
            'total_hp_diff': p1_total_hp - p2_total_hp,
            'total_hp_ratio': (p1_total_hp + 0.1) / (p2_total_hp + 0.1),
            'effective_hp_diff': p1_effective_hp - p2_effective_hp,
            'p1_fainted': p1_fainted,
            'p2_fainted': p2_fainted,
            'fainted_diff': p2_fainted - p1_fainted,
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
            'p1_spikes_active': int(p1_spikes_used > 0),
            'p2_spikes_active': int(p2_spikes_used > 0),
            'spikes_chip_diff': p2_spikes_chip_taken - p1_spikes_chip_taken,
            'p1_setup_moves_used': p1_setup_moves_used,
            'p2_setup_moves_used': p2_setup_moves_used,
            'setup_moves_diff': p1_setup_moves_used - p2_setup_moves_used,
            'recovery_moves_diff': p1_recovery_used - p2_recovery_used,
            'phaze_moves_diff': p1_phaze_used - p2_phaze_used,
            'p1_switches': p1_switches,
            'p2_switches': p2_switches,
            'switch_diff': p2_switches - p1_switches,
            'late_switch_diff': p2_late_switches - p1_late_switches,
            'action_efficiency_diff': p2_missed_or_no_move - p1_missed_or_no_move,
            'hp_diff_turn_10': hp_diff_turn_10,
            'hp_diff_turn_20': hp_diff_turn_20,
            'hp_momentum_20_to_30': hp_diff_turn_30 - hp_diff_turn_20,
            'hp_momentum_10_to_30': hp_diff_turn_30 - hp_diff_turn_10,
            'p1_team_type_offense': p1_team_type_offense,
            'p2_team_type_offense': p2_team_type_offense,
            'rem_team_type_advantage': p1_team_type_offense - p2_team_type_offense,
            'p1_counter_coverage_mean': p1_counter_coverage_mean,
            'p1_counter_coverage_min': p1_counter_coverage_min,
            'alive_bst_diff': p1_alive_bst - p2_alive_bst,
            'max_alive_spe_diff': p1_max_alive_spe - p2_max_alive_spe,
            'alive_meta_score_diff': p1_alive_meta_score - p2_alive_meta_score,
            'final_boost_diff': p1_final_boosts - p2_final_boosts,
            'final_off_boost_diff': p1_final_off_boosts - p2_final_off_boosts,
            'sub_active_diff': p1_has_sub - p2_has_sub,
        })

    battle_df = pd.DataFrame(battle_rows)
    battle_df = battle_df.merge(turn_grp, on='battle_id', how='left')
    battle_df = battle_df.merge(late_turns, on='battle_id', how='left').fillna(0.0)
    battle_df = battle_df.merge(t31_df, on='battle_id', how='left')

    if 'player_won' in p2_lead_df.columns:
        battle_df = battle_df.merge(p2_lead_df[['battle_id', 'player_won']], on='battle_id', how='left')

    return battle_df

t31_features_df, TYPE_MATRIX_17x17 = compute_turn31_expected_type_features_v2(
    turn_df, meta_kb, p1_switch_model, p2_switch_model, p1_next_type_model, p2_next_type_model
)
battle_features_df = build_battle_level_features_v2(
    p1_team_df, timeline_df, p2_lead_df, turn_df, t31_features_df, meta_kb, TYPE_MATRIX_17x17
)
print(f"Upgraded V2 Battle-Level Dataset Shape: {battle_features_df.shape}")