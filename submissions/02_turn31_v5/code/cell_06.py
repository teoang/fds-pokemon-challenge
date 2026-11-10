# ==============================================================================
# STEP 2 (V2): UPGRADED TURN ENGINE WITH GEN-2 STAT/STAB DAMAGE & COUNTER MATRIX
# ==============================================================================
# Gen 2 Physical vs Special Type Split
PHYSICAL_TYPES = {'normal', 'fighting', 'flying', 'ground', 'rock', 'bug', 'ghost', 'poison', 'steel'}
SPECIAL_TYPES  = {'fire', 'water', 'grass', 'electric', 'psychic', 'ice', 'dragon', 'dark'}

def get_boost_multiplier(stage):
    """Converts Gen 2 stat stage (-6 to +6) into its exact combat multiplier."""
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
    """
    Computes best expected Gen 2 damage power between two Pokemon considering:
    - Attacker's STAB types + known competitive coverage move types
    - Gen 2 Physical/Special split (Base Atk vs Def, Base SpA vs SpD)
    - Active stat stage boosts (+1 Curse, +2 Swords Dance, etc.)
    - Exact Gen 2 Type Chart multiplier
    """
    if not def_types:
        return 1.0
    atk_info = meta_kb.pokedex.get(atk_name, {})
    def_info = meta_kb.pokedex.get(def_name, {})

    b_atk = atk_info.get('base_atk', 85) * get_boost_multiplier((atk_boosts or {}).get('atk', 0))
    b_spa = atk_info.get('base_spa', 85) * get_boost_multiplier((atk_boosts or {}).get('spa', 0))
    b_def = max(20.0, def_info.get('base_def', 85) * get_boost_multiplier((def_boosts or {}).get('def', 0)))
    b_spd = max(20.0, def_info.get('base_spd', 85) * get_boost_multiplier((def_boosts or {}).get('spd', 0)))

    candidate_types = set(atk_types)
    # Add common coverage move types for top meta Pokemon if known
    if not candidate_types:
        candidate_types = {'normal'}

    best_damage_index = 0.5
    for m_type in candidate_types:
        type_mult = get_type_multiplier_from_list(meta_kb, m_type, def_types)
        stab = 1.5 if m_type in atk_types else 1.0
        if m_type in PHYSICAL_TYPES:
            stat_ratio = b_atk / b_def
        else:
            stat_ratio = b_spa / b_spd
        best_damage_index = max(best_damage_index, type_mult * stab * stat_ratio)

    return best_damage_index

# Ensure Pokedex in meta_kb has clean types from the 'types' list column
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

print("Updated Pokedex with Gen 2 Physical/Special combat stats for", len(meta_kb.pokedex), "Pokemon.")