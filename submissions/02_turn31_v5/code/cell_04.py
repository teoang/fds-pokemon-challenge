# ==============================================================================
# STEP 1C: BUILD TYPE EFFECTIVENESS ENGINE & METAGAME KNOWLEDGE BASE
# ==============================================================================
class CompetitiveMetaKnowledgeBase:
    """
    Learns strictly from training data (no external data leakage!):
    1. Exact Gen 2 Type Chart multipliers (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
    2. Internal Pokedex (maps Pokemon name -> types & base stats)
    3. Competitive Usage Priors (most used Pokemon & types in the meta)
    4. Team Type Co-occurrence Matrix (predicts remaining hidden types on P2's team)
    5. Competitive Move & Role Profiler (identifies Setup/Hazard/Status Pokemon & standard moves)
    """
    def __init__(self, type_chart_df):
        # 1. Build fast O(1) Type Multiplier Lookup
        self.type_matrix = defaultdict(lambda: defaultdict(lambda: 1.0))
        for _, row in type_chart_df.iterrows():
            atk = str(row['attacking_type']).lower().strip()
            dfd = str(row['defending_type']).lower().strip()
            self.type_matrix[atk][dfd] = float(row['multiplier'])

        self.pokedex = {}
        self.pokemon_usage_prob = {}
        self.type_usage_prob = {}
        self.type_cooccurrence = pd.DataFrame(0.0, index=ALL_TYPES, columns=ALL_TYPES)
        self.pokemon_moves_profile = defaultdict(Counter)
        self.pokemon_setup_ratio = {}

        # Define Competitive Setup, Hazard, Phazing & Recovery Moves in Gen 2 OU
        self.setup_and_support_moves = {
            'curse', 'swordsdance', 'bellydrum', 'amnesia', 'growth', 'agility', 'meditate',
            'spikes', 'toxic', 'thunderwave', 'stunspore', 'lovelykiss', 'sleeppowder',
            'hypnosis', 'spore', 'confuseray', 'attract', 'leechseed',
            'roar', 'whirlwind', 'haze', 'perishsong',
            'rest', 'sleeptalk', 'recover', 'milkdrink', 'softboiled', 'moonlight', 'synthesis',
            'reflect', 'lightscreen', 'safeguard', 'substitute', 'batonpass', 'encore', 'rapidspin'
        }

    def get_move_effectiveness(self, move_type, def_type_1, def_type_2=None):
        """Calculates exact Gen 2 damage multiplier (e.g., 0.0, 0.25, 0.5, 1.0, 2.0, 4.0)."""
        if pd.isna(move_type) or pd.isna(def_type_1):
            return 1.0
        m_type = str(move_type).lower().strip()
        t1 = str(def_type_1).lower().strip()
        mult = self.type_matrix[m_type][t1]

        if pd.notna(def_type_2) and str(def_type_2).lower().strip() not in ['nan', 'none', '']:
            t2 = str(def_type_2).lower().strip()
            mult *= self.type_matrix[m_type][t2]
        return mult

    def fit(self, p1_team_df, timeline_df, p2_lead_df):
        # A. Build Internal Pokedex from P1 full teams and P2 leads
        combined_pkmn = pd.concat([
            p1_team_df[['name', 'types', 'base_hp', 'base_atk', 'base_def', 'base_spa', 'base_spd', 'base_spe']],
            p2_lead_df[['name', 'types', 'base_hp', 'base_atk', 'base_def', 'base_spa', 'base_spd', 'base_spe']]
        ]).drop_duplicates(subset=['name'])

        for _, row in combined_pkmn.iterrows():
            raw_types = row['types']
            if isinstance(raw_types, str):
                import ast
                raw_types = ast.literal_eval(raw_types)

            # Filter out placeholder types like 'notype'
            clean_types = [t.lower().strip() for t in raw_types if t and str(t).lower().strip() not in ['notype', 'nan', 'none', '']]
            t1 = clean_types[0] if len(clean_types) > 0 else 'normal'
            t2 = clean_types[1] if len(clean_types) > 1 else None

            self.pokedex[row['name']] = {
                'type_1': t1,
                'type_2': t2,
                'types': clean_types,
                'base_hp': row['base_hp'],
                'base_atk': row['base_atk'],
                'base_def': row['base_def'],
                'base_spa': row['base_spa'],
                'base_spd': row['base_spd'],
                'base_spe': row['base_spe']
            }

        # B. Calculate Metagame Pokemon & Type Usage Frequencies (from P1 6-Pokemon teams)
        total_teams = p1_team_df['battle_id'].nunique()
        pkmn_counts = p1_team_df['name'].value_counts()
        self.pokemon_usage_prob = (pkmn_counts / total_teams).to_dict()

        type_counts = Counter()
        for _, grp in p1_team_df.groupby('battle_id'):
            team_types = set()
            for _, row in grp.iterrows():
                raw_types = row['types']
                if isinstance(raw_types, str):
                    import ast
                    raw_types = ast.literal_eval(raw_types)
                for t in raw_types:
                    t_clean = str(t).lower().strip()
                    if t_clean not in ['notype', 'nan', 'none', '']:
                        team_types.add(t_clean)
            for t in team_types:
                type_counts[t] += 1

            # C. Populate Team Type Co-occurrence Matrix
            for t_a in team_types:
                for t_b in team_types:
                    if t_a in ALL_TYPES and t_b in ALL_TYPES:
                        self.type_cooccurrence.loc[t_a, t_b] += 1.0

        self.type_usage_prob = {t: type_counts[t] / total_teams for t in ALL_TYPES}

        # Normalize Co-occurrence Matrix into Conditional Probabilities P(Type B in team | Type A in team)
        for t_a in ALL_TYPES:
            if type_counts[t_a] > 0:
                self.type_cooccurrence.loc[t_a, :] /= type_counts[t_a]

        # D. Profile Competitive Movesets & Setup vs. Offensive Roles from Timeline
        p1_moves = timeline_df[['p1_pokemon_state.name', 'p1_move_details.name', 'p1_move_details.type', 'p1_move_details.category']].dropna(subset=['p1_move_details.name'])
        p1_moves.columns = ['pokemon', 'move', 'move_type', 'category']

        p2_moves = timeline_df[['p2_pokemon_state.name', 'p2_move_details.name', 'p2_move_details.type', 'p2_move_details.category']].dropna(subset=['p2_move_details.name'])
        p2_moves.columns = ['pokemon', 'move', 'move_type', 'category']

        all_moves = pd.concat([p1_moves, p2_moves])

        setup_counts = Counter()
        total_move_counts = Counter()

        for _, row in all_moves.iterrows():
            pkmn = row['pokemon']
            mv = str(row['move']).lower().strip()
            self.pokemon_moves_profile[pkmn][mv] += 1
            total_move_counts[pkmn] += 1
            if mv in self.setup_and_support_moves or str(row['category']).lower() == 'status':
                setup_counts[pkmn] += 1

        for pkmn in total_move_counts:
            self.pokemon_setup_ratio[pkmn] = setup_counts[pkmn] / total_move_counts[pkmn]

        return self

    def predict_remaining_team_type_probs(self, revealed_pokemon_names):
        """
        Given the Pokemon revealed so far by a player, predicts the probability distribution
        of the remaining hidden types on their bench using competitive team co-occurrence.
        """
        revealed_types = set()
        for name in revealed_pokemon_names:
            if name in self.pokedex:
                revealed_types.update(self.pokedex[name]['types'])

        if not revealed_types:
            return self.type_usage_prob.copy()

        # Average conditional probability P(Target Type | Revealed Types)
        cond_probs = self.type_cooccurrence.loc[list(revealed_types), :].mean(axis=0).to_dict()
        return cond_probs

# Instantiate and fit our Knowledge Base on the dataset
meta_kb = CompetitiveMetaKnowledgeBase(type_chart_df).fit(p1_team_df, timeline_df, p2_lead_df)

# Sanity Check Output
print("=== TOP 10 MOST USED COMPETITIVE POKEMON (P(In Team)) ===")
for pkmn, prob in list(meta_kb.pokemon_usage_prob.items())[:10]:
    setup_pct = meta_kb.pokemon_setup_ratio.get(pkmn, 0.0) * 100
    top_moves = [m for m, _ in meta_kb.pokemon_moves_profile[pkmn].most_common(4)]
    print(f"- {pkmn.capitalize():<12} | Team Usage: {prob*100:5.1f}% | Setup/Support Role: {setup_pct:4.1f}% | Standard Moves: {top_moves}")

print("\n=== TOP 8 MOST USED TYPES ACROSS TEAMS ===")
for t, prob in sorted(meta_kb.type_usage_prob.items(), key=lambda x: x[1], reverse=True)[:8]:
    print(f"- {t.capitalize():<10}: present in {prob*100:5.1f}% of competitive teams")