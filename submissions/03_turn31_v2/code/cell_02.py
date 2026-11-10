# ==============================================================================
# STEP 1B: DATA LOADER (Supports both unpacked Excel and raw JSONL)
# ==============================================================================
def load_competition_data():
    """
    Loads p1_team_details, battle_timeline, and p2_lead_details from either
    train.jsonl (if available) or pokemon_json_spacchettato.xlsx.
    """
    if os.path.exists(EXCEL_UNPACKED_PATH) and not os.path.exists(TRAIN_JSONL_PATH):
        print(f"Loading unpacked data from {EXCEL_UNPACKED_PATH}...")
        xls = pd.ExcelFile(EXCEL_UNPACKED_PATH)
        p1_team_df = pd.read_excel(xls, 'p1_team_details')
        timeline_df = pd.read_excel(xls, 'battle_timeline')
        p2_lead_df = pd.read_excel(xls, 'p2_lead_details')
        return p1_team_df, timeline_df, p2_lead_df

    elif os.path.exists(TRAIN_JSONL_PATH):
        print(f"Loading raw JSONL data from {TRAIN_JSONL_PATH}...")
        p1_rows, timeline_rows, p2_rows = [], [], []
        with open(TRAIN_JSONL_PATH, 'r') as f:
            for line in f:
                b = json.loads(line)
                b_id = b['battle_id']

                # P2 Lead & Target
                p2 = b['p2_lead_details'].copy()
                p2['battle_id'] = b_id
                p2['player_won'] = int(b['player_won'])
                p2_rows.append(p2)

                # P1 Team (6 Pokemon)
                for pkmn in b['p1_team_details']:
                    p1 = pkmn.copy()
                    p1['battle_id'] = b_id
                    p1_rows.append(p1)

                # Timeline (Up to 30 turns)
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
    else:
        raise FileNotFoundError("Could not find pokemon_json_spacchettato.xlsx or train.jsonl!")

p1_team_df, timeline_df, p2_lead_df = load_competition_data()
type_chart_df = pd.read_csv(TYPE_CHART_PATH)

print(f"Loaded {p2_lead_df['battle_id'].nunique()} battles.")
print(f"P1 Team rows: {len(p1_team_df)} | Timeline rows: {len(timeline_df)} | P2 Lead rows: {len(p2_lead_df)}")