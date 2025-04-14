import pandas as pd

# Load data
results_df = pd.read_csv('../i1/24_25/I1_130425.csv')
features_df = pd.read_csv('match_features_flattened_prob.csv')

# --- STEP 1: Make sure your raw team names match the format in features_df ---
# If necessary, build a dictionary that maps raw names to standardized ones.
# For this example, let's assume raw team names are already in the correct format.
# If they’re not, you need to load a separate mapping dictionary and apply it here.

# --- STEP 2: Create 'mg_HomeTeam' and 'mg_AwayTeam' in results_df directly from 'HomeTeam' and 'AwayTeam'
# (assuming names already match those in features_df) ---
results_df['mg_HomeTeam'] = results_df['HomeTeam']
results_df['mg_AwayTeam'] = results_df['AwayTeam']

# --- STEP 3: Select columns from results_df to keep ---
columns_to_keep = [
    'mg_HomeTeam', 'mg_AwayTeam', 'Div', 'Date', 'Time', 'FTHG', 'FTAG', 'FTR',
    'HTHG', 'HTAG', 'HTR', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC',
    'HY', 'AY', 'HR', 'AR', 'B365H', 'B365D', 'B365A'
]

# --- STEP 4: Merge with features_df ---
merged_df = pd.merge(
    results_df[columns_to_keep],
    features_df,
    on=['mg_HomeTeam', 'mg_AwayTeam'],
    how='inner'
)

# --- STEP 5: Save the merged file ---
merged_df.to_csv('i1_merged_match_data.csv', index=False)
