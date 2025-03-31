import pandas as pd
import numpy as np
from datetime import datetime

def calculate_match_features(home_team, away_team, data, match_date=None):
    """
    Calculate features for an upcoming match between two teams.
    Uses the latest 5 matches to determine form and rolling averages.
    """
    data['Date'] = pd.to_datetime(data['Date'])
    
    if match_date is None:
        match_date = datetime.now()

    # Filter last 5 matches for each team
    home_team_matches = data[
        ((data['HomeTeam'] == home_team) | (data['AwayTeam'] == home_team)) & (data['Date'] < match_date)
    ].sort_values(by='Date', ascending=False).head(5)

    away_team_matches = data[
        ((data['HomeTeam'] == away_team) | (data['AwayTeam'] == away_team)) & (data['Date'] < match_date)
    ].sort_values(by='Date', ascending=False).head(5)

    # Separate home and away matches for rolling average calculation
    home_team_home_matches = home_team_matches[home_team_matches['HomeTeam'] == home_team]
    home_team_away_matches = home_team_matches[home_team_matches['AwayTeam'] == home_team]
    
    away_team_home_matches = away_team_matches[away_team_matches['HomeTeam'] == away_team]
    away_team_away_matches = away_team_matches[away_team_matches['AwayTeam'] == away_team]

    match_features = {
        'HomeTeam': home_team,
        'AwayTeam': away_team,
    }

    # Compute rolling averages **only for relevant matches** (home for home team, away for away team)
    def rolling_avg(matches, column):
        return matches[column].mean() if not matches.empty else 0

    match_features.update({
        'RollingAvg_HomeHST': rolling_avg(home_team_home_matches, 'HST'),
        'RollingAvg_AwayAST': rolling_avg(away_team_away_matches, 'AST'),
        'RollingAvg_HomeHC': rolling_avg(home_team_home_matches, 'HC'),
        'RollingAvg_AwayAC': rolling_avg(away_team_away_matches, 'AC'),
        'RollingAvg_HomeHF': rolling_avg(home_team_home_matches, 'HF'),
        'RollingAvg_AwayAF': rolling_avg(away_team_away_matches, 'AF'),
        'RollingAvg_FTHG': rolling_avg(home_team_home_matches, 'FTHG'),
        'RollingAvg_FTAG': rolling_avg(away_team_away_matches, 'FTAG'),
    })

    # Calculate total rolling stats (sum of both teams' averages)
    match_features['RollingAvg_TotalShots'] = match_features['RollingAvg_HomeHST'] + match_features['RollingAvg_AwayAST']
    match_features['RollingAvg_TotalCorners'] = match_features['RollingAvg_HomeHC'] + match_features['RollingAvg_AwayAC']

    # Calculate form points (3 for win, 1 for draw, 0 for loss)
    def calculate_form(matches, team):
        form_points = 0
        for _, match in matches.iterrows():
            if match['HomeTeam'] == team:
                form_points += 3 if match['FTR'] == 'H' else 1 if match['FTR'] == 'D' else 0
            else:
                form_points += 3 if match['FTR'] == 'A' else 1 if match['FTR'] == 'D' else 0
        return form_points

    match_features['HomeTeam_Form'] = calculate_form(home_team_matches, home_team)
    match_features['AwayTeam_Form'] = calculate_form(away_team_matches, away_team)

    return pd.DataFrame([match_features])

# Example usage
def prepare_match_prediction(data, home_team, away_team, match_date=None):
    """
    Prepare features for predicting a specific match.
    Returns a DataFrame with the match features.
    """
    match_data = calculate_match_features(home_team, away_team, data, match_date)
    return match_data

# Load dataset and compute match features
data = pd.read_csv('e0/E0_24_25.csv')
match_data = prepare_match_prediction(data, 'Newcastle', 'Aston Villa')

print("\nMatch Features:")
print(match_data)

# Save the features for later use
match_data.to_csv("/home/alexandros/ml_bet/match_data_features_3.csv", index=None, sep='|')
