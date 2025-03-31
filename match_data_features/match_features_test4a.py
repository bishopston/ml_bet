import pandas as pd
import numpy as np
from datetime import datetime

def calculate_match_features(home_team, away_team, data, match_date=None):
    """
    Calculate features for an upcoming match between two teams.
    Uses the latest 5 matches to determine form and rolling averages.
    """
    # Ensure Date column is in datetime format
    data['Date'] = pd.to_datetime(data['Date'])
    
    if match_date is None:
        match_date = datetime.now()

    # Filter the last 5 matches for each team
    home_team_matches = data[
        (data['HomeTeam'] == home_team) | (data['AwayTeam'] == home_team)
    ].sort_values(by='Date', ascending=False).head(5)

    away_team_matches = data[
        (data['HomeTeam'] == away_team) | (data['AwayTeam'] == away_team)
    ].sort_values(by='Date', ascending=False).head(5)

    # Initialize match features dictionary
    match_features = {
        'HomeTeam': home_team,
        'AwayTeam': away_team,
    }

    # Calculate form points (3 for win, 1 for draw, 0 for loss)
    def calculate_form(matches, team):
        form_points = 0
        for _, match in matches.iterrows():
            if match['HomeTeam'] == team:
                if match['FTR'] == 'H':
                    form_points += 3
                elif match['FTR'] == 'D':
                    form_points += 1
            else:
                if match['FTR'] == 'A':
                    form_points += 3
                elif match['FTR'] == 'D':
                    form_points += 1
        return form_points

    def calculate_shots_on_target(matches, team):
        shots_on_target = 0
        for _, match in matches.iterrows():
            if match['HomeTeam'] == team:
                shots_on_target += match['HST']
            else:
                shots_on_target += match['AST']
        return shots_on_target / 5

    def calculate_shots(matches, team):
        shots = 0
        for _, match in matches.iterrows():
            if match['HomeTeam'] == team:
                shots += match['HS']
            else:
                shots += match['AS']
        return shots / 5

    match_features['HomeTeam_Form'] = calculate_form(home_team_matches, home_team)
    match_features['AwayTeam_Form'] = calculate_form(away_team_matches, away_team)

    match_features['HST'] = calculate_shots_on_target(home_team_matches, home_team)
    match_features['AST'] = calculate_shots_on_target(away_team_matches, away_team)

    match_features['HS'] = calculate_shots(home_team_matches, home_team)
    match_features['AS'] = calculate_shots(away_team_matches, away_team)

    return pd.DataFrame([match_features])

# Example usage
def prepare_match_prediction(data, home_team, away_team, match_date=None):
    """
    Prepare features for predicting a specific match.
    Returns a DataFrame with the match features.
    """
    match_data = calculate_match_features(home_team, away_team, data, match_date)
    return match_data

# Example usage
data = pd.read_csv('../e0/E0_24_25.csv')
match_data = prepare_match_prediction(data, 'Newcastle', 'Aston Villa')

print("\nMatch Features:")
print(match_data)

# Save the features for later use
match_data.to_csv("/home/alexandros/ml_bet/match_data_features/match_data_features_4a.csv", index=None, sep='|')
