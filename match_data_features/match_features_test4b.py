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

    window_form = 2
    window = 5
    window_total = 0

    # Filter the last x matches for each team for form calculation
    home_team_matches_form = data[
        (data['HomeTeam'] == home_team) | (data['AwayTeam'] == home_team)
    ].sort_values(by='Date', ascending=False).head(window_form)

    away_team_matches_form = data[
        (data['HomeTeam'] == away_team) | (data['AwayTeam'] == away_team)
    ].sort_values(by='Date', ascending=False).head(window_form)

    # Filter the last x matches for each team for rest features
    home_team_matches = data[
        (data['HomeTeam'] == home_team) | (data['AwayTeam'] == home_team)
    ].sort_values(by='Date', ascending=False).head(window)

    away_team_matches = data[
        (data['HomeTeam'] == away_team) | (data['AwayTeam'] == away_team)
    ].sort_values(by='Date', ascending=False).head(window)

    # Get all matches for each team for rest features
    home_team_matches_total = data[
        (data['HomeTeam'] == home_team) | (data['AwayTeam'] == home_team)
    ].sort_values(by='Date', ascending=False)

    print(len(home_team_matches_total))

    away_team_matches_total = data[
        (data['HomeTeam'] == away_team) | (data['AwayTeam'] == away_team)
    ].sort_values(by='Date', ascending=False)

    print(len(away_team_matches_total))

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

    def calculate_goals(matches, team):
        goals = 0
        for _, match in matches.iterrows():
            if match['HomeTeam'] == team:
                goals += match['FTHG']
            else:
                goals += match['FTAG']
        return goals / window

    def calculate_shots_on_target(matches, team):
        shots_on_target = 0
        for _, match in matches.iterrows():
            if match['HomeTeam'] == team:
                shots_on_target += match['HST']
            else:
                shots_on_target += match['AST']
        return shots_on_target / window

    def calculate_shots(matches, team):
        shots = 0
        for _, match in matches.iterrows():
            if match['HomeTeam'] == team:
                shots += match['HS']
                window_total = len(home_team_matches_total)
            else:
                shots += match['AS']
                window_total = len(away_team_matches_total)
        return shots / window_total

    def calculate_fouls(matches, team):
        fouls = 0
        for _, match in matches.iterrows():
            if match['HomeTeam'] == team:
                fouls += match['HF']
                window_total = len(home_team_matches_total)
            else:
                fouls += match['AF']
                window_total = len(away_team_matches_total)
        return fouls / window_total

    def calculate_corners(matches, team):
        corners = 0
        for _, match in matches.iterrows():
            if match['HomeTeam'] == team:
                corners += match['HC']
                window_total = len(home_team_matches_total)
            else:
                corners += match['AC']
                window_total = len(away_team_matches_total)
        return corners / window_total
    
    def calculate_yellows(matches, team):
        yellows = 0
        for _, match in matches.iterrows():
            if match['HomeTeam'] == team:
                yellows += match['HY']
                window_total = len(home_team_matches_total)
            else:
                yellows += match['AY']
                window_total = len(away_team_matches_total)
        return yellows / window_total

    def calculate_reds(matches, team):
        reds = 0
        for _, match in matches.iterrows():
            if match['HomeTeam'] == team:
                reds += match['HR']
                window_total = len(home_team_matches_total)
            else:
                reds += match['AR']
                window_total = len(away_team_matches_total)
        return reds / window_total

    def calculate_shots_on_target_total(matches, team):
        shots_on_target_total = 0
        for _, match in matches.iterrows():
            if match['HomeTeam'] == team:
                shots_on_target_total += match['HST']
                window_total = len(home_team_matches_total)
            else:
                shots_on_target_total += match['AST']
                window_total = len(away_team_matches_total)
        return shots_on_target_total / window_total

    def calculate_total_shots(matches, team):
        shots_on_target_total = 0
        for _, match in matches.iterrows():
            if match['HomeTeam'] == team:
                shots_on_target_total += match['HS']
                shots_on_target_total += match['AS']
                window_total = len(home_team_matches_total)
            else:
                shots_on_target_total += match['AS']
                shots_on_target_total += match['HS']
                window_total = len(away_team_matches_total)
        return shots_on_target_total / window_total

    match_features['HomeTeam_Form'] = calculate_form(home_team_matches_form, home_team)
    match_features['AwayTeam_Form'] = calculate_form(away_team_matches_form, away_team)

    match_features['RollingAvg_FTHG'] = calculate_goals(home_team_matches, home_team)
    match_features['RollingAvg_FTAG'] = calculate_goals(away_team_matches, away_team)

    match_features['RollingAvg_HomeHST'] = calculate_shots_on_target(home_team_matches, home_team)
    match_features['RollingAvg_AwayAST'] = calculate_shots_on_target(away_team_matches, away_team)

    match_features['HS'] = calculate_shots(home_team_matches_total, home_team)
    match_features['AS'] = calculate_shots(away_team_matches_total, away_team)

    match_features['HF'] = calculate_fouls(home_team_matches_total, home_team)
    match_features['AF'] = calculate_fouls(away_team_matches_total, away_team)

    match_features['HC'] = calculate_corners(home_team_matches_total, home_team)
    match_features['AC'] = calculate_corners(away_team_matches_total, away_team)

    match_features['HY'] = calculate_yellows(home_team_matches_total, home_team)
    match_features['AY'] = calculate_yellows(away_team_matches_total, away_team)

    match_features['HR'] = calculate_reds(home_team_matches_total, home_team)
    match_features['AR'] = calculate_reds(away_team_matches_total, away_team)

    match_features['HST'] = calculate_shots_on_target_total(home_team_matches_total, home_team)
    match_features['AST'] = calculate_shots_on_target_total(away_team_matches_total, away_team)

    #match_features['TotalShots'] = calculate_total_shots(home_team_matches_total, home_team)

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
match_data.to_csv("/home/alexandros/ml_bet/match_data_features/match_data_features_4b.csv", index=None, sep='|')
