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

    # Filter the last x matches for each team for form calculation
    home_team_matches_form = data[
        (data['HomeTeam'] == home_team) | (data['AwayTeam'] == home_team)
    ].sort_values(by='Date', ascending=False).head(window_form)

    away_team_matches_form = data[
        (data['HomeTeam'] == away_team) | (data['AwayTeam'] == away_team)
    ].sort_values(by='Date', ascending=False).head(window_form)

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

    match_features['HomeTeam_Form'] = calculate_form(home_team_matches_form, home_team)
    match_features['AwayTeam_Form'] = calculate_form(away_team_matches_form, away_team)

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

def calculate_league_avg_shots(data):
    """
    Calculate the league-wide average total shots per match.
    """
    data['TotalShots'] = data['HS'] + data['AS']  # Total shots per match
    league_avg_shots = data['TotalShots'].mean()  # League-wide average
    return round(league_avg_shots, 2)

def calculate_league_avg_home_shots(data):
    """
    Calculate the league-wide average shots per match for home teams.
    """
    league_avg_HS = data['HS'].mean()  # Average home team shots per match
    return round(league_avg_HS, 2)

def calculate_league_avg_away_shots(data):
    """
    Calculate the league-wide average shots per match for away teams.
    """
    league_avg_AS = data['AS'].mean()  # Average away team shots per match
    return round(league_avg_AS, 2)

def calculate_league_avg_home_shots_on_target(data):

    league_avg_HST = data['HST'].mean()  
    return round(league_avg_HST, 2)

def calculate_league_avg_away_shots_on_target(data):

    league_avg_AST = data['AST'].mean()  
    return round(league_avg_AST, 2)

def calculate_league_avg_home_fouls(data):

    league_avg_HF = data['HF'].mean()  
    return round(league_avg_HF, 2)

def calculate_league_avg_away_fouls(data):

    league_avg_AF = data['AF'].mean()  
    return round(league_avg_AF, 2)

def calculate_league_avg_shots(data):
    """
    Calculate the league-wide average total shots per match.
    """
    data['TotalShots'] = data['HS'] + data['AS']  # Total shots per match
    league_avg_shots = data['TotalShots'].mean()  # League-wide average
    return round(league_avg_shots, 2)

# Compute league-wide average total shots
# league_avg_total_shots = calculate_league_avg_shots(data)

# print(f"League-Wide Average Total Shots per Match: {league_avg_total_shots}")

match_data = prepare_match_prediction(data, 'Newcastle', 'Aston Villa')

match_data['HS'] = calculate_league_avg_home_shots(data)
match_data['AS'] = calculate_league_avg_away_shots(data)
match_data['HST'] = calculate_league_avg_home_shots_on_target(data)
match_data['AST'] = calculate_league_avg_away_shots_on_target(data)
match_data['HF'] = calculate_league_avg_home_fouls(data)
match_data['AF'] = calculate_league_avg_away_fouls(data)
match_data['TotalShots'] = calculate_league_avg_shots(data)

print("\nMatch Features:")
print(match_data)

# Save the features for later use
match_data.to_csv("/home/alexandros/ml_bet/match_data_features/match_data_features_4c.csv", index=None, sep='|')
