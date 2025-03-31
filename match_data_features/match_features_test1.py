import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def calculate_match_features(home_team, away_team, data, match_date=None):
    """
    Calculate features for an upcoming match between two teams.
    Uses historical data to generate realistic feature values.
    """
    # Ensure Date column is in datetime format
    data['Date'] = pd.to_datetime(data['Date'])
    
    if match_date is None:
        match_date = datetime.now()
    
    # Initialize match features dictionary
    match_features = {
        'HomeTeam': home_team,
        'AwayTeam': away_team,
        'HS': 12,      # Average home team shots per match
        'AS': 9,       # Average away team shots per match
        'HST': 4,      # Average home team shots on target
        'AST': 3,      # Average away team shots on target
        'HF': 12,      # Average home team fouls
        'AF': 11,      # Average away team fouls
        'HC': 5,       # Average home team corners
        'AC': 4,       # Average away team corners
        'HY': 1,       # Average home team yellow cards
        'AY': 1,       # Average away team yellow cards
        'HR': 0,       # Average home team red cards
        'AR': 0,       # Average away team red cards
        'B365H': 2.50, 'B365D': 3.30, 'B365A': 2.70,  # Typical bookmaker odds
        'TotalShots': 21,  # Average total shots per match
        'TotalCorners': 9,   # Average total corners per match
        'RollingAvg_FTHG': 1.5,  # Average home team goals
        'RollingAvg_FTAG': 1.0,  # Average away team goals conceded
        'RollingAvg_TotalShots': 21.0,
        'RollingAvg_TotalCorners': 9.0,
        'RollingAvg_HomeHST': 5,
        'RollingAvg_AwayAST': 6,
        'HomeTeam_Form': 4,
        'AwayTeam_Form': 4
    }
    
    # Calculate recent form (last 5 matches)
    window = 5
    cutoff_date = match_date - timedelta(days=window * 7)  # Assuming one match per week
    
    # Get recent matches for both teams
    home_team_matches = data[
        (data['HomeTeam'] == home_team) | 
        (data['AwayTeam'] == home_team)
    ]
    away_team_matches = data[
        (data['HomeTeam'] == away_team) | 
        (data['AwayTeam'] == away_team)
    ]
    
    # Calculate form points (3 for win, 1 for draw, 0 for loss)
    home_form = 0
    away_form = 0
    
    # Calculate home team form
    for _, match in home_team_matches.iterrows():
        if match['Date'] >= cutoff_date:
            if match['HomeTeam'] == home_team:
                if match['FTR'] == 'H':
                    home_form += 3
                elif match['FTR'] == 'D':
                    home_form += 1
            else:
                if match['FTR'] == 'A':
                    home_form += 3
                elif match['FTR'] == 'D':
                    home_form += 1
    
    # Calculate away team form
    for _, match in away_team_matches.iterrows():
        if match['Date'] >= cutoff_date:
            if match['AwayTeam'] == away_team:
                if match['FTR'] == 'A':
                    away_form += 3
                elif match['FTR'] == 'D':
                    away_form += 1
            else:
                if match['FTR'] == 'H':
                    away_form += 3
                elif match['FTR'] == 'D':
                    away_form += 1
    
    # Update form features
    match_features['HomeTeam_Form'] = home_form
    match_features['AwayTeam_Form'] = away_form
    
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
data = pd.read_csv('e0/E0_24_25.csv')
match_data = prepare_match_prediction(data, 'Newcastle', 'Aston Villa')
print("\nMatch Features:")
print(match_data)

match_data.to_csv("/home/alexandros/ml_bet/match_data_features.csv", index=None, sep='|')