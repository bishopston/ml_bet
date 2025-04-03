import pandas as pd
import numpy as np
from datetime import datetime

def compute_home_shots_rolling_avg(data, home_team, match_date=None, window=5):
    if match_date is None:
        match_date = datetime.now()
    home_matches = data[data['HomeTeam'] == home_team]
    home_matches = home_matches[home_matches['Date'] < match_date]
    recent_home_matches = home_matches.sort_values('Date', ascending=False).head(window)
    return float(recent_home_matches['HS'].mean()) if not recent_home_matches.empty else 0.0

def compute_away_shots_rolling_avg(data, away_team, match_date=None, window=5):
    if match_date is None:
        match_date = datetime.now()
    away_matches = data[data['AwayTeam'] == away_team]
    away_matches = away_matches[away_matches['Date'] < match_date]
    recent_away_matches = away_matches.sort_values('Date', ascending=False).head(window)
    return float(recent_away_matches['AS'].mean()) if not recent_away_matches.empty else 0.0

def compute_home_fouls_rolling_avg(data, home_team, match_date=None, window=5):
    if match_date is None:
        match_date = datetime.now()
    home_matches = data[data['HomeTeam'] == home_team]
    home_matches = home_matches[home_matches['Date'] < match_date]
    recent_home_matches = home_matches.sort_values('Date', ascending=False).head(window)
    return float(recent_home_matches['HF'].mean()) if not recent_home_matches.empty else 0.0

def compute_away_fouls_rolling_avg(data, away_team, match_date=None, window=5):
    if match_date is None:
        match_date = datetime.now()
    away_matches = data[data['AwayTeam'] == away_team]
    away_matches = away_matches[away_matches['Date'] < match_date]
    recent_away_matches = away_matches.sort_values('Date', ascending=False).head(window)
    return float(recent_away_matches['AF'].mean()) if not recent_away_matches.empty else 0.0

def calculate_match_features(home_team, away_team, data, match_date=None, window=5):
    """
    Calculate features for a match between two teams using rolling averages.
    """
    # Ensure data is a pandas DataFrame
    if not isinstance(data, pd.DataFrame):
        raise ValueError("data must be a pandas DataFrame")
    
    data['Date'] = pd.to_datetime(data['Date'])
    if match_date is None:
        match_date = datetime.now()
    
    match_features = {
        'home_team': home_team,
        'away_team': away_team,
        'features': {
            'HS': float(compute_home_shots_rolling_avg(data, home_team, match_date, window)),  # Convert to pure float
            'AS': float(compute_away_shots_rolling_avg(data, away_team, match_date, window)),  # Convert to pure float
            'HF': float(compute_home_fouls_rolling_avg(data, home_team, match_date, window)),  # Convert to pure float
            'AF': float(compute_away_fouls_rolling_avg(data, away_team, match_date, window)),  # Convert to pure float
        }
    }
    return match_features

def prepare_all_matches(data, upcoming_matches):
    """ Prepare match features for multiple upcoming matches. """
    match_features_list = [calculate_match_features(row['HomeTeam'], row['AwayTeam'], data) for _, row in upcoming_matches.iterrows()]
    return match_features_list

# Example usage
data = pd.read_csv('../e0/E0_24_25.csv')  # Make sure this is the correct path to your CSV
data['Date'] = pd.to_datetime(data['Date'])  # Ensure the 'Date' column is in datetime format

# List of upcoming matches
upcoming_matches = pd.DataFrame({
    'HomeTeam': ['Chelsea', 'Liverpool'],
    'AwayTeam': ['Tottenham', 'Man City']
})

# Get match features for multiple upcoming matches
match_features = prepare_all_matches(data, upcoming_matches)

# Print match features
print(match_features)
