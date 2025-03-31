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
    ].sort_values(by='Date', ascending=False).head(2)

    away_team_matches = data[
        (data['HomeTeam'] == away_team) | (data['AwayTeam'] == away_team)
    ].sort_values(by='Date', ascending=False).head(2)

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

def compute_home_shots_rolling_avg(data, home_team, match_date=None, window=5):

    if match_date is None:
        match_date = datetime.now()

    # Filter matches where the team played at home
    home_matches = data[data['HomeTeam'] == home_team]

    # Filter only matches before the match date
    home_matches = home_matches[home_matches['Date'] < match_date]

    # Sort matches by date (most recent first) and take the last 'window' matches
    recent_home_matches = home_matches.sort_values('Date', ascending=False).head(window)

    # Compute rolling average of home shots (HS)
    return recent_home_matches['HS'].mean() if not recent_home_matches.empty else 0

def compute_away_shots_rolling_avg(data, away_team, match_date=None, window=5):

    if match_date is None:
        match_date = datetime.now()

    # Filter matches where the team played away
    away_matches = data[data['AwayTeam'] == away_team]

    # Filter only matches before the match date
    away_matches = away_matches[away_matches['Date'] < match_date]

    # Sort matches by date (most recent first) and take the last 'window' matches
    recent_away_matches = away_matches.sort_values('Date', ascending=False).head(window)

    # Compute rolling average of away shots (AS)
    return recent_away_matches['AS'].mean() if not recent_away_matches.empty else 0

def compute_home_fouls_rolling_avg(data, home_team, match_date=None, window=5):

    if match_date is None:
        match_date = datetime.now()

    # Filter matches where the team played at home
    home_matches = data[data['HomeTeam'] == home_team]

    # Filter only matches before the match date
    home_matches = home_matches[home_matches['Date'] < match_date]

    # Sort matches by date (most recent first) and take the last 'window' matches
    recent_home_matches = home_matches.sort_values('Date', ascending=False).head(window)

    # Compute rolling average of home shots (HS)
    return recent_home_matches['HF'].mean() if not recent_home_matches.empty else 0

def compute_away_fouls_rolling_avg(data, away_team, match_date=None, window=5):

    if match_date is None:
        match_date = datetime.now()

    # Filter matches where the team played away
    away_matches = data[data['AwayTeam'] == away_team]

    # Filter only matches before the match date
    away_matches = away_matches[away_matches['Date'] < match_date]

    # Sort matches by date (most recent first) and take the last 'window' matches
    recent_away_matches = away_matches.sort_values('Date', ascending=False).head(window)

    # Compute rolling average of away shots (AS)
    return recent_away_matches['AF'].mean() if not recent_away_matches.empty else 0

def compute_home_corners_rolling_avg(data, home_team, match_date=None, window=5):

    if match_date is None:
        match_date = datetime.now()

    # Filter matches where the team played at home
    home_matches = data[data['HomeTeam'] == home_team]

    # Filter only matches before the match date
    home_matches = home_matches[home_matches['Date'] < match_date]

    # Sort matches by date (most recent first) and take the last 'window' matches
    recent_home_matches = home_matches.sort_values('Date', ascending=False).head(window)

    # Compute rolling average of home shots (HS)
    return recent_home_matches['HC'].mean() if not recent_home_matches.empty else 0

def compute_away_corners_rolling_avg(data, away_team, match_date=None, window=5):

    if match_date is None:
        match_date = datetime.now()

    # Filter matches where the team played away
    away_matches = data[data['AwayTeam'] == away_team]

    # Filter only matches before the match date
    away_matches = away_matches[away_matches['Date'] < match_date]

    # Sort matches by date (most recent first) and take the last 'window' matches
    recent_away_matches = away_matches.sort_values('Date', ascending=False).head(window)

    # Compute rolling average of away shots (AS)
    return recent_away_matches['AC'].mean() if not recent_away_matches.empty else 0

def compute_home_goals_rolling_avg(data, home_team, match_date=None, window=5):

    if match_date is None:
        match_date = datetime.now()

    # Filter matches where the team played at home
    home_matches = data[data['HomeTeam'] == home_team]

    # Filter only matches before the match date
    home_matches = home_matches[home_matches['Date'] < match_date]

    # Sort matches by date (most recent first) and take the last 'window' matches
    recent_home_matches = home_matches.sort_values('Date', ascending=False).head(window)

    # Compute rolling average of home shots (HS)
    return recent_home_matches['FTHG'].mean() if not recent_home_matches.empty else 0

def compute_away_goals_rolling_avg(data, away_team, match_date=None, window=5):

    if match_date is None:
        match_date = datetime.now()

    # Filter matches where the team played away
    away_matches = data[data['AwayTeam'] == away_team]

    # Filter only matches before the match date
    away_matches = away_matches[away_matches['Date'] < match_date]

    # Sort matches by date (most recent first) and take the last 'window' matches
    recent_away_matches = away_matches.sort_values('Date', ascending=False).head(window)

    # Compute rolling average of away shots (AS)
    return recent_away_matches['FTAG'].mean() if not recent_away_matches.empty else 0

def compute_home_shots_on_target_rolling_avg(data, home_team, match_date=None, window=5):

    if match_date is None:
        match_date = datetime.now()

    # Filter matches where the team played at home
    home_matches = data[data['HomeTeam'] == home_team]

    # Filter only matches before the match date
    home_matches = home_matches[home_matches['Date'] < match_date]

    # Sort matches by date (most recent first) and take the last 'window' matches
    recent_home_matches = home_matches.sort_values('Date', ascending=False).head(window)

    # Compute rolling average of home shots (HS)
    return recent_home_matches['HST'].mean() if not recent_home_matches.empty else 0

def compute_away_shots_on_target_rolling_avg(data, away_team, match_date=None, window=5):

    if match_date is None:
        match_date = datetime.now()

    # Filter matches where the team played away
    away_matches = data[data['AwayTeam'] == away_team]

    # Filter only matches before the match date
    away_matches = away_matches[away_matches['Date'] < match_date]

    # Sort matches by date (most recent first) and take the last 'window' matches
    recent_away_matches = away_matches.sort_values('Date', ascending=False).head(window)

    # Compute rolling average of away shots (AS)
    return recent_away_matches['AST'].mean() if not recent_away_matches.empty else 0


def compute_team_shots_rolling_avg(data, team_name, is_home, match_date=None, window=5):

    if match_date is None:
        match_date = datetime.now()

    if is_home:
        team_matches = data[data['HomeTeam'] == team_name]
        shots_column = 'HS'  # Home shots
    else:
        team_matches = data[data['AwayTeam'] == team_name]
        shots_column = 'AS'  # Away shots

    # Filter matches before the match date
    team_matches = team_matches[team_matches['Date'] < match_date]

    # Sort matches in descending order and take last 'window' matches
    recent_matches = team_matches.sort_values('Date', ascending=False).head(window)

    # Compute rolling average of shots
    return recent_matches[shots_column].mean() if not recent_matches.empty else 0

def estimate_total_shots(data, home_team, away_team, match_date=None, window=5):

    if match_date is None:
        match_date = datetime.now()

    avg_home_shots = compute_team_shots_rolling_avg(data, home_team, is_home=True, match_date=match_date, window=window)
    avg_away_shots = compute_team_shots_rolling_avg(data, away_team, is_home=False, match_date=match_date, window=window)

    total_shots_estimate = avg_home_shots + avg_away_shots
    return total_shots_estimate

def compute_team_corners_rolling_avg(data, team_name, is_home, match_date=None, window=5):

    if match_date is None:
        match_date = datetime.now()

    if is_home:
        team_matches = data[data['HomeTeam'] == team_name]
        corners_column = 'HC'  # Home shots
    else:
        team_matches = data[data['AwayTeam'] == team_name]
        corners_column = 'AC'  # Away shots

    # Filter matches before the match date
    team_matches = team_matches[team_matches['Date'] < match_date]

    # Sort matches in descending order and take last 'window' matches
    recent_matches = team_matches.sort_values('Date', ascending=False).head(window)

    # Compute rolling average of shots
    return recent_matches[corners_column].mean() if not recent_matches.empty else 0

def estimate_total_corners(data, home_team, away_team, match_date=None, window=5):

    if match_date is None:
        match_date = datetime.now()

    avg_home_corners = compute_team_corners_rolling_avg(data, home_team, is_home=True, match_date=match_date, window=window)
    avg_away_corners = compute_team_corners_rolling_avg(data, away_team, is_home=False, match_date=match_date, window=window)

    total_corners_estimate = avg_home_corners + avg_away_corners
    return total_corners_estimate

# Example usage
data = pd.read_csv('../e0/E0_24_25.csv')

data['Date'] = pd.to_datetime(data['Date'])  # Ensure Date column is in datetime format
home_team = "Brighton"
away_team = "Aston Villa"

#estimated_home_shots = compute_home_shots_rolling_avg(data, home_team)
#print(f"Estimated Rolling Average HS for {home_team} at home: {estimated_home_shots}")

match_data = prepare_match_prediction(data, home_team, away_team)
match_data['HS'] = compute_home_shots_rolling_avg(data, home_team)
match_data['AS'] = compute_away_shots_rolling_avg(data, away_team)
match_data['HF'] = compute_home_fouls_rolling_avg(data, home_team)
match_data['AF'] = compute_away_fouls_rolling_avg(data, away_team)
match_data['HC'] = compute_home_corners_rolling_avg(data, home_team)
match_data['AC'] = compute_away_corners_rolling_avg(data, away_team)
match_data['RollingAvg_FTHG'] = compute_home_goals_rolling_avg(data, home_team)
match_data['RollingAvg_FTAG'] = compute_away_goals_rolling_avg(data, away_team)
match_data['RollingAvg_TotalShots'] = estimate_total_shots(data, home_team, away_team)
match_data['RollingAvg_TotalCorners'] = estimate_total_corners(data, home_team, away_team)
match_data['HST'] = compute_home_shots_on_target_rolling_avg(data, home_team)
match_data['AST'] = compute_away_shots_on_target_rolling_avg(data, away_team)

print("\nMatch Features:")
print(match_data)

# Save the features for later use
match_data.to_csv("/home/alexandros/ml_bet/e0/match_data_features.csv", index=None, sep='|')

# Convert the upcoming match DataFrame (single row) to a dictionary
match_features_dict = match_data.iloc[0].to_dict()

# Print the dictionary
print(match_features_dict)