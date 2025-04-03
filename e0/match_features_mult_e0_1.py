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
    

def calculate_form_features(home_team, away_team, data, match_date=None):
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

    return {'HomeTeam_Form': calculate_form(home_team_matches, home_team),
            'AwayTeam_Form': calculate_form(away_team_matches, away_team)}


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
            'HS': float(compute_home_shots_rolling_avg(data, home_team, match_date, window)),
            'AS': float(compute_away_shots_rolling_avg(data, away_team, match_date, window)),
            'HF': float(compute_home_fouls_rolling_avg(data, home_team, match_date, window)),
            'AF': float(compute_away_fouls_rolling_avg(data, away_team, match_date, window)),
            'HC': float(compute_home_corners_rolling_avg(data, home_team, match_date, window)),
            'AC': float(compute_away_corners_rolling_avg(data, away_team, match_date, window)),
            'RollingAvg_FTHG': float(compute_home_goals_rolling_avg(data, home_team, match_date, window)),
            'RollingAvg_FTAG': float(compute_away_goals_rolling_avg(data, away_team, match_date, window)),
            'RollingAvg_TotalShots': float(estimate_total_shots(data, home_team, away_team, match_date, window)),
            'RollingAvg_TotalCorners': float(estimate_total_corners(data, home_team, away_team, match_date, window)),
            'RollingAvg_HomeHST': float(compute_home_shots_on_target_rolling_avg(data, home_team, match_date, window)),
            'RollingAvg_AwayAST': float(compute_away_shots_on_target_rolling_avg(data, away_team, match_date, window)),
            'HomeTeam_Form': float(calculate_form_features(home_team, away_team, data, match_date)['HomeTeam_Form']),
            'AwayTeam_Form': float(calculate_form_features(home_team, away_team, data, match_date)['AwayTeam_Form']),
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
    'AwayTeam': ['Tottenham', 'Everton']
})

# Get match features for multiple upcoming matches
match_features = prepare_all_matches(data, upcoming_matches)

# Print match features
print(match_features)
