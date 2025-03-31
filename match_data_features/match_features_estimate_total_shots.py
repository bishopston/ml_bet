import pandas as pd

def compute_team_rolling_avg(data, team_name, is_home, match_date, window=5):
    """
    Compute the rolling average of shots (HS or AS) for a given team up to a match date.
    
    :param data: DataFrame containing match data
    :param team_name: Name of the team
    :param is_home: True if calculating for home shots (HS), False for away shots (AS)
    :param match_date: Date of the upcoming match
    :param window: Number of past matches to use in the rolling average
    :return: Rolling average of shots
    """
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

def estimate_total_shots(data, home_team, away_team, match_date, window=5):
    """
    Estimate TotalShots for an upcoming match using rolling averages.
    
    :param data: DataFrame containing match data
    :param home_team: Name of the home team
    :param away_team: Name of the away team
    :param match_date: Date of the upcoming match
    :param window: Number of past matches to use in the rolling average
    :return: Estimated TotalShots
    """
    avg_home_shots = compute_team_rolling_avg(data, home_team, is_home=True, match_date=match_date, window=window)
    avg_away_shots = compute_team_rolling_avg(data, away_team, is_home=False, match_date=match_date, window=window)

    total_shots_estimate = avg_home_shots + avg_away_shots
    return total_shots_estimate

# Example usage:
# Load dataset
data = pd.read_csv('../e0/E0_24_25.csv')
data['Date'] = pd.to_datetime(data['Date'])  # Ensure Date column is in datetime format

# Example upcoming match
home_team = "Newcastle"
away_team = "Aston Villa"
match_date = pd.to_datetime("2025-04-01")  # Adjust to your actual match date

# Estimate TotalShots
estimated_total_shots = estimate_total_shots(data, home_team, away_team, match_date)
print(f"Estimated TotalShots for {home_team} vs {away_team}: {estimated_total_shots}")
