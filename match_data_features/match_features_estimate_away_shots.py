import pandas as pd

def compute_away_shots_rolling_avg(data, away_team, match_date, window=5):
    """
    Compute the rolling average of away shots (AS) for a given team playing away before a match date.
    
    :param data: DataFrame containing match data
    :param away_team: Name of the away team
    :param match_date: Date of the upcoming match
    :param window: Number of past away matches to use in the rolling average
    :return: Rolling average of AS for the away team
    """
    # Filter matches where the team played away
    away_matches = data[data['AwayTeam'] == away_team]

    # Filter only matches before the match date
    away_matches = away_matches[away_matches['Date'] < match_date]

    # Sort matches by date (most recent first) and take the last 'window' matches
    recent_away_matches = away_matches.sort_values('Date', ascending=False).head(window)

    # Compute rolling average of away shots (AS)
    return recent_away_matches['AS'].mean() if not recent_away_matches.empty else 0

# Example usage:
# Load dataset
data = pd.read_csv('../e0/E0_24_25.csv')
data['Date'] = pd.to_datetime(data['Date'])  # Ensure Date column is in datetime format

# Example upcoming match
away_team = "Aston Villa"
match_date = pd.to_datetime("2025-04-01")  # Adjust to your actual match date

# Estimate Rolling Average of AS
estimated_away_shots = compute_away_shots_rolling_avg(data, away_team, match_date)
print(f"Estimated Rolling Average AS for {away_team} away: {estimated_away_shots}")
