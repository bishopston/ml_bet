import pandas as pd

def compute_home_shots_rolling_avg(data, home_team, match_date, window=5):
    """
    Compute the rolling average of home shots (HS) for a given team at home before a match date.
    
    :param data: DataFrame containing match data
    :param home_team: Name of the home team
    :param match_date: Date of the upcoming match
    :param window: Number of past home matches to use in the rolling average
    :return: Rolling average of HS for the home team
    """
    # Filter matches where the team played at home
    home_matches = data[data['HomeTeam'] == home_team]

    # Filter only matches before the match date
    home_matches = home_matches[home_matches['Date'] < match_date]

    # Sort matches by date (most recent first) and take the last 'window' matches
    recent_home_matches = home_matches.sort_values('Date', ascending=False).head(window)

    # Compute rolling average of home shots (HS)
    return recent_home_matches['HS'].mean() if not recent_home_matches.empty else 0

# Example usage:
# Load dataset
data = pd.read_csv('../e0/E0_24_25.csv')
data['Date'] = pd.to_datetime(data['Date'])  # Ensure Date column is in datetime format

# Example upcoming match
home_team = "Newcastle"
match_date = pd.to_datetime("2025-04-01")  # Adjust to your actual match date

# Estimate Rolling Average of HS
estimated_home_shots = compute_home_shots_rolling_avg(data, home_team, match_date)
print(f"Estimated Rolling Average HS for {home_team} at home: {estimated_home_shots}")
