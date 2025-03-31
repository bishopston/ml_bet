def calculate_team_form(data, team_name, match_date, window=5):
    """
    Calculate team form from last N matches.
    Returns points based on: win=3, draw=1, loss=0
    """
    # Filter matches for the team up to the given date
    team_matches = data[
        (data['HomeTeam'] == team_name) | 
        (data['AwayTeam'] == team_name)
    ]
    team_matches = team_matches[team_matches['Date'] <= match_date]
    
    # Get last N matches
    recent_matches = team_matches.sort_values('Date', ascending=False).head(window)
    
    # Initialize form points
    form_points = 0
    
    # Calculate points for each match
    for _, match in recent_matches.iterrows():
        if match['HomeTeam'] == team_name:
            # Home team
            if match['FTR'] == 'H':
                form_points += 3  # Win
            elif match['FTR'] == 'D':
                form_points += 1  # Draw
            else:
                form_points += 0  # Loss
        else:
            # Away team
            if match['FTR'] == 'A':
                form_points += 3  # Win
            elif match['FTR'] == 'D':
                form_points += 1  # Draw
            else:
                form_points += 0  # Loss
    
    return form_points

def add_form_features(data, window=5):
    """
    Add form features to the dataset for each match.
    """
    # Create new columns for form features
    data['HomeTeam_Form'] = 0
    data['AwayTeam_Form'] = 0
    
    # Calculate form for each match
    for idx, match in data.iterrows():
        home_team = match['HomeTeam']
        away_team = match['AwayTeam']
        match_date = match['Date']
        
        # Calculate form for both teams
        home_form = calculate_team_form(data, home_team, match_date, window)
        away_form = calculate_team_form(data, away_team, match_date, window)
        
        # Update the form values in the DataFrame
        data.at[idx, 'HomeTeam_Form'] = home_form
        data.at[idx, 'AwayTeam_Form'] = away_form
    
    return data
