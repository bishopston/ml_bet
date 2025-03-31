import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

# Load the dataset
data = pd.read_csv('E0_24_25.csv')

# Ensure Date column is in datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Check column names
print(data.columns)

def calculate_team_form(data, team_name, match_date, window=2):
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

def add_form_features(data, window=2):
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

# Add form features to the dataset
data = add_form_features(data)

# Compute additional features if needed
data['TotalShots'] = data['HS'] + data['AS']
data['TotalShotsOnTarget'] = data['HST'] + data['AST']
data['TotalCorners'] = data['HC'] + data['AC']

# Define function to compute rolling averages
def compute_rolling_avg(data, window=5):
    # Compute rolling averages for the last 'window' matches
    data['RollingAvg_FTHG'] = data['FTHG'].rolling(window=window, min_periods=1).mean()
    data['RollingAvg_FTAG'] = data['FTAG'].rolling(window=window, min_periods=1).mean()
    data['RollingAvg_TotalShots'] = data['TotalShots'].rolling(window=window, min_periods=1).mean()
    data['RollingAvg_TotalCorners'] = data['TotalCorners'].rolling(window=window, min_periods=1).mean()
    data['RollingAvg_HomeHS'] = data['HS'].rolling(window=window, min_periods=1).mean()
    data['RollingAvg_HomeHST'] = data['HST'].rolling(window=window, min_periods=1).mean()
    data['RollingAvg_HomeHF'] = data['HF'].rolling(window=window, min_periods=1).mean()
    data['RollingAvg_HomeHC'] = data['HC'].rolling(window=window, min_periods=1).mean()
    data['RollingAvg_AwayAS'] = data['AS'].rolling(window=window, min_periods=1).mean()
    data['RollingAvg_AwayAST'] = data['AST'].rolling(window=window, min_periods=1).mean()
    data['RollingAvg_AwayAF'] = data['AF'].rolling(window=window, min_periods=1).mean()
    data['RollingAvg_AwayAC'] = data['AC'].rolling(window=window, min_periods=1).mean()
    return data

# Apply the function to compute rolling averages
data = compute_rolling_avg(data)

# Define features (include FTR separately)
features = ['HomeTeam', 'AwayTeam', 'HS', 'AS',  
            'HF', 'AF', 'HC', 'AC', 
            'B365H', 'B365D', 'B365A', 
            'RollingAvg_FTHG', 'RollingAvg_FTAG', 
            'RollingAvg_TotalShots',
            'RollingAvg_TotalCorners',
            'RollingAvg_HomeHST', 
            'RollingAvg_AwayAST', 
            # 'RollingAvg_HomeHS',
            # 'RollingAvg_AwayAS',
            # 'RollingAvg_HomeHC',
            # 'RollingAvg_AwayAC',
            'HomeTeam_Form',
            'AwayTeam_Form']

# Create dataset with features + target
data_filtered = data[features + ['FTR']]

# Drop missing values
data_filtered.dropna(inplace=True)

# Encode categorical columns
#le = LabelEncoder()

# Encode categorical columns using separate LabelEncoders
home_team_le = LabelEncoder()
away_team_le = LabelEncoder()
result_le = LabelEncoder()

data_filtered['HomeTeam'] = home_team_le.fit_transform(data_filtered['HomeTeam'])
data_filtered['AwayTeam'] = away_team_le.fit_transform(data_filtered['AwayTeam'])
data_filtered['FTR'] = result_le.fit_transform(data_filtered['FTR'])

# Store the team encoders for later use
team_encoders = {
    'home': home_team_le,
    'away': away_team_le
}

# Define X and y
X = data_filtered.drop('FTR', axis=1)
y = data_filtered['FTR']

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Show split data shapes
print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")

# Initialize XGBoost model
model = XGBClassifier(eval_metric='mlogloss', use_label_encoder=False)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
print(f"Model Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))

def prepare_match_prediction(home_team, away_team):
    """
    Prepare features for predicting a specific match.
    Now includes more realistic default values based on Premier League averages.
    """
    # Create base features dictionary with realistic defaults
    match_features = {
        'HomeTeam': home_team,
        'AwayTeam': away_team,
        'HS': 12,      # Average home team shots per match
        'AS': 9,       # Average away team shots per match
        #'HST': 4,      # Average home team shots on target
        #'AST': 3,      # Average away team shots on target
        'HF': 12,      # Average home team fouls
        'AF': 11,      # Average away team fouls
        'HC': 5,       # Average home team corners
        'AC': 4,       # Average away team corners
        # 'HY': 1,       # Average home team yellow cards
        # 'AY': 1,       # Average away team yellow cards
        # 'HR': 0,       # Average home team red cards
        # 'AR': 0,       # Average away team red cards
        'B365H': 2.50, 'B365D': 3.30, 'B365A': 2.70,  # Typical bookmaker odds
        #'TotalShots': 21,  # Average total shots per match
        #'TotalCorners': 9,   # Average total corners per match
        'RollingAvg_FTHG': 1.5,  # Average home team goals
        'RollingAvg_FTAG': 1.0,  # Average away team goals conceded
        'RollingAvg_TotalShots': 21.0,
        'RollingAvg_TotalCorners': 9.0,
        'RollingAvg_HomeHST': 5,
        'RollingAvg_AwayAST': 6,
        # 'RollingAvg_HomeHS': 7,
        # 'RollingAvg_AwayAS': 7,
        # 'RollingAvg_HomeHC': 6,
        # 'RollingAvg_AwayAC': 6,
        'HomeTeam_Form': 4,
        'AwayTeam_Form': 4
    }
    
    return pd.DataFrame([match_features])

def predict_match(model, home_team_name, away_team_name):
    """
    Make prediction for a specific upcoming match.
    Now uses the correct team encoders.
    """
    # Get team encodings from existing model
    available_home_teams = set(team_encoders['home'].classes_)
    available_away_teams = set(team_encoders['away'].classes_)
    
    # Print available teams for debugging
    print("\nAvailable teams in dataset:")
    print("Home teams:", ", ".join(sorted(list(available_home_teams))[:10]) + "...")
    print("Away teams:", ", ".join(sorted(list(available_away_teams))[:10]) + "...")
    
    # Try to find exact team name match
    if home_team_name not in available_home_teams or away_team_name not in available_away_teams:
        print("\nWarning: Teams not found in training data.")
        print(f"Looking for: {home_team_name}, {away_team_name}")
        
        # Try to find similar team names
        similar_teams = []
        for team in available_home_teams:
            if home_team_name.lower() in team.lower():
                similar_teams.append(team)
        for team in available_away_teams:
            if away_team_name.lower() in team.lower():
                similar_teams.append(team)
                
        if similar_teams:
            print("\nSimilar team names found:")
            for team in similar_teams:
                print(f"- {team}")
                
        return None
    
    # Prepare match features
    home_team = team_encoders['home'].transform([home_team_name])[0]
    away_team = team_encoders['away'].transform([away_team_name])[0]
    
    match_data = prepare_match_prediction(int(home_team), int(away_team))
    
    # Make prediction
    predictions = model.predict_proba(match_data)
    
    # Map predictions to readable format
    result_map = {0: 'Home Win', 1: 'Draw', 2: 'Away Win'}
    probabilities = {
        'Home Win': round(predictions[0][0] * 100, 2),
        'Draw': round(predictions[0][1] * 100, 2),
        'Away Win': round(predictions[0][2] * 100, 2)
    }
    
    return probabilities

# Example usage
team_names = ['Ipswich', 'Nott\'m Forest']  # Note the correct team name format
probabilities = predict_match(model, team_names[0], team_names[1])

if probabilities is None:
    print("\nPlease use the exact team names as they appear in the dataset.")
else:
    print("\nPrediction Results:")
    for outcome, prob in probabilities.items():
        print(f"{outcome}: {prob}%")

