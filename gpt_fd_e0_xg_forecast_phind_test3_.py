import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

# Load the dataset
data = pd.read_csv('football_data_co_uk/E0.csv')

# Check column names
print(data.columns)

# Compute additional features if needed
data['TotalShots'] = data['HS'] + data['AS']
data['TotalCorners'] = data['HC'] + data['AC']

# Define function to compute rolling averages
def compute_rolling_avg(data, window=5):
    # Compute rolling averages for the last 'window' matches
    data['RollingAvg_FTHG'] = data['FTHG'].rolling(window=window, min_periods=1).mean()
    data['RollingAvg_FTAG'] = data['FTAG'].rolling(window=window, min_periods=1).mean()
    data['RollingAvg_TotalShots'] = data['TotalShots'].rolling(window=window, min_periods=1).mean()
    data['RollingAvg_TotalCorners'] = data['TotalCorners'].rolling(window=window, min_periods=1).mean()
    return data

# Apply the function to compute rolling averages
data = compute_rolling_avg(data)

# Define features (include FTR separately)
features = ['HomeTeam', 'AwayTeam', 'HS', 'AS', 'HST', 'AST', 
            'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR', 
            'B365H', 'B365D', 'B365A', 'TotalShots', 'TotalCorners', 
            'RollingAvg_FTHG', 'RollingAvg_FTAG', 'RollingAvg_TotalShots', 'RollingAvg_TotalCorners']

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
    This uses your existing model and feature engineering approach.
    """
    # Create base features dictionary
    match_features = {
        'HomeTeam': home_team,
        'AwayTeam': away_team,
        'HS': 0,      # Shots taken by home team
        'AS': 0,      # Shots taken by away team
        'HST': 0,     # Shots on target by home team
        'AST': 0,     # Shots on target by away team
        'HF': 0,      # Fouls committed by home team
        'AF': 0,      # Fouls committed by away team
        'HC': 0,      # Corners taken by home team
        'AC': 0,      # Corners taken by away team
        'HY': 0,      # Yellow cards received by home team
        'AY': 0,      # Yellow cards received by away team
        'HR': 0,      # Red cards received by home team
        'AR': 0,      # Red cards received by away team
        'B365H': 2.50, 'B365D': 3.30, 'B365A': 2.70,  # Example odds
        'TotalShots': 25,  # Average total shots per match
        'TotalCorners': 9,   # Average corners per match
        'RollingAvg_FTHG': 1.5,  # Home team goals scored
        'RollingAvg_FTAG': 1.0,  # Away team goals conceded
        'RollingAvg_TotalShots': 24.0,
        'RollingAvg_TotalCorners': 9.0
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

