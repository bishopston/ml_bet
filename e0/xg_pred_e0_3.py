import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

# Load the dataset
data = pd.read_csv('E0_24_25.csv')

# Ensure Date column is in datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Define correct FTR mapping
ftr_mapping = {'H': 0, 'D': 1, 'A': 2}

def calculate_team_form(data, team_name, match_date, window=2):
    """
    Calculate team form from last N matches.
    Returns points based on: win=3, draw=1, loss=0
    """
    team_matches = data[(data['HomeTeam'] == team_name) | (data['AwayTeam'] == team_name)]
    team_matches = team_matches[team_matches['Date'] <= match_date]
    recent_matches = team_matches.sort_values('Date', ascending=False).head(window)
    
    form_points = 0
    for _, match in recent_matches.iterrows():
        if match['HomeTeam'] == team_name:
            form_points += 3 if match['FTR'] == 'H' else (1 if match['FTR'] == 'D' else 0)
        else:
            form_points += 3 if match['FTR'] == 'A' else (1 if match['FTR'] == 'D' else 0)
    
    return form_points

def add_form_features(data, window=2):
    """
    Add form features to the dataset for each match.
    """
    data['HomeTeam_Form'] = 0
    data['AwayTeam_Form'] = 0
    
    for idx, match in data.iterrows():
        home_team = match['HomeTeam']
        away_team = match['AwayTeam']
        match_date = match['Date']
        
        home_form = calculate_team_form(data, home_team, match_date, window)
        away_form = calculate_team_form(data, away_team, match_date, window)
        
        data.at[idx, 'HomeTeam_Form'] = home_form
        data.at[idx, 'AwayTeam_Form'] = away_form
    
    return data

data = add_form_features(data)

# Compute additional features
data['TotalShots'] = data['HS'] + data['AS']
data['TotalShotsOnTarget'] = data['HST'] + data['AST']
data['TotalCorners'] = data['HC'] + data['AC']

# Rolling averages
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

data = compute_rolling_avg(data)

# Define features
features = [
    'HomeTeam', 'AwayTeam', 'HS', 'AS', 'HF', 'AF', 'HC', 'AC',
    'RollingAvg_FTHG', 'RollingAvg_FTAG', 'RollingAvg_TotalShots',
    'RollingAvg_TotalCorners', 'RollingAvg_HomeHST', 'RollingAvg_AwayAST',
    'HomeTeam_Form', 'AwayTeam_Form'
]

data_filtered = data[features + ['FTR']].dropna()

# Encode categorical columns
home_team_le = LabelEncoder()
away_team_le = LabelEncoder()

data_filtered['HomeTeam'] = home_team_le.fit_transform(data_filtered['HomeTeam'])
data_filtered['AwayTeam'] = away_team_le.fit_transform(data_filtered['AwayTeam'])

# Manually encode FTR to ensure correct mapping
data_filtered['FTR'] = data_filtered['FTR'].map(ftr_mapping)

# Store team encoders
team_encoders = {'home': home_team_le, 'away': away_team_le}

# Define X and y
X = data_filtered.drop('FTR', axis=1)
y = data_filtered['FTR']

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost model
model = XGBClassifier(
    eval_metric='mlogloss', 
    use_label_encoder=False,
    reg_alpha=0.1,  # L1 regularization
    reg_lambda=1,   # L2 regularization
    max_depth=6,    # Limiting tree depth
    learning_rate=0.05  # Adjust learning rate for slower learning
)
model.fit(X_train, y_train)

# Model evaluation
y_pred = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))

# Check accuracy on training data
train_pred = model.predict(X_train)
train_acc = accuracy_score(y_train, train_pred)

# Check accuracy on test data
test_pred = model.predict(X_test)
test_acc = accuracy_score(y_test, test_pred)

print(f"Training Accuracy: {train_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

# Perform 5-fold cross-validation
cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

print(f"Cross-validation accuracy scores: {cv_scores}")
print(f"Mean accuracy: {cv_scores.mean():.4f}")

# Prepare match prediction
def prepare_match_prediction(home_team, away_team):
    match_features = {
        'HomeTeam': home_team, 'AwayTeam': away_team,
        'HS': 15.2, 'AS': 14, 'HF': 11.6, 'AF': 13.6, 'HC': 6.4, 'AC': 4.8,
        'RollingAvg_FTHG': 2.0, 'RollingAvg_FTAG': 1.8,
        'RollingAvg_TotalShots': 29.2, 'RollingAvg_TotalCorners': 11.2,
        'RollingAvg_HomeHST': 4.6, 'RollingAvg_AwayAST': 3.4,
        'HomeTeam_Form': 4, 'AwayTeam_Form': 3
    }
    return pd.DataFrame([match_features])

# Prediction function
def predict_match(model, home_team_name, away_team_name):
    available_home_teams = set(team_encoders['home'].classes_)
    available_away_teams = set(team_encoders['away'].classes_)
    
    if home_team_name not in available_home_teams or away_team_name not in available_away_teams:
        print("\nWarning: Teams not found in training data.")
        return None
    
    home_team = team_encoders['home'].transform([home_team_name])[0]
    away_team = team_encoders['away'].transform([away_team_name])[0]
    
    match_data = prepare_match_prediction(int(home_team), int(away_team))
    
    # Predict probabilities
    predictions = model.predict_proba(match_data)
    
    # Ensure correct mapping for predictions
    result_map = {0: 'Home Win', 1: 'Draw', 2: 'Away Win'}
    probabilities = {
        'Home Win': round(predictions[0][0] * 100, 2),
        'Draw': round(predictions[0][1] * 100, 2),
        'Away Win': round(predictions[0][2] * 100, 2)
    }
    
    return probabilities

# Example usage
team_names = ['Arsenal', 'Fulham']
probabilities = predict_match(model, team_names[0], team_names[1])

if probabilities is None:
    print("\nPlease use the exact team names as they appear in the dataset.")
else:
    print("\nPrediction Results:")
    for outcome, prob in probabilities.items():
        print(f"{outcome}: {prob}%")
