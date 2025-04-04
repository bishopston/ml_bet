import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from match_features_mult_f1_1 import match_features

# Load the dataset
data = pd.read_csv('../f1/24_25/F1_020425.csv')

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
    reg_alpha=0.5,  # Increase L1 regularization
    reg_lambda=2,   # Keep L2 regularization
    max_depth=6,    # Limiting tree depth
    learning_rate=0.03,
    subsample=0.7,  # Subsample fraction for each tree
    colsample_bytree=0.7  # Column sampling for each tree
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

# Get the feature importances from the model
importances = model.feature_importances_

# Sort the importances in descending order
indices = np.argsort(importances)[::-1]

# Create a DataFrame to hold the feature names and their corresponding importances
feature_importance_df = pd.DataFrame({
    'Feature': [features[i] for i in indices],
    'Importance': importances[indices]
})

# Save the table as a CSV file
feature_importance_df.to_csv('feature_importances.csv', index=False)

# Optionally, print the table
#print(feature_importance_df)

# Plot feature importances
plt.figure(figsize=(10, 8))
plt.barh(range(len(indices)), importances[indices])
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Feature Importance')
plt.title('Feature Importances')
plt.tight_layout()

# Save the plot to a file
plt.savefig('feature_importances.png')

def prepare_match_predictions(matches_with_features):
    match_dataframes = []
    
    for match in matches_with_features:
        home_team = match['home_team']
        away_team = match['away_team']
        features = match['features']

        # Transform team names to encoded values
        home_team_encoded = team_encoders['home'].transform([home_team])[0]
        away_team_encoded = team_encoders['away'].transform([away_team])[0]
        
        match_features = {
            'HomeTeam': home_team_encoded,
            'AwayTeam': away_team_encoded,
            'HS': features['HS'],
            'AS': features['AS'],
            'HF': features['HF'],
            'AF': features['AF'],
            'HC': features['HC'],
            'AC': features['AC'],
            'RollingAvg_FTHG': features['RollingAvg_FTHG'],
            'RollingAvg_FTAG': features['RollingAvg_FTAG'],
            'RollingAvg_TotalShots': features['RollingAvg_TotalShots'],
            'RollingAvg_TotalCorners': features['RollingAvg_TotalCorners'],
            'RollingAvg_HomeHST': features['RollingAvg_HomeHST'],
            'RollingAvg_AwayAST': features['RollingAvg_AwayAST'],
            'HomeTeam_Form': features['HomeTeam_Form'],
            'AwayTeam_Form': features['AwayTeam_Form']
        }
        
        match_dataframes.append(pd.DataFrame([match_features]))
    
    return match_dataframes

def predict_matches(model, matches_with_features):
    """
    Predict outcomes for multiple matches at once.
    
    Args:
        model: Trained XGBClassifier model
        matches_with_features: List of dictionaries containing match details and features
        
    Returns:
        Dictionary mapping (home_team, away_team) to prediction probabilities
    """
    available_home_teams = set(team_encoders['home'].classes_)
    available_away_teams = set(team_encoders['away'].classes_)
    
    predictions = {}
    
    for match in matches_with_features:
        # Check if teams exist in training data
        if match['home_team'] not in available_home_teams or match['away_team'] not in available_away_teams:
            print(f"\nWarning: Teams not found in training data: {match['home_team']} vs {match['away_team']}")
            continue
            
        # Transform team names to encoded values
        home_team = team_encoders['home'].transform([match['home_team']])[0]
        away_team = team_encoders['away'].transform([match['away_team']])[0]
        
        # Prepare match features
        match_data = prepare_match_predictions([match])[0]
        
        # Get predictions
        predictions_proba = model.predict_proba(match_data)
        
        # Convert to readable format
        result_map = {0: 'Home Win', 1: 'Draw', 2: 'Away Win'}
        probabilities = {
            'Home Win': round(predictions_proba[0][0] * 100, 2),
            'Draw': round(predictions_proba[0][1] * 100, 2),
            'Away Win': round(predictions_proba[0][2] * 100, 2)
        }
        
        predictions[(match['home_team'], match['away_team'])] = probabilities
    
    return predictions

# Example usage with multiple games
# matches_with_features = [
#     {
#         'home_team': 'Chelsea',
#         'away_team': 'Tottenham',
#         'features': {
#             'HS': 21.2, 'AS': 11.2, 'HF': 10.4, 'AF': 11.6,
#             'HC': 6.4, 'AC': 4.8, 'RollingAvg_FTHG': 2.4,
#             'RollingAvg_FTAG': 1.8, 'RollingAvg_TotalShots': 32.4,
#             'RollingAvg_TotalCorners': 11.2, 'RollingAvg_HomeHST': 7.4,
#             'RollingAvg_AwayAST': 4.0, 'HomeTeam_Form': 3, 'AwayTeam_Form': 1
#         }
#     },
#     {
#         'home_team': 'Liverpool',
#         'away_team': 'Man City',
#         'features': {
#             'HS': 20.5, 'AS': 12.1, 'HF': 11.3, 'AF': 10.8,
#             'HC': 7.2, 'AC': 5.1, 'RollingAvg_FTHG': 2.6,
#             'RollingAvg_FTAG': 1.9, 'RollingAvg_TotalShots': 33.6,
#             'RollingAvg_TotalCorners': 12.5, 'RollingAvg_HomeHST': 8.1,
#             'RollingAvg_AwayAST': 4.2, 'HomeTeam_Form': 4, 'AwayTeam_Form': 2
#         }
#     }
# ]

matches_with_features = match_features

#print(match_features)

# Get predictions for all matches
predictions = predict_matches(model, matches_with_features)

# Print results in a formatted way
print("\nMultiple Match Predictions:")
print("-" * 50)
for (home_team, away_team), probs in predictions.items():
    print(f"\n{home_team} vs {away_team}:")
    for outcome, prob in probs.items():
        print(f"{outcome}: {prob}%")