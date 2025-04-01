import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('I1_24_25.csv')

# Ensure Date column is in datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Fill missing market odds with median values
for col in ['B365H', 'B365D', 'B365A']:
    data[col] = data[col].fillna(data[col].median())

# # Convert odds to implied probability (Optional)
# data['Implied_Prob_H'] = 1 / data['B365H']
# data['Implied_Prob_D'] = 1 / data['B365D']
# data['Implied_Prob_A'] = 1 / data['B365A']

# # Normalize implied probabilities (Optional)
# scaler = StandardScaler()
# data[['Implied_Prob_H', 'Implied_Prob_D', 'Implied_Prob_A']] = scaler.fit_transform(
#     data[['Implied_Prob_H', 'Implied_Prob_D', 'Implied_Prob_A']]
# )

# Define correct FTR mapping
ftr_mapping = {'H': 0, 'D': 1, 'A': 2}

def calculate_team_form(data, team_name, match_date, window=5):
    """
    Calculate team form from last N matches.
    Returns points based on: win=3, draw=1, loss=0
    """
    team_matches = data[(data['HomeTeam'] == team_name) | (data['AwayTeam'] == team_name)]
    team_matches = team_matches[team_matches['Date'] < match_date]  # Exclude current match
    recent_matches = team_matches.sort_values('Date', ascending=False).head(window)
    
    form_points = 0
    for _, match in recent_matches.iterrows():
        if match['HomeTeam'] == team_name:
            form_points += 3 if match['FTR'] == 'H' else (1 if match['FTR'] == 'D' else 0)
        else:
            form_points += 3 if match['FTR'] == 'A' else (1 if match['FTR'] == 'D' else 0)
    
    return form_points

def add_form_features(data, window=5):
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

# Normalize form by dividing by window size
data['HomeTeam_Form'] = data['HomeTeam_Form'] / 5  # Scale to 0-1
data['AwayTeam_Form'] = data['AwayTeam_Form'] / 5  # Scale to 0-1

# Compute rolling averages per team
def compute_team_rolling_avg(data, window=5):
    """
    Compute rolling averages for team-specific stats over the last 'window' matches.
    """
    rolling_features = ['HS', 'HST', 'HF', 'HC', 'AS', 'AST', 'AF', 'AC', 'B365H', 'B365D', 'B365A']

    for feature in rolling_features:
        # Rolling average for home team stats
        data[f'RollingAvg_Home_{feature}'] = (
            data.groupby('HomeTeam')[feature]
            .rolling(window=window, min_periods=1)
            .mean()
            .reset_index(0, drop=True)
        )

        # Rolling average for away team stats
        data[f'RollingAvg_Away_{feature}'] = (
            data.groupby('AwayTeam')[feature]
            .rolling(window=window, min_periods=1)
            .mean()
            .reset_index(0, drop=True)
        )

    return data

data = compute_team_rolling_avg(data)



# Define features
features = [
    'HomeTeam', 'AwayTeam',
    'RollingAvg_Home_HS', 'RollingAvg_Home_HST', 'RollingAvg_Home_HF', 'RollingAvg_Home_HC',
    'RollingAvg_Away_AS', 'RollingAvg_Away_AST', 'RollingAvg_Away_AF', 'RollingAvg_Away_AC',
    'HomeTeam_Form', 'AwayTeam_Form',
    'B365H', 'B365D', 'B365A'
]

data_filtered = data[features + ['FTR']].dropna()
data_filtered.to_csv("/home/alexandros/ml_bet/data_ext.csv", index=None, sep='|')

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

# Cross-validation
cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"Cross-validation accuracy: {cv_scores.mean():.4f}")

# Feature importance
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

# Save feature importance
feature_importance_df = pd.DataFrame({
    'Feature': [features[i] for i in indices],
    'Importance': importances[indices]
})
feature_importance_df.to_csv('feature_importances.csv', index=False)

# Prepare match prediction
def prepare_match_prediction(home_team, away_team):
    return pd.DataFrame([{
        'HomeTeam': home_team, 'AwayTeam': away_team,
        'RollingAvg_Home_HS': 18.0, 'RollingAvg_Home_HST': 6.6, 
        'RollingAvg_Home_HF': 15.2, 'RollingAvg_Home_HC': 5.6,
        'RollingAvg_Away_AS': 9.8, 'RollingAvg_Away_AST': 2.4, 
        'RollingAvg_Away_AF': 13.0, 'RollingAvg_Away_AC': 4.8,
        'HomeTeam_Form': 0.0, 'AwayTeam_Form': 2.0,
        'B365H': 1.62, 'B365D': 3.25, 'B365A': 5.25
    }])

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

# Predict match
team_names = ['Juventus', 'Genoa']
probabilities = predict_match(model, team_names[0], team_names[1])
print(probabilities)
