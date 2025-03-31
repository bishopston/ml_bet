import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

# Load the dataset
data = pd.read_csv('i1/I1_24_25.csv')

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
    data['RollingAvg_TotalShotsOnTarget'] = data['TotalShotsOnTarget'].rolling(window=window, min_periods=1).mean()
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
print(data.head())

#data.to_csv("/home/alexandros/ml_bet/data_form.csv", index=None, sep='|')

# Display the first few rows to verify the form features
print("\nFirst few rows of the dataset with form features:")
print(data[['Date', 'HomeTeam', 'AwayTeam', 'HomeTeam_Form', 'AwayTeam_Form', 'FTR']].head())

# Define features (include FTR separately)
features = ['HomeTeam', 'AwayTeam', 'HS', 'AS', 'HST', 'AST', 
            'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR', 
            'B365H', 'B365D', 'B365A', 'TotalShots', 'TotalCorners', 
            'RollingAvg_FTHG', 'RollingAvg_FTAG', 'RollingAvg_TotalShots', 'RollingAvg_TotalCorners',
            'RollingAvg_HomeHST', 
            'RollingAvg_AwayAST', 
            'HomeTeam_Form',
            'AwayTeam_Form']

# Create dataset with features + target
data_filtered = data[features + ['FTR']]

# Drop missing values
data_filtered.dropna(inplace=True)

# Encode categorical columns
le = LabelEncoder()
data_filtered['HomeTeam'] = le.fit_transform(data_filtered['HomeTeam'])
data_filtered['AwayTeam'] = le.fit_transform(data_filtered['AwayTeam'])
data_filtered['FTR'] = le.fit_transform(data_filtered['FTR'])  # Home Win = 0, Draw = 1, Away Win = 2

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
