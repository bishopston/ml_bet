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
    # Additional rolling averages for home/away stats if needed
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
features = ['HomeTeam', 'AwayTeam', 'HS', 'AS', 'HST', 'AST', 
            'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR', 
            'B365H', 'B365D', 'B365A', 'TotalShots', 'TotalCorners', 
            'RollingAvg_FTHG', 'RollingAvg_FTAG', 'RollingAvg_TotalShots', 'RollingAvg_TotalCorners',
            'RollingAvg_HomeHS', 'RollingAvg_HomeHST', 'RollingAvg_HomeHF', 'RollingAvg_HomeHC',
            'RollingAvg_AwayAS', 'RollingAvg_AwayAST', 'RollingAvg_AwayAF', 'RollingAvg_AwayAC']

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


# Function to preprocess future match data
def transform_teams(df, le):
    # Transform HomeTeam and AwayTeam using the same LabelEncoder fitted on training data
    df['HomeTeam'] = le.transform(df['HomeTeam'])
    df['AwayTeam'] = le.transform(df['AwayTeam'])
    return df

# Example data for a future match (Replace with actual teams playing today)
future_match = pd.DataFrame({
    'HomeTeam': ['Everton', 'Arsenal'],  # Example teams playing today
    'AwayTeam': ['Brighton', 'Manchester United'],
    'HS': [15, 12],  # Home Shots
    'AS': [10, 8],   # Away Shots
    'HST': [5, 4],   # Home Shots on Target
    'AST': [3, 2],   # Away Shots on Target
    'HF': [15, 10],  # Home Fouls
    'AF': [12, 9],   # Away Fouls
    'HC': [6, 4],    # Home Corners
    'AC': [4, 3],    # Away Corners
    'HY': [1, 2],    # Home Yellow Cards
    'AY': [1, 1],    # Away Yellow Cards
    'HR': [0, 0],    # Home Red Cards
    'AR': [0, 0],    # Away Red Cards
    'B365H': [2.5, 1.8],  # Bookmaker Home Odds
    'B365D': [3.2, 3.5],  # Bookmaker Draw Odds
    'B365A': [2.8, 4.0],  # Bookmaker Away Odds
})

# Compute additional features for future match
future_match['TotalShots'] = future_match['HS'] + future_match['AS']
future_match['TotalCorners'] = future_match['HC'] + future_match['AC']

# Compute rolling averages for future match
future_match = compute_rolling_avg(future_match)

# Transform teams (encode HomeTeam and AwayTeam)
future_match = transform_teams(future_match, le)

# Ensure future match data has the same columns as the training data
future_match = future_match[features]

# Predict the outcome for the future match
future_predictions = model.predict(future_match)

# Output the predictions
print(f"Predictions for the future match:")
print(future_predictions)  # 0: Home Win, 1: Draw, 2: Away Win
