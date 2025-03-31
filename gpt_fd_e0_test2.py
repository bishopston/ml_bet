import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
data = pd.read_csv('football_data_co_uk/E0.csv')

# Check column names
print(data.columns)

# Compute additional features if needed
data['TotalShots'] = data['HS'] + data['AS']
data['TotalCorners'] = data['HC'] + data['AC']

# Define function to compute rolling averages for goals, shots, and corners
def compute_rolling_averages(df, team_col, goal_col, shots_col, corners_col, prefix):
    df = df.sort_values(by=['Date'])  # Sort by Date (important for rolling features)
    df[f'{prefix}_AvgGoals_Last5'] = df.groupby(team_col)[goal_col].transform(lambda x: x.shift(1).rolling(5).mean())
    df[f'{prefix}_AvgConceded_Last5'] = df.groupby(team_col)[goal_col].transform(lambda x: x.shift(1).rolling(5).mean())
    df[f'{prefix}_AvgShotsOnTarget_Last5'] = df.groupby(team_col)[shots_col].transform(lambda x: x.shift(1).rolling(5).mean())
    df[f'{prefix}_AvgCorners_Last5'] = df.groupby(team_col)[corners_col].transform(lambda x: x.shift(1).rolling(5).mean())
    return df

# Ensure Date column is in datetime format
data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)

# Sort data by Date (important for rolling features)
data = data.sort_values(by=['Date'])

# Create a copy to store rolling features
rolling_features = data[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'HST', 'AST', 'HC', 'AC']].copy()

# Compute rolling averages for Home and Away teams
rolling_features = compute_rolling_averages(rolling_features, 'HomeTeam', 'FTHG', 'HST', 'HC', 'Home')
rolling_features = compute_rolling_averages(rolling_features, 'AwayTeam', 'FTAG', 'AST', 'AC', 'Away')

# Merge rolling averages back to original dataset
data = data.merge(rolling_features.drop(columns=['FTHG', 'FTAG', 'HST', 'AST', 'HC', 'AC']), 
                  on=['Date', 'HomeTeam', 'AwayTeam'], how='left')

# Drop rows with NaN values (first 5 matches for each team will have NaNs)
data.dropna(inplace=True)

# Define features (include FTR separately)
features = ['HomeTeam', 'AwayTeam', 'HS', 'AS', 'HST', 'AST', 
            'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR', 
            'B365H', 'B365D', 'B365A', 'Home_AvgGoals_Last5', 'Home_AvgConceded_Last5', 
            'Home_AvgShotsOnTarget_Last5', 'Home_AvgCorners_Last5', 
            'Away_AvgGoals_Last5', 'Away_AvgConceded_Last5', 
            'Away_AvgShotsOnTarget_Last5', 'Away_AvgCorners_Last5', 
            'TotalShots', 'TotalCorners']

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

# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate
print(f"Model Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))
