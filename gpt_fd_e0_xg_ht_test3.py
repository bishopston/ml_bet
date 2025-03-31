import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
data = pd.read_csv('football_data_co_uk/E0.csv')

# Check column names
print(data.columns)

# Compute additional features if needed
data['TotalShots'] = data['HS'] + data['AS']
data['TotalCorners'] = data['HC'] + data['AC']

# Define features (include FTR separately)
features = ['HomeTeam', 'AwayTeam', 'HS', 'AS', 'HST', 'AST', 
            'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR', 
            'B365H', 'B365D', 'B365A', 'TotalShots', 'TotalCorners']

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

# Initialize the model
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

# Define the parameter grid
param_grid = {
    'learning_rate': [0.01, 0.1, 0.2, 0.3],
    'max_depth': [3, 4, 5, 6],
    'n_estimators': [50, 100, 200],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
    'gamma': [0, 0.1, 0.2],
}

# Setup the grid search
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, 
                           scoring='accuracy', cv=3, verbose=2, n_jobs=-1)

# Fit grid search
grid_search.fit(X_train, y_train)

# Get the best parameters
best_params = grid_search.best_params_
print(f"Best Parameters: {best_params}")

# Train the model with the best parameters
best_model = grid_search.best_estimator_

# Make predictions
y_pred = best_model.predict(X_test)

# Evaluate
print(f"Model Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))

