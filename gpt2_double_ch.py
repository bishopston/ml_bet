import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Sample dataset
data = {
    'home_team': ['Team A', 'Team C', 'Team E', 'Team G', 'Team I'],
    'away_team': ['Team B', 'Team D', 'Team F', 'Team H', 'Team J'],
    'home_goals': [2, 1, 0, 3, 1],
    'away_goals': [1, 1, 1, 2, 2],
    'home_odds': [1.80, 2.10, 1.95, 1.75, 2.50],
    'draw_odds': [3.50, 3.20, 3.40, 3.60, 3.10],
    'away_odds': [4.00, 3.50, 3.80, 4.20, 2.90]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Define outcome for Double Chance
df['outcome'] = np.where(df['home_goals'] > df['away_goals'], 1,  # Home Win
                          np.where(df['home_goals'] == df['away_goals'], 3, 2))  # Draw or Away Win

# Select Features & Target
X = df[['home_odds', 'draw_odds', 'away_odds']]
y = df['outcome']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale Features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Model (Logistic Regression for multiclass)
model = LogisticRegression(multi_class='ovr', solver='liblinear')  # 'ovr' = One-vs-Rest
model.fit(X_train, y_train)

# Make Predictions
y_pred = model.predict(X_test)

# Evaluate Model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
print(classification_report(y_test, y_pred))
