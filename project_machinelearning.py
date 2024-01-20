import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error
import sklearn.ensemble
import sklearn.metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import LabelEncoder
from scipy.stats import randint

#Read Dataset:
player_data = pd.read_csv("TransferMarkt.csv")
player_data.dropna(inplace=True)

#Data Cleaning
league_encoder = LabelEncoder()
player_data['league_encoded'] = league_encoder.fit_transform(player_data['League'])

club_encoder = LabelEncoder()
player_data['club_encoded'] = club_encoder.fit_transform(player_data['club'])

position_encoder = LabelEncoder()
player_data['position_encoded'] = position_encoder.fit_transform(player_data['position'])

player_data = player_data[player_data['Player name'] != 'Jude Bellingham']

#player_data = player_data[player_data['League'] != 'N/A']
print(len(player_data))

#Prepare Features and Target:
X = player_data[['Age', 'league_encoded', 'club_encoded', 'position_encoded']].values
y = player_data['market value'].values


#Scatter plot for Age and market value:
X_market_value = player_data['Age'].values 
y = y.reshape(-1,1)
X_market_value = X_market_value.reshape(-1,1)
plt.scatter(X_market_value, y)
plt.ylabel('Market value')
plt.xlabel('Age')
print(plt.show())


# Group by 'League' and calculate the average market value
average_market_value_by_league = player_data.groupby('League')['market value'].mean()
# Create a bar plot
plt.figure(figsize=(12, 6))
average_market_value_by_league.plot(kind='bar')
plt.xlabel('League')
plt.ylabel('Average Market Value')
plt.title('Average Market Value by league')
plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
plt.tight_layout()  # Ensure all labels are visible
plt.show()

#boxplot
plt.figure(figsize=(12, 6))
plt.boxplot([player_data[player_data['position'] == league]['market value'] for league in player_data['position'].unique()],
            labels=player_data['position'].unique())
plt.xlabel('position')
plt.ylabel('Market Value')
plt.title('Distribution of Market Values by position')
plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
plt.tight_layout()  # Ensure all labels are visible
plt.show()


X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2, random_state=42)



param_dist = {
    'n_estimators': randint(100, 1000),  # Number of trees in the forest
    'max_depth': [None] + list(np.random.randint(1, 30, 20)),  # Maximum depth of the trees
    'min_samples_split': randint(2, 20),  # Minimum number of samples required to split an internal node
    'min_samples_leaf': randint(1, 20),  # Minimum number of samples required to be at a leaf node
    'max_features': ['sqrt', 'log2'],  # Number of features to consider when looking for the best split
}

# Create a Random Forest Regressor
rf_model = RandomForestRegressor(random_state=42)

# Initialize RandomizedSearchCV with the Random Forest Regressor and parameter grid
random_search = RandomizedSearchCV(rf_model, param_distributions=param_dist, n_iter=100, cv=5,
                                   scoring='neg_mean_squared_error', n_jobs=-1, random_state=42)

# Fit the random search to your data
random_search.fit(X_train, y_train.ravel())

# Get the best parameters from the random search
best_params = random_search.best_params_

# Create a Random Forest Regressor with the best parameters
best_rf_model = RandomForestRegressor(**best_params, random_state=42)

# Fit the best model to the training data
best_rf_model.fit(X_train, y_train.ravel())

# Predict with the best model
y_pred = best_rf_model.predict(X_test)



#metrices
print('Random Forest Regression Scores:')
print(f'r2 score: {sklearn.metrics.r2_score(y_test, y_pred)}')
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
rmse = np.sqrt(mean_squared_error(y_pred, y_test))
print("Root Mean Squared Error: {}".format(rmse))



# Get the feature importances as a NumPy array
feature_importances = best_rf_model.feature_importances_

#remove encoded from the feature names
feature_names = ['Age', 'league_encoded', 'club_encoded', 'position_encoded']
feature_names = [name.replace("_encoded", "") for name in feature_names]

# Create a DataFrame 
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})

# Sort the DataFrame by importance in descending order
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Print or visualize the sorted feature importances
print(importance_df)



plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importances')
plt.show()
