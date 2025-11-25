


import pandas as pd

# CSV ko load karna
data = pd.read_csv('data/candidate_profiles.csv')

print("Data Loaded Successfully!")
print(data.head())


print(data.isnull().sum())

print(data.dtypes)

cols_to_drop = ['Candidate_ID', 'Name']
data = data.drop(columns=[col for col in cols_to_drop if col in data.columns])

print(data.columns)


from sklearn.preprocessing import LabelEncoder

label = LabelEncoder()

categorical_columns = ['education', 'previous_role', 'skills', 'certifications']

for col in categorical_columns:
    if col in data.columns:

        data[col] = label.fit_transform(data[col])

        print("\nEncoded Data:")
print(data.head())


# Step 3: Create Suitability Score (Target Column)
score_columns = [
    'communication_score',
    'domain_knowledge_score',
    'behaviour_score',
    'trust_score'
]

data['suitability_score'] = data[score_columns].mean(axis=1)

print("\nSuitability Score Added:")
print(data[['suitability_score']].head())

# Step 4: Prepare Data for Model Training
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Features (X) = all columns except suitability_score
X = data.drop(columns=['suitability_score', 'name'])  # name drop karna zaroori hai

# Target (y)
y = data['suitability_score']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nData Split Done!")

# Step 5: Train ML Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("\nModel Training Completed!")

# Step 6: Test Model
predictions = model.predict(X_test)
print("\nSample Predictions:")
print(predictions[:5])























