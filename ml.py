
import pandas as pd

# Sample data

df = pd.read_csv(r"C:\Users\fathi\OneDrive\Desktop\ICT\Social_Network_Ads.csv")

# Feature selection (optional)
features = ['Gender', 'Age', 'EstimatedSalary']
X = df[features]

# Convert categorical features to numerical (optional)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X['Gender'] = le.fit_transform(X['Gender'])

# Target variable
y = df['Purchased']

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Machine learning model (Random Forest)
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Save the model for later use
import pickle
with open('model.pkl', 'wb') as f:
    pickle.dump(clf, f)

print("Model created and saved successfully!")

