import pickle
import scikit-learn as sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load dataset (Ensure your dataset is loaded correctly)
df = pd.read_csv("Paitients_Files_Train.csv")  # Adjust path if needed

# Preprocessing
df.drop(columns=["ID"], inplace=True)  # Drop ID column
df["Sepssis"] = df["Sepssis"].map({"Negative": 0, "Positive": 1})  # Convert to binary

# Split Data
X = df.drop(columns=["Sepssis"])
y = df["Sepssis"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Save model & scaler
with open("sepsis_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
