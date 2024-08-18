import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from pyswarm import pso
from sklearn.preprocessing import StandardScaler
from joblib import dump
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load dataset
df = pd.read_csv('dataset.csv')

# Split the data into features and target
X = df.drop(columns=['target'])
y = df['target']

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# PSO Feature Selection
def evaluate_particles(particles):
    selected_features = [int(i) for i in particles]
    selected_columns = X_train[:, selected_features]
    
    model = GaussianNB()
    model.fit(selected_columns, y_train)
    y_pred = model.predict(X_test[:, selected_features])
    
    accuracy = accuracy_score(y_test, y_pred)
    return -accuracy  # Minimize negative accuracy for PSO

# Initialize PSO parameters
num_particles = 10  # Number of particles
num_features = X_train.shape[1]  # Number of features
lb = [0] * num_features  # Lower bound of 0 for each feature
ub = [1] * num_features  # Upper bound of 1 for each feature

# Run PSO
best_features, best_score = pso(evaluate_particles, lb, ub, swarmsize=num_particles, maxiter=100)

# Get selected features
selected_features = [int(i) for i in best_features]

# Train the final model with the selected features
X_train_selected = X_train[:, selected_features]
X_test_selected = X_test[:, selected_features]

final_model = GaussianNB()
final_model.fit(X_train_selected, y_train)

# Save the model and scaler
dump(final_model, 'heart_disease_model.joblib')
dump(scaler, 'scaler.joblib')
dump(selected_features, 'selected_features.joblib')

# Predictions and Metrics
y_pred = final_model.predict(X_test_selected)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Print metrics
print(f"Model Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"Confusion Matrix:\n{conf_matrix}")

# Plotting the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title(f'Confusion Matrix\nAccuracy: {accuracy:.2f} | Precision: {precision:.2f} | Recall: {recall:.2f} | F1 Score: {f1:.2f}')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('confusion_matrix.png')
plt.show()

# Save metrics to a file
metrics = {
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1_score': f1,
    'confusion_matrix': conf_matrix.tolist()
}

with open('metrics.txt', 'w') as f:
    f.write(str(metrics))
