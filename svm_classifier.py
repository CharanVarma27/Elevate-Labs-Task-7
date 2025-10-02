import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC # Support Vector Classifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Load, Preprocess, and Split Data ---
df = pd.read_csv('breast-cancer.csv')

# Drop irrelevant 'id' column
df = df.drop('id', axis=1)

# Encode the target variable ('diagnosis'): M=1 (Malignant), B=0 (Benign)
le = LabelEncoder()
df['diagnosis'] = le.fit_transform(df['diagnosis']) # M becomes 1, B becomes 0

# Define Features (X) and Target (y)
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

# Scaling is mandatory for SVM!
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
print("Data preparation complete (Scaled and Split).\n")

# --- 2. Train and Evaluate Linear SVM ---
print("--- Linear Kernel SVM ---")
# Use kernel='linear' for Linear SVM
linear_model = SVC(kernel='linear', C=1, random_state=42)
linear_model.fit(X_train, y_train)
linear_pred = linear_model.predict(X_test)
linear_accuracy = accuracy_score(y_test, linear_pred)

print(f"Linear SVM Accuracy: {linear_accuracy:.4f}")
print("Linear SVM Classification Report:\n", classification_report(y_test, linear_pred))

# --- 3. Train and Evaluate RBF SVM ---
print("\n--- RBF Kernel SVM ---")
# Use kernel='rbf' (default) for RBF SVM
rbf_model = SVC(kernel='rbf', C=1, gamma='scale', random_state=42)
rbf_model.fit(X_train, y_train)
rbf_pred = rbf_model.predict(X_test)
rbf_accuracy = accuracy_score(y_test, rbf_pred)

print(f"RBF SVM Accuracy: {rbf_accuracy:.4f}")
print("RBF SVM Classification Report:\n", classification_report(y_test, rbf_pred))

# --- 4. Visualization (RBF Confusion Matrix) ---
# Visualizing the confusion matrix for the better performing model (usually RBF)
cm = confusion_matrix(y_test, rbf_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
            xticklabels=['Benign (0)', 'Malignant (1)'], 
            yticklabels=['Benign (0)', 'Malignant (1)'])
plt.title('RBF SVM Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
