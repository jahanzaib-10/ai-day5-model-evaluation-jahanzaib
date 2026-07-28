import os
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# 1. Data load karein
X_train = pd.read_csv('./data/processed/train_processed.csv')
X_test = pd.read_csv('./data/processed/test_processed.csv')

y_train = X_train['Survived']
X_train = X_train.drop(columns=['Survived'])

# 2. Sirf numeric aur common columns rakhein
common_cols = [col for col in X_train.columns if col in X_test.columns and col != 'Survived']
X_train = X_train[common_cols].select_dtypes(include=['number'])

# 3. Model train karein
print("Training the model with matched features...")
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# 4. Save model
os.makedirs('./models', exist_ok=True)
with open('./models/best_model.pkl', 'wb') as f:
    pickle.dump(clf, f)

print("Model successfully retrained and saved to models/best_model.pkl!")