import pandas as pd
import numpy as np
import os
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from torch.utils.data import random_split, DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, precision_score, f1_score

# ─── Segment 1: Data Loading and Initial Inspection ───

DATA_PATH = r"C:\Users\Omo Oba\Downloads\New 7022"
columns = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'sex',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
    'income'
]

train_data = pd.read_csv(
    os.path.join(DATA_PATH, "adult.data"),
    names=columns, sep=r'\s*,\s*', engine='python', na_values='?'
)
test_data = pd.read_csv(
    os.path.join(DATA_PATH, "adult.test"),
    names=columns, sep=r'\s*,\s*', engine='python', skiprows=1, na_values='?'
)

# ─── Segment 2: Cleaning & Correlation Analysis ───

# Drop missing rows
train_clean = train_data.dropna().copy()

# Standardize income labels and create binary target
train_clean['income'] = train_clean['income'].str.strip().replace({'<=50K.':'<=50K','>50K.':'>50K'})
train_clean['income_binary'] = train_clean['income'].map({'>50K':1, '<=50K':0})

# Correlation on numeric features + target
numeric_feats = ['age','fnlwgt','education_num','capital_gain','capital_loss','hours_per_week','income_binary']
corr_matrix = train_clean[numeric_feats].corr()
plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix (Training Set)")
plt.tight_layout()
#plt.show()
#print(corr_matrix)

# ─── Segment 3: One‑Hot Encoding & Feature Scaling ───

# Define columns
numeric_cols = ['age','fnlwgt','education_num','capital_gain','capital_loss','hours_per_week']
categorical_cols = ['workclass','education','marital_status','occupation','relationship','race','sex','native_country']

# Clean test set identically
test_clean = test_data.dropna().copy()
test_clean['income'] = test_clean['income'].str.strip().replace({'<=50K.':'<=50K','>50K.':'>50K'})
test_clean['income_binary'] = test_clean['income'].map({'>50K':1, '<=50K':0})

# Split into X/y
X_train = train_clean.drop(['income','income_binary'], axis=1)
y_train = train_clean['income_binary']
X_test  = test_clean.drop(['income','income_binary'], axis=1)
y_test  = test_clean['income_binary']

# Build preprocessor
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_cols),
    ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols)
])

# Fit/transform
X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc  = preprocessor.transform(X_test)

print("Processed shapes – X_train:", X_train_proc.shape, "X_test:", X_test_proc.shape)

# ─── Segment 4: Model Definition (PyTorch) ───

class AdultNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

input_dim = X_train_proc.shape[1]
model = AdultNet(input_dim)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ─── Segment 5: Training Loop with Early Stopping ───

# Build datasets & loaders
train_ds = TensorDataset(
    torch.tensor(X_train_proc, dtype=torch.float32),
    torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
)
val_frac = 0.2
val_size = int(val_frac * len(train_ds))
train_size = len(train_ds) - val_size
train_subset, val_subset = random_split(train_ds, [train_size, val_size])

train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_subset,   batch_size=64, shuffle=False)
test_ds = TensorDataset(
    torch.tensor(X_test_proc, dtype=torch.float32),
    torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)
)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

num_epochs    = 20
learning_rate = 1e-3
weight_decay  = 1e-5
best_val_loss = float('inf')
best_model    = None
patience      = 3
stale_epochs  = 0

for epoch in range(1, num_epochs+1):
    # Train
    model.train()
    train_loss = 0.0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * xb.size(0)
    train_loss /= len(train_loader.dataset)

    # Validate
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            val_loss += criterion(model(xb), yb).item() * xb.size(0)
    val_loss /= len(val_loader.dataset)

    print(f"Epoch {epoch} – Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        stale_epochs = 0
        torch.save(model.state_dict(), "best_model.pth")
    else:
        stale_epochs += 1
        if stale_epochs >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

# Load best model
model.load_state_dict(torch.load("best_model.pth"))

# ─── Segment 6: Final Evaluation ───

model.eval()
running_loss = 0.0
all_preds = []
all_labels = []

with torch.no_grad():
    for xb, yb in test_loader:
        preds = model(xb)
        running_loss += criterion(preds, yb).item() * xb.size(0)
        all_preds.extend((preds >= 0.5).int().squeeze().tolist())
        all_labels.extend(yb.int().squeeze().tolist())

test_loss = running_loss / len(test_loader.dataset)
print(f"\nTest Loss: {test_loss:.4f}")

# Classification metrics
#print("\nClassification Report:")
#print(classification_report(all_labels, all_preds, target_names=['<=50K','>50K']))

prec = precision_score(all_labels, all_preds)
f1   = f1_score(all_labels, all_preds)
print(f"Precision (>50K): {prec:.4f}")
print(f"F1 Score   (>50K): {f1:.4f}")

from sklearn.metrics import accuracy_score  # Ensure this import is present

# Calculate accuracy
accuracy = accuracy_score(all_labels, all_preds)
print(f"NNClass Accuracy     : {accuracy:.4f}")
print(f"NNClass Precision    : {prec:.4f}")
print(f"NNClass F1 Score      : {f1:.4f}")



# ─── Segment 7: Random Forest Classification ───

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score, classification_report

# 1) Instantiate the Random Forest
rf_model = RandomForestClassifier(
    n_estimators=300,      # number of trees
    max_depth=None,
    min_samples_split=10,
    min_samples_leaf=5,  
    random_state=42,
    n_jobs=-1              # use all CPU cores
)

# 2) Fit on the training data
rf_model.fit(X_train_proc, y_train)

# 3) Predict on the test set
y_pred = rf_model.predict(X_test_proc)

# 4) Compute metrics
accuracy  = accuracy_score(y_test,  y_pred)
precision = precision_score(y_test, y_pred)
f1        = f1_score(y_test,    y_pred)

# 5) Print results
print(f"Random Forest Test Accuracy : {accuracy:.4f}")
print(f"Random Forest Precision     : {precision:.4f}")
print(f"Random Forest F1 Score      : {f1:.4f}")

# 6) (Optional) Detailed classification report
#print("\nClassification Report:")
#print(classification_report(y_test, y_pred, target_names=['<=50K', '>50K']))



from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score, classification_report

# ─── XGBoost Classification ───

# 1) Instantiate the model
xgb_model = XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False,
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1
)

# 2) Fit on the training data
xgb_model.fit(X_train_proc, y_train)

# 3) Predict on the test set
y_pred = xgb_model.predict(X_test_proc)

# 4) Compute metrics
accuracy  = accuracy_score(y_test,  y_pred)
precision = precision_score(y_test, y_pred)
f1        = f1_score(y_test,    y_pred)

print(f"XGBoost Test Accuracy : {accuracy:.4f}")
print(f"XGBoost Precision     : {precision:.4f}")
print(f"XGBoost F1 Score      : {f1:.4f}")

# 5) (Optional) Detailed classification report
#print("\nClassification Report:")
#print(classification_report(y_test, y_pred, target_names=['<=50K', '>50K']))


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, f1_score, classification_report

# ─── Logistic Regression Classification ───

# 1) Instantiate the logistic regression model
logreg = LogisticRegression(
    solver='lbfgs',      # robust solver for small-to-medium datasets
    max_iter=1000,       # ensure convergence
    random_state=42,
    n_jobs=-1            # use all CPU cores
)

# 2) Fit on the training data
logreg.fit(X_train_proc, y_train)

# 3) Predict on the test set
y_pred = logreg.predict(X_test_proc)

# 4) Compute metrics
accuracy  = accuracy_score(y_test,  y_pred)
precision = precision_score(y_test, y_pred)
f1        = f1_score(y_test,    y_pred)

print(f"Logistic Regression Test Accuracy : {accuracy:.4f}")
print(f"Logistic Regression Precision     : {precision:.4f}")
print(f"Logistic Regression F1 Score      : {f1:.4f}")

# 5) (Optional) Detailed classification report
#print("\nClassification Report:")
#print(classification_report(y_test, y_pred, target_names=['<=50K', '>50K']))


from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, f1_score, classification_report

# ─── Support Vector Classification ───

# 1) Instantiate the SVC model
svc = SVC(
    kernel='rbf',      # radial basis function kernel
    C=1.0,             # regularization parameter
    gamma='scale',     # kernel coefficient
    random_state=42
)

# 2) Fit on the training data
svc.fit(X_train_proc, y_train)

# 3) Predict on the test set
y_pred = svc.predict(X_test_proc)

# 4) Compute metrics
accuracy  = accuracy_score(y_test,  y_pred)
precision = precision_score(y_test, y_pred)
f1        = f1_score(y_test,    y_pred)

print(f"SVC Test Accuracy : {accuracy:.4f}")
print(f"SVC Precision     : {precision:.4f}")
print(f"SVC F1 Score      : {f1:.4f}")

# 5) (Optional) Detailed classification report
#print("\nClassification Report:")
#print(classification_report(y_test, y_pred, target_names=['<=50K', '>50K']))



# ─── Segment 7: Baseline Model Performance Plot (custom x‑axes) ───

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.ensemble    import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm         import SVC
from sklearn.neural_network import MLPClassifier
from xgboost             import XGBClassifier

# Define the five baseline models
models = {
    'XGBoost Classification': XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ),
    'Support Vector Classification': SVC(
        kernel='rbf', C=1.0, gamma='scale', random_state=42
    ),
    'Random Forest Classification': RandomForestClassifier(
        n_estimators=100, random_state=42, n_jobs=-1
    ),
    'Neural Network Classification': MLPClassifier(
        hidden_layer_sizes=(32,16),
        activation='relu', solver='adam',
        max_iter=200, random_state=42
    ),
    'Logistic Regression': LogisticRegression(
        solver='lbfgs', max_iter=1000,
        random_state=42, n_jobs=-1
    )
}

# 5‑fold CV for accuracy & precision
names       = list(models.keys())
acc_means   = []; acc_stds   = []
prec_means  = []; prec_stds  = []

for name, model in models.items():
    acc = cross_val_score(model, X_train_proc, y_train,
                          cv=5, scoring='accuracy',  n_jobs=-1)
    prc = cross_val_score(model, X_train_proc, y_train,
                          cv=5, scoring='precision', n_jobs=-1)
    acc_means.append(acc.mean() * 100)
    acc_stds.append(acc.std() * 100)
    prec_means.append(prc.mean() * 100)
    prec_stds.append(prc.std() * 100)

# Plot side‑by‑side error‑bar charts with custom x‑axes
y_pos = np.arange(len(names))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

# Accuracy panel: x-axis 80–90
ax1.errorbar(acc_means, y_pos, xerr=acc_stds, fmt='o', capsize=5)
ax1.set_xlim(82, 88)
ax1.set_xticks(np.arange(80, 91, 2))
ax1.set_title('Accuracy')
ax1.set_xlabel('Mean Accuracy (%)')
ax1.set_yticks(y_pos)
ax1.set_yticklabels(names)
ax1.invert_yaxis()

# Precision panel: x-axis 60–80
ax2.errorbar(prec_means, y_pos, xerr=prec_stds, fmt='o', capsize=5)
ax2.set_xlim(65, 80)
ax2.set_xticks(np.arange(60, 81, 5))
ax2.set_title('Precision')
ax2.set_xlabel('Mean Precision (%)')
ax2.set_yticks(y_pos)
ax2.set_yticklabels(names)
ax2.invert_yaxis()

plt.tight_layout()
plt.show()
