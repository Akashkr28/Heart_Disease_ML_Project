from preprocessing import preprocess_data
from ml_models import train_ml_models
from evaluation import evaluate_model

from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

import matplotlib.pyplot as plt

# Store result for plotting

results_summary = {}

# 1. Load & Preprocess data

X_train, X_test, y_train, y_test = preprocess_data("data/heart_cleveland_upload.csv")

# 2. Baseline ML Models

models = train_ml_models(X_train, y_train)

for name, model in models.items():
    results = evaluate_model(model, X_test, y_test)
    print(name, results)

# 3. PCA Feature Learning

pca = PCA(n_components=8)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

rf_pca = RandomForestClassifier(n_estimators=100, random_state=42)
rf_pca.fit(X_train_pca, y_train)

pca_results = evaluate_model(rf_pca, X_test_pca, y_test)
results_summary["PCA + Random Forest"] = pca_results["Accuracy"]

print("\n=== PCA + Random Forest ===")
print(pca_results)

# 4. RFE Feature Selection

rfe = RFE(
    estimator=LogisticRegression(max_iter=1000),
    n_features_to_select=8
)

X_train_rfe = rfe.fit_transform(X_train, y_train)
X_test_rfe = rfe.transform(X_test)

svm_rfe = SVC(probability=True)
svm_rfe.fit(X_train_rfe, y_train)

rfe_results = evaluate_model(svm_rfe, X_test_rfe, y_test)
results_summary["RFE + SVM"] = rfe_results["Accuracy"]

print("\n=== RFE + SVM ===")
print(rfe_results)

# 5. Ensemble (Proposed Model)

voting = VotingClassifier(
    estimators=[
        ("rf", RandomForestClassifier(n_estimators=100, random_state=42)),
        ("svm", SVC(probability=True)),
        ("lr", LogisticRegression(max_iter=1000))
    ],
    voting="soft"
)

voting.fit(X_train_pca, y_train)

voting_results = evaluate_model(voting, X_test_pca, y_test)
results_summary["EFEM-HDP (Voting Classifier)"] = voting_results["Accuracy"]

print("\n=== Proposed EFEM-HDP (Voting Classifier) ===")
print(voting_results)

# 6. Accuracy Comparison Bar Chart

plt.figure()
plt.bar(results_summary.keys(), results_summary.values())
plt.title("Accuracy Comparison of Models")
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.xticks(rotation=45, ha="right")

accuracies = list(results_summary.values())
for i, v in enumerate(accuracies):
    plt.text(i, v + 0.01, f"{v:.2f}", ha="center")

plt.tight_layout()
plt.show()