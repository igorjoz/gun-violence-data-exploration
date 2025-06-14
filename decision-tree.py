import os
import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

# Create directory for figures
fig_dir = 'figures'
os.makedirs(fig_dir, exist_ok=True)

# 1. Load the data
df = pd.read_csv('incidents.csv')

# 2. Create binary target: mortality = 1 if n_killed > 0
df['mortality'] = (df['n_killed'] > 0).astype(int)

# 3. Compute average participant age feature
def parse_avg_age(age_str):
    if pd.isna(age_str) or age_str == '':
        return np.nan
    parts = age_str.split('||')
    ages = []
    for p in parts:
        try:
            ages.append(int(p.split('::')[1]))
        except:
            continue
    return np.mean(ages) if ages else np.nan

df['avg_age'] = df['participant_age'].apply(parse_avg_age)

# 4. Compute gender counts per incident
# participant_gender format: '0::Male||1::Female||...'
def parse_gender_counts(gender_str):
    male = 0
    female = 0
    if pd.isna(gender_str) or gender_str == '':
        return pd.Series({'male_count': 0, 'female_count': 0})
    parts = gender_str.split('||')
    for p in parts:
        try:
            val = p.split('::')[1]
            if 'Male' in val:
                male += 1
            elif 'Female' in val:
                female += 1
        except:
            continue
    return pd.Series({'male_count': male, 'female_count': female})

gender_counts = df['participant_gender'].apply(parse_gender_counts)
df = pd.concat([df, gender_counts], axis=1)

# 5. Select features (drop gun_type)
features = ['n_injured', 'n_guns_involved', 'avg_age', 'male_count', 'female_count', 'state']
X = df[features]
y = df['mortality']

# 6. Fill missing values
X = X.fillna({
    'n_injured': 0,
    'n_guns_involved': 0,
    'avg_age': X['avg_age'].median(),
    'male_count': 0,
    'female_count': 0,
    'state': 'Unknown'
})

# 7. Encode state into sparse one-hot
categorical = ['state']
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
X_cat = encoder.fit_transform(X[categorical])

# 8. Numeric features to sparse
numeric_cols = ['n_injured', 'n_guns_involved', 'avg_age', 'male_count', 'female_count']
X_num = sparse.csr_matrix(X[numeric_cols].values)

# 9. Combine numeric and categorical into sparse matrix
X_sparse = sparse.hstack([X_num, X_cat], format='csr')

# 10. Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_sparse, y, test_size=0.3, stratify=y, random_state=42
)

# 11. Balance classes with SMOTE (use dense for SMOTE)
X_train_dense = X_train.toarray()
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_dense, y_train)
X_train_res_sparse = sparse.csr_matrix(X_train_res)

# 12. Hyperparameter tuning
param_grid = {
    'max_depth': [5, 7, 9, None],
    'min_samples_leaf': [1, 5, 10],
    'class_weight': ['balanced']
}
clf_base = DecisionTreeClassifier(random_state=42)
grid = GridSearchCV(
    clf_base, param_grid, cv=3, scoring='recall', n_jobs=1
)
grid.fit(X_train_res_sparse, y_train_res)
clf = grid.best_estimator_
print(f"Best params: {grid.best_params_}")

# 13. Train final model
clf.fit(X_train_res_sparse, y_train_res)

# 14. Predict probabilities and compute ROC Predict probabilities and compute ROC
y_proba = clf.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_proba)

# 15. Threshold selection
def select_threshold(fpr, tpr, thr):
    spec = 1 - fpr
    candidates = [(r, s, t) for r, s, t in zip(tpr, spec, thr) if r >= 0.80 and s >= 0.60]
    if candidates:
        return max(c[2] for c in candidates)
    iscores = [r * s for r, s in zip(tpr, spec)]
    return thr[np.argmax(iscores)]

thresh = select_threshold(fpr, tpr, thresholds)
print(f"Selected threshold: {thresh:.2f}")

# 16. Final predictions
y_pred = (y_proba >= thresh).astype(int)

# 17. Evaluation
print("=== Classification Report ===")
print(classification_report(y_test, y_pred))
print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))
print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.3f}")

# 18. Save ROC curve
roc_path = os.path.join(fig_dir, 'roc_curve.png')
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc_score(y_test, y_proba):.3f}')
plt.plot([0, 1], [0, 1], '--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.savefig(roc_path)
plt.close()

# 19. Save feature importances
feature_names = numeric_cols + list(encoder.get_feature_names_out(categorical))
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1][:10]
fi_path = os.path.join(fig_dir, 'feature_importances.png')
plt.figure(figsize=(10, 6))
plt.barh(range(len(indices)), importances[indices][::-1])
plt.yticks(range(len(indices)), [feature_names[i] for i in indices[::-1]])
plt.xlabel('Importance')
plt.title('Top 10 Feature Importances')
plt.savefig(fi_path)
plt.close()

# 20. Save deeper tree visualization (depth=5)
tree_path = os.path.join(fig_dir, 'decision_tree_depth5.png')
plt.figure(figsize=(30, 15))
plot_tree(
    clf,
    feature_names=feature_names,
    class_names=['NoDeath', 'Death'],
    filled=True,
    max_depth=5,
    fontsize=6
)
plt.savefig(tree_path)
plt.close()

# 21. Export tree rules
txt = export_text(clf, feature_names=feature_names)
rules_path = os.path.join(fig_dir, 'tree_rules.txt')
with open(rules_path, 'w') as f:
    f.write(txt)

print(f"Saved figures and tree rules in '{fig_dir}/'")

# 22. Prepare dense data for cross-validation
X_dense = X_sparse.toarray()

# 23. 10-Fold Cross-Validation with ROC plots
from sklearn.model_selection import StratifiedKFold
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import recall_score

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
recalls = []
specificities = []

for fold, (train_idx, test_idx) in enumerate(cv.split(X_dense, y), start=1):
    X_tr, X_te = X_dense[train_idx], X_dense[test_idx]
    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
    # Pipeline for this fold
    model = DecisionTreeClassifier(
        **grid.best_params_,
        random_state=42
    )
    pipe = ImbPipeline([
        ('smote', SMOTE(random_state=42)),
        ('clf', model)
    ])
    pipe.fit(X_tr, y_tr)
    y_pred = pipe.predict(X_te)
    y_proba = pipe.predict_proba(X_te)[:, 1]
    # Metrics
    rec = recall_score(y_te, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_te, y_pred).ravel()
    spec = tn / (tn + fp)
    recalls.append(rec)
    specificities.append(spec)
    # ROC for fold
    fpr_i, tpr_i, _ = roc_curve(y_te, y_proba)
    plt.figure(figsize=(8,6))
    plt.plot(fpr_i, tpr_i, label=f'Fold {fold} (AUC={roc_auc_score(y_te, y_proba):.3f})')
    plt.plot([0,1], [0,1], '--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - Fold {fold}')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(fig_dir, f'roc_fold_{fold}.png'))
    plt.close()

# Summary CV results
print("10-Fold CV Results:")
print(f"Recall: {np.mean(recalls):.3f} ± {np.std(recalls):.3f}")
print(f"Specificity: {np.mean(specificities):.3f} ± {np.std(specificities):.3f}")

# 24. Save decision tree plots for each fold
for fold in range(1, 11):
    # load fold-specific model from earlier loop? if needed, refit or store models
    # Here, re-train on each fold to plot tree
    train_idx, test_idx = list(cv.split(X_dense, y))[fold-1]
    X_tr, y_tr = X_dense[train_idx], y.iloc[train_idx]
    # train model
    model = DecisionTreeClassifier(
        **grid.best_params_,
        random_state=42
    )
    # no SMOTE for plotting tree structure (use original data)
    model.fit(X_tr, y_tr)
    # plot tree
    plt.figure(figsize=(20, 10))
    plot_tree(
        model,
        feature_names=feature_names,
        class_names=['NoDeath', 'Death'],
        filled=True,
        max_depth=3,
        fontsize=8
    )
    tree_fold_path = os.path.join(fig_dir, f'decision_tree_fold_{fold}.png')
    plt.savefig(tree_fold_path)
    plt.close()

print(f"Saved decision tree plots for each fold in '{fig_dir}/'")
