import os
import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

# Create directory for figures
fig_dir = 'figures'
os.makedirs(fig_dir, exist_ok=True)

# 1. Load data
df = pd.read_csv('incidents.csv')
# 2. Target
df['mortality'] = (df['n_killed'] > 0).astype(int)

# 3. Compute avg_age
def parse_avg_age(s):
    if pd.isna(s) or not s:
        return np.nan
    ages = []
    for p in s.split('||'):
        try:
            ages.append(int(p.split('::')[1]))
        except:
            pass
    return np.mean(ages) if ages else np.nan

df['avg_age'] = df['participant_age'].apply(parse_avg_age)

# 4. Gender counts
def parse_gender_counts(s):
    m = f = 0
    if isinstance(s, str):
        for p in s.split('||'):
            if 'Male' in p:
                m += 1
            elif 'Female' in p:
                f += 1
    return pd.Series({'male_count': m, 'female_count': f})

gender_df = df['participant_gender'].apply(parse_gender_counts)
df = pd.concat([df, gender_df], axis=1)

# 5. Encode state once
states = df['state'].fillna('Unknown').values.reshape(-1, 1)
state_enc = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
X_state = state_enc.fit_transform(states)
state_cols = state_enc.get_feature_names_out(['state'])

# 6. Feature subsets to test
tests = {
    'all': ['n_injured','n_guns_involved','avg_age','male_count','female_count','state'],
    'top1': ['male_count'],
    'top2': ['male_count','n_injured'],
    'top3': ['male_count','n_injured','female_count'],
    'top5': ['male_count','n_injured','female_count','n_guns_involved','avg_age']
}

y = df['mortality']

for name, feats in tests.items():
    print(f"\n=== RandomForest with {name} features ===")
    # numeric features
    num_feats = [c for c in feats if c != 'state']
    X_num = df[num_feats].fillna(0).values
    X_num_sp = sparse.csr_matrix(X_num)
    # combine
    if 'state' in feats:
        X_sp = sparse.hstack([X_num_sp, X_state], format='csr')
        feature_names = num_feats + list(state_cols)
    else:
        X_sp = X_num_sp
        feature_names = num_feats.copy()

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_sp, y, test_size=0.3, stratify=y, random_state=42
    )
    # SMOTE
    X_train_dense = X_train.toarray()
    sm = SMOTE(random_state=42)
    X_tr_res, y_tr_res = sm.fit_resample(X_train_dense, y_train)
    X_tr_res_sp = sparse.csr_matrix(X_tr_res)

    # RandomForest hyperparam tuning
    rf_grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        {'n_estimators':[100,200], 'max_depth':[5,7,9,None], 'min_samples_leaf':[1,5], 'class_weight':['balanced']},
        cv=3, scoring='recall', n_jobs=-1
    )
    rf_grid.fit(X_tr_res_sp, y_tr_res)
    rf = rf_grid.best_estimator_
    print(f"Best params for RandomForest: {rf_grid.best_params_}")

    # Evaluate on test
    y_prob = rf.predict_proba(X_test)[:,1]
    fpr, tpr, thr = roc_curve(y_test, y_prob)
    opt_thr = thr[np.argmax(tpr * (1 - fpr))]
    y_pred = (y_prob >= opt_thr).astype(int)
    rec = recall_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    spec = tn / (tn + fp)
    auc = roc_auc_score(y_test, y_prob)
    print(f"RF AUC={auc:.3f} Recall={rec:.3f} Specificity={spec:.3f}")

    # ROC plot
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
    plt.plot([0,1],[0,1],'--', color='gray')
    plt.title(f'RandomForest ROC - {name}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f'RF_roc_{name}.png'))
    plt.close()

    # Feature importances plot
importances = rf.feature_importances_
idx = np.argsort(importances)[::-1][:10]
plt.figure(figsize=(10, 6))  # Increased figure size for label visibility
plt.barh(np.arange(len(idx)), importances[idx][::-1])
plt.yticks(np.arange(len(idx)), [feature_names[i] for i in idx[::-1]], fontsize=8)
plt.title(f'RandomForest Top 10 Importances - {name}')
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, f'RF_imps_{name}.png'))
plt.close()()

print('\nExperiment complete.')
