"""Worked-example numbers quoted in learn/08_svm_xgboost.md (train and valid of split C only).

Run from repo root: PYTHONPATH=. .venv/Scripts/python -W ignore scripts/lesson08_examples.py
"""
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits
from src.train_classical import load_windows, GENRES
from src.models import svm_pipeline, xgb_model
X_tr, y_tr = load_windows("C", "train"); X_va, y_va = load_windows("C", "valid")
Xs = StandardScaler().fit(X_tr).transform(X_tr)
# 1. RBF kernel values for typical pairs
rng = np.random.default_rng(0); i, j = rng.integers(0, len(Xs), (2, 5000))
d2 = ((Xs[i] - Xs[j]) ** 2).sum(1); d2 = d2[i != j]
same = (y_tr.to_numpy()[i] == y_tr.to_numpy()[j])[i != j]
print("median sq dist all pairs %.1f, same-genre %.1f, diff-genre %.1f" % (np.median(d2), np.median(d2[same]), np.median(d2[~same])))
for g in [0.001, 1/57, 0.01, 0.1]:
    print("gamma %.4f: K at median d2 = %.4g ; 10th pct d2 %.1f -> %.4g" % (g, np.exp(-g*np.median(d2)), np.percentile(d2,10), np.exp(-g*np.percentile(d2,10))))
# 2. Gini: depth-1 tree
counts = y_tr.value_counts().reindex(GENRES); p = counts / counts.sum()
print("train window counts", counts.to_dict()); print("root gini %.4f" % (1 - (p**2).sum()))
st = DecisionTreeClassifier(max_depth=1, random_state=0).fit(X_tr, y_tr)
t = st.tree_; f = X_tr.columns[t.feature[0]]
print("stump: %s <= %.4g ; left n=%d gini %.4f, right n=%d gini %.4f" % (f, t.threshold[0], t.n_node_samples[1], t.impurity[1], t.n_node_samples[2], t.impurity[2]))
left = X_tr[f] <= t.threshold[0]
print("left top genres", y_tr[left].value_counts().head(3).to_dict(), "right top", y_tr[~left].value_counts().head(3).to_dict())
# 3. best SVM: support vectors and train fit
with threadpool_limits(1):
    svm = svm_pipeline(C=100, gamma=0.001).fit(X_tr, y_tr)
print("SVM C=100 g=0.001: support vectors %d of %d, train window acc %.3f, valid window acc %.3f" % (svm[-1].support_.size, len(X_tr), (svm.predict(X_tr) == y_tr).mean(), (svm.predict(X_va) == y_va).mean()))
print("  predict vs argmax(predict_proba) disagree on %d valid windows" % (svm.predict(X_va) != svm.classes_[svm.predict_proba(X_va).argmax(1)]).sum())
svm2 = svm_pipeline(C=100, gamma=0.1).fit(X_tr, y_tr)
print("SVM C=100 g=0.1: support vectors %d, train window acc %.3f" % (svm2[-1].support_.size, (svm2.predict(X_tr) == y_tr).mean()))
svm3 = svm_pipeline(C=0.1, gamma=0.1).fit(X_tr, y_tr)
print("SVM C=0.1 g=0.1: support vectors %d, train window acc %.3f" % (svm3[-1].support_.size, (svm3.predict(X_tr) == y_tr).mean()))
# 4. trees ignore scaling
yc = pd.Categorical(y_tr, categories=GENRES).codes
kw = dict(n_estimators=54, max_depth=3, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8)
a = xgb_model(**kw).fit(X_tr, yc).predict_proba(X_va)
sc = StandardScaler().fit(X_tr)
b = xgb_model(**kw).fit(sc.transform(X_tr), yc).predict_proba(sc.transform(X_va))
print("xgb raw vs scaled: max |prob diff| %.2e, argmax disagreements %d" % (np.abs(a-b).max(), (a.argmax(1)!=b.argmax(1)).sum()))
print("xgb first-round root splits:")
df = xgb_model(**kw).fit(X_tr, yc).get_booster().trees_to_dataframe()
print(df[(df.Node==0) & (df.Tree<10)][["Tree","Feature","Split","Gain"]].assign(genre=GENRES).to_string(index=False))
