"""Model factories for the classical (window-feature) models.

Every model that needs scaled inputs gets its StandardScaler inside a Pipeline,
so the scaler is refit on the training part of each fit and never sees valid/test.
Pipelines name their final step "clf", so grid keys look like "clf__C".
"""
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from src.utils import load_config

CFG = load_config()


def dummy_most_frequent() -> DummyClassifier:
    """Floor: always predicts the most common training genre."""
    return DummyClassifier(strategy="most_frequent")


def dummy_stratified(seed: int = CFG["seed"]) -> DummyClassifier:
    """Floor: random genre per window, drawn with the training class frequencies."""
    return DummyClassifier(strategy="stratified", random_state=seed)


def _scaled(clf) -> Pipeline:
    return Pipeline([("scale", StandardScaler()), ("clf", clf)])


def knn_pipeline(n_neighbors: int = 5, weights: str = "uniform") -> Pipeline:
    """Standardise, then k-nearest neighbours (Euclidean)."""
    return _scaled(KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights))


def logreg_pipeline(C: float = 1.0) -> Pipeline:
    """Standardise, then multinomial logistic regression with an L2 penalty."""
    return _scaled(LogisticRegression(C=C, max_iter=CFG["logreg_max_iter"]))


def svm_pipeline(C: float = 1.0, gamma: str | float = "scale", seed: int = CFG["seed"]) -> Pipeline:
    """Standardise, then RBF SVM; probability=True for soft voting."""
    # ponytail: probability=True is deprecated in sklearn 1.9 (gone in 1.11); fine on our pin 1.9.1,
    # switch to CalibratedClassifierCV(SVC(), ensemble=False) when upgrading
    return _scaled(SVC(kernel="rbf", C=C, gamma=gamma, probability=True, random_state=seed))


def xgb_model(seed: int = CFG["seed"], **params) -> XGBClassifier:
    """Gradient-boosted trees (no scaling needed). Labels must be integer-encoded."""
    return XGBClassifier(random_state=seed, n_jobs=-1, **params)


# name used on the CLI / in result files -> (factory, grid key in config or None)
MODELS = {
    "dummy_most_frequent": (dummy_most_frequent, None),
    "dummy_stratified": (dummy_stratified, None),
    "knn": (knn_pipeline, "knn"),
    "logreg": (logreg_pipeline, "logreg"),
}
