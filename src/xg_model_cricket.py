import numpy as np
import pandas as pd
import shap
import xgboost
from matplotlib import pyplot as plt
from scipy.stats import uniform
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBClassifier

from src.dataset_utils import load_john_doe


class AttrSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attributes):
        self.attributes = attributes

    # Needs attribute parameter from sklearn base class
    def fit(self, attributes):
        return self

    def transform(self, X):
        return X[self.attributes]


def full_preprocessing_pipeline(X_train, X_test, categorical_features, numeric_features, fit=True):
    categorical_attr_pipeline = Pipeline([
        ('selector', AttrSelector(categorical_features)),
        ('one_hot_encoder', OneHotEncoder())
    ])

    numerical_attr_pipeline = Pipeline([
        ('selector', AttrSelector(numeric_features)),
        ('std_scaler', StandardScaler())
    ])

    combined_pipeline = ColumnTransformer(transformers=[('num', numerical_attr_pipeline, numeric_features),
                                                        ('cat', categorical_attr_pipeline, categorical_features)])

    if fit:
        X_train_preprocessed = combined_pipeline.fit_transform(X_train)
        X_test_preprocessed = combined_pipeline.transform(X_test)

        return X_train_preprocessed, X_test_preprocessed, combined_pipeline

    return combined_pipeline


def hyperparameter_opt(classifier, hyperparameter_dist, y_train, scoring="f1", n_iter=10):
    """
    Runs randomized hyperparameter search on preprocessed data for specified classifer.

    Arguments:
    ----------
    classifier (sklearn classifier object)
        - classifer we want to optimize
    hyperparameter_dist (dict or list of dicts)
        - dictionary with parameters names (string) as keys and distributions or lists of hyperparameters to try
    X_train (pd.DataFrame)
    y_train (pd.Series)

    Returns:
    --------
    (tuple) pd.DataFrame with hyperparameter cross validation results, and dict with best hyperparameters
    """
    print("Optimizing hyperparameters for {} model...".format(type(classifier).__name__))
    random_search = RandomizedSearchCV(estimator=classifier, param_distributions=hyperparameter_dist, cv=10, verbose=0,
                                       scoring=scoring, n_iter=n_iter)
    random_search.fit(X_train_preprocessed, y_train.astype('int'))
    print("Best hyperparameters {0}".format(random_search.best_params_))
    print("Best cross validation score {0}\n".format(random_search.best_score_))

    return pd.DataFrame(random_search.cv_results_), random_search.best_estimator_


features, targets, categorical_attributes, numerical_attributes = load_john_doe()

name = 'JosButtler_RightArmSeam_'

X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2)
X_train, X_valid, y_train, y_valid = train_test_split(features.loc[X_train.index], targets[X_train.index],
                                                      test_size=0.2)

training_set = pd.merge(X_train, y_train, left_index=True, right_index=True)
valid_set = pd.merge(X_valid, y_valid, left_index=True, right_index=True)
test_set = pd.merge(X_test, y_test, left_index=True, right_index=True)

X_train_preprocessed, X_valid_preprocessed, preprocessor = full_preprocessing_pipeline(X_train=training_set,
                                                                                       X_test=valid_set,
                                                                                       categorical_features=categorical_attributes,
                                                                                       numeric_features=numerical_attributes,
                                                                                       fit=True)

categorical_features_transformed = preprocessor.transformers[1][1]['one_hot_encoder'].fit(
    X_train[categorical_attributes], y_train).get_feature_names_out(categorical_attributes)

all_features_transformed = list(numerical_attributes) + list(categorical_features_transformed)

classifiers = {
    'xgboost': (XGBClassifier(), [{"max_depth": np.arange(3, 15),
                                   "num_leaves": np.arange(0, 100),
                                   "eta": uniform(0, 0.05)}])
}

# store results of hyperparameter search for each model
cv_results = {
    'xgboost': None,
}

for classifier_name, classifier_obj in classifiers.items():
    results = hyperparameter_opt(classifier=classifier_obj[0],
                                 hyperparameter_dist=classifier_obj[1],
                                 y_train=y_train,
                                 scoring="neg_log_loss",
                                 n_iter=20)
    cv_results[classifier_name] = results

validation_set_predictions = {
    'xgboost': None
}

for classifier_name, classifier_obj in cv_results.items():
    classifier = cv_results[classifier_name][1]
    y_train_pred = classifier.predict_proba(X_train_preprocessed)
    y_valid_pred = classifier.predict_proba(X_valid_preprocessed)

    validation_set_predictions[classifier_name] = y_valid_pred

    print(classifier_name)
    print("training set results: {0}".format(log_loss(y_train.astype('int'), y_train_pred, normalize=True)))
    print("validation set results: {0}".format(log_loss(y_valid.astype('int'), y_valid_pred, normalize=True)))
    print("")

ensemble_xg_model = VotingClassifier(
    estimators=[
        ('xgboost', cv_results["xgboost"][1]),
    ],
    voting='soft')

ensemble_xg_model.fit(X_train_preprocessed, y_train.astype('int'))
y_valid_pred = ensemble_xg_model.predict_proba(X_valid_preprocessed)
print("validation set results: {0}".format(log_loss(y_valid.astype('int'), y_valid_pred, normalize=True)))

X_train_full_preprocessed = np.concatenate((X_train_preprocessed, X_valid_preprocessed), axis=0)
cv_results["xgboost"][1].fit(X_train_full_preprocessed, y_train.append(y_valid).astype('int'))

X_train_full_preprocessed = np.concatenate((X_train_preprocessed, X_valid_preprocessed), axis=0)
y_train_full = y_train.append(y_valid)
X_test_preprocessed, X_test_preprocessed, preprocessor = full_preprocessing_pipeline(X_train=test_set,
                                                                                     X_test=test_set,
                                                                                     categorical_features=categorical_attributes,
                                                                                     numeric_features=numerical_attributes,
                                                                                     fit=True)

test_set_predictions = {
    'xgboost': None,
}

for classifier_name, classifier_obj in cv_results.items():
    classifier = cv_results[classifier_name][1]
    classifier.fit(X_train_full_preprocessed, y_train.append(y_valid).astype('int'))
    y_test_pred_prob = classifier.predict_proba(X_test_preprocessed)
    y_test_pred = classifier.predict(X_test_preprocessed)

    test_set_predictions[classifier_name] = y_test_pred_prob

    print(classifier_name)
    print("test set results: {0}".format(log_loss(y_test.astype('int'), y_test_pred_prob, normalize=True)))
    accuracy = accuracy_score(y_test.astype('int'), y_test_pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    print("")

    classifier.save_model('../models/' + name + "_".join(numerical_attributes) + '.json')
    # xgboost.plot_importance(classifier)
    xgboost.plot_tree(classifier, num_trees=5, rankdir='LR')
    plt.show()

xgb = cv_results["xgboost"][1]
explainer = shap.TreeExplainer(xgb)
shap_values = explainer.shap_values(X_train_preprocessed)
shap.summary_plot(shap_values, X_train_preprocessed, feature_names=all_features_transformed)
plt.show()

test_set_results = test_set.copy()
test_set_results["xgboost xg"] = test_set_predictions["xgboost"][:, 1]

test_set_results.sort_values("xgboost xg", ascending=False)
