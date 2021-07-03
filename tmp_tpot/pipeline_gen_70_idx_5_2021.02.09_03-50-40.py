import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import SelectPercentile, f_regression
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures, StandardScaler
from sklearn.svm import LinearSVR
from tpot.builtins import StackingEstimator

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)

# Average CV score on the training set was: -3.647453286147803
exported_pipeline = make_pipeline(
    SelectPercentile(score_func=f_regression, percentile=98),
    StackingEstimator(estimator=GradientBoostingRegressor(alpha=0.8, learning_rate=1.0, loss="quantile", max_depth=1, max_features=0.25, min_samples_leaf=12, min_samples_split=11, n_estimators=100, subsample=0.6500000000000001)),
    StandardScaler(),
    MinMaxScaler(),
    PolynomialFeatures(degree=2, include_bias=False, interaction_only=False),
    StackingEstimator(estimator=SGDRegressor(alpha=0.0, eta0=1.0, fit_intercept=False, l1_ratio=0.0, learning_rate="invscaling", loss="huber", penalty="elasticnet", power_t=1.0)),
    LinearSVR(C=1.0, dual=True, epsilon=1.0, loss="epsilon_insensitive", tol=0.01)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
