import numpy as np
import pandas as pd
from sklearn.cluster import FeatureAgglomeration
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.feature_selection import SelectPercentile, f_regression
from sklearn.linear_model import RidgeCV, SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVR
from tpot.builtins import StackingEstimator
from xgboost import XGBRegressor

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)

# Average CV score on the training set was: -3.2825658619421776
exported_pipeline = make_pipeline(
    SelectPercentile(score_func=f_regression, percentile=92),
    StackingEstimator(estimator=KNeighborsRegressor(n_neighbors=48, p=1, weights="uniform")),
    StackingEstimator(estimator=XGBRegressor(learning_rate=0.001, max_depth=1, min_child_weight=3, n_estimators=100, n_jobs=1, objective="reg:squarederror", subsample=0.9500000000000001, verbosity=0)),
    MinMaxScaler(),
    StackingEstimator(estimator=SGDRegressor(alpha=0.01, eta0=0.01, fit_intercept=False, l1_ratio=0.0, learning_rate="constant", loss="huber", penalty="elasticnet", power_t=0.0)),
    StackingEstimator(estimator=LinearSVR(C=25.0, dual=True, epsilon=0.1, loss="epsilon_insensitive", tol=0.0001)),
    FeatureAgglomeration(affinity="manhattan", linkage="complete"),
    SelectPercentile(score_func=f_regression, percentile=6),
    StackingEstimator(estimator=ExtraTreesRegressor(bootstrap=False, max_features=0.8, min_samples_leaf=19, min_samples_split=10, n_estimators=400)),
    StackingEstimator(estimator=GradientBoostingRegressor(alpha=0.9, learning_rate=0.01, loss="quantile", max_depth=5, max_features=0.9000000000000001, min_samples_leaf=8, min_samples_split=4, n_estimators=200, subsample=0.05)),
    FeatureAgglomeration(affinity="l1", linkage="complete"),
    RidgeCV()
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
