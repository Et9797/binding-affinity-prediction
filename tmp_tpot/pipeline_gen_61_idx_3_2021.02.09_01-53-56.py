import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectPercentile, f_regression
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVR
from tpot.builtins import StackingEstimator

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)

# Average CV score on the training set was: -3.6788274480134397
exported_pipeline = make_pipeline(
    SelectPercentile(score_func=f_regression, percentile=96),
    MinMaxScaler(),
    StackingEstimator(estimator=SGDRegressor(alpha=0.01, eta0=1.0, fit_intercept=False, l1_ratio=0.0, learning_rate="constant", loss="epsilon_insensitive", penalty="elasticnet", power_t=10.0)),
    LinearSVR(C=1.0, dual=True, epsilon=1.0, loss="epsilon_insensitive", tol=1e-05)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
