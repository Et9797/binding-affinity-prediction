import numpy as np
import pandas as pd
from sklearn.cluster import FeatureAgglomeration
from sklearn.feature_selection import SelectPercentile, f_regression
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LassoLarsCV
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor
from tpot.builtins import StackingEstimator

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)

# Average CV score on the training set was: -3.6926304620666826
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=KNeighborsRegressor(n_neighbors=48, p=1, weights="uniform")),
    MinMaxScaler(),
    StackingEstimator(estimator=LinearSVR(C=25.0, dual=True, epsilon=0.01, loss="epsilon_insensitive", tol=0.0001)),
    StackingEstimator(estimator=DecisionTreeRegressor(max_depth=8, min_samples_leaf=17, min_samples_split=9)),
    SelectPercentile(score_func=f_regression, percentile=46),
    FeatureAgglomeration(affinity="l2", linkage="average"),
    RBFSampler(gamma=0.75),
    StackingEstimator(estimator=LassoLarsCV(normalize=True)),
    LinearSVR(C=1.0, dual=True, epsilon=1.0, loss="epsilon_insensitive", tol=1e-05)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
