import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.svm import LinearSVR
from tpot.builtins import StackingEstimator
from sklearn.preprocessing import FunctionTransformer
from copy import copy

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)

# Average CV score on the training set was: -4.60699470868297
exported_pipeline = make_pipeline(
    make_union(
        make_union(
            make_union(
                make_union(
                    make_union(
                        FunctionTransformer(copy),
                        make_union(
                            FunctionTransformer(copy),
                            FunctionTransformer(copy)
                        )
                    ),
                    FunctionTransformer(copy)
                ),
                make_union(
                    make_union(
                        make_union(
                            FunctionTransformer(copy),
                            FunctionTransformer(copy)
                        ),
                        FunctionTransformer(copy)
                    ),
                    FunctionTransformer(copy)
                )
            ),
            FunctionTransformer(copy)
        ),
        FunctionTransformer(copy)
    ),
    LinearSVR(C=0.0001, dual=False, epsilon=0.001, loss="squared_epsilon_insensitive", tol=1e-05)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
