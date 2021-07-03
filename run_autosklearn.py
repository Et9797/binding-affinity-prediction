if __name__ == "__main__":
    import numpy as np
    from autosklearn.regression import AutoSklearnRegressor
    from autosklearn.metrics import mean_squared_error
    import pickle

    #load X, y
    _file = open('data_BA.pkl', 'rb')
    X, y = pickle.load(_file)
    _file.close()

    #autosklearn
    regr = AutoSklearnRegressor(time_left_for_this_task=172800,
        per_run_time_limit = 600,
        resampling_strategy='cv',
        resampling_strategy_arguments={'folds': 4},
        metric=mean_squared_error,
        n_jobs=2,
    )
    regr.fit(X, y)

    #pickle best regressor
    _file = open('Autoskl_bestmodel.pkl', "wb")
    pickle.dump(regr, _file)
    _file.close()
