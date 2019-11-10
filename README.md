# Summary

This pipeline is designed to be as simple as possible while still fully configurable.

To keep this short, the pipeline works like this:
1. run.sh activates the python script
2. python script loads parameters from yaml config file
3. loads data from specified url from config into a pandas dataframe
4. performs preprocessing (only columns to drop can be configured)
5. splits dataframe into train-test sets (randomly or by time, test_size can be adjusted)
6. trains the various models (only linear regression, random forest and lightgbm supported) and generates/prints relevant evaluation metrics (can be toggled, but only mae, r2, mse supported), and also cross validation scoring and folds.

# Configuration

The config file basically controls every aspect of the python script. I've opted to use yaml as it's much easier to read compared to json.

###### url: the link we download and load the data from.
i.e. https://aisgaiap.blob.core.windows.net/aiap5-assessment-data/traffic_data.csv

###### to_drop: Boolean that determines whether we drop any columns.

###### drop_list: A python list that determines columns we drop, if to_drop is set to True. 
i.e. ['snow_1h','rain_1h','clouds_all','weather_description']

###### train_test_split:

- split_by_time: Boolean - if True, we split the train-test based on time. If False, we split randomly.
  
- test_size: A float between 0 and 1. Determines the size of the test set (0.2 would mean 20% of the dataset is used as the test set)
  
- random_state: Random seed if train-test are split randomly.

###### eval_metric:
  to_round: Boolean, if we want to round our evaluation metrics to a certain number of decimal points.
  rounding_decimals: Number of decimal points to be rounded to.
  show_mae: Boolean to determine if mean absolute error is to be printed.
  show_mse: Boolean to determine if mean squared error is to be printed.
  show_r2: Boolean to determine if r2 score is to be printed.
  do_cv: Boolean to determine if cross validation score is to be printed.
  cv_folds: Cross validation folds for cross_val_score.
  cv_scoring: Scoring metric for cross_val_score. Can be 'r2_score', 'mean_absolute_error', 'mean_squared_error'

###### models:
  LinearRegression: Boolean to determine whether the script builds a linear regression model.
  RandomForestRegressor: Boolean to determine whether the script builds a random forest regressor model.
  LGBMRegressor: Boolean to determine whether the script builds a LGBM regression model.

###### LinearRegression: Basically the parameters for sklearn's OLS linear model.
Refer to https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression

###### RandomForest: Basically the parameters for sklearn's OLS linear model.
Refer to https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html#sklearn.ensemble.RandomForestRegressor

###### LightGBM: Basically the parameters for LightGBM's regression model
Refer to https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html#lightgbm.LGBMRegressor
