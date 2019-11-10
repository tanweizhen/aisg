import yaml
with open(r'mlp/config.yml') as file:
    cfg = yaml.load(file, Loader=yaml.FullLoader)
print("\nLoaded config file.")

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score,train_test_split
if cfg['models']['LinearRegression']:
    from sklearn.linear_model import LinearRegression
if cfg['models']['RandomForestRegressor']:
    from sklearn.ensemble import RandomForestRegressor
if cfg['models']['LGBMRegressor']:
    from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

def rounding_figures(number):
    if cfg['eval_metric']['to_round']:
        return round(number,cfg['eval_metric']['rounding_decimals'])
    else:
        return number

def generate_eval_metrics(estimator,target,predictor):
    model = estimator
    X_test = predictor
    y_test = target
    if cfg['eval_metric']['show_mae']:
        print('Mean absolute error:',rounding_figures(mean_absolute_error(y_test,model.predict(X_test))))
    if cfg['eval_metric']['show_mse']:
        print('Mean squared error:',rounding_figures(mean_squared_error(y_test,model.predict(X_test))))    
    if cfg['eval_metric']['show_mse']:
        print('R2 score:',rounding_figures(r2_score(y_test,model.predict(X_test))))
    if cfg['eval_metric']['do_cv']:
        print(cfg['eval_metric']['cv_scoring'],'CV:',rounding_figures(np.average(cross_val_score(model,X_train,y_train,cv=cfg['eval_metric']['cv_folds']))))

## Loading url
data = pd.read_csv(cfg['url'])
print('Columns:',list(data.columns))

## Preprocessing
data['date_time'] = pd.to_datetime(data['date_time'])
print("Converted date_time to datetime format.")

data.set_index("date_time",inplace=True)
print("Set date_time as index.")

data['day_of_week']=[i.weekday() for i in data.index]
data['day_of_week']= data['day_of_week'].map({0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun',})
data['hour']=[i.hour for i in data.index]
data['month']=[i.month for i in data.index]
print("Engineered time-based features from date_time.")

for col in ['hour','month']:
    data[col] = data[col].astype(object)
print('Setting hour of the day and month to object dtype.')

data['holiday']=data['holiday'].map({'None':0})
data['holiday'].fillna(value=1,inplace=True)
print('Set holiday to one-hot encoding.')

if cfg['to_drop']:
    data.drop(cfg['drop_list'],axis=1,inplace=True)
print("Columns dropped:",cfg['drop_list'])
print('Columns left:',list(data.columns))


for col in list(data.columns):
    if data[col].dtypes=='O':
        data = pd.concat([data,pd.get_dummies(data[col],drop_first=True)],axis=1)
        data.drop(col,axis=1,inplace=True)
print('Dummified object columns.')

data_y = data['traffic_volume']
data_x = data.drop(['traffic_volume'],axis=1)
del data

if type(cfg['train_test_split']['test_size']) != float:
    raise ValueError('Test_size is not a float.')
## Splitting train-test sets
if cfg['train_test_split']['split_by_time']:
    split = int(data_x.shape[0]*(1-cfg['train_test_split']['test_size']))
    X_train = data_x[:split]
    X_test = data_x[split:]
    y_train = data_y[:split]
    y_test = data_y[split:]
    print("\nTrain-test split by time.")
    print("Test size:",cfg['train_test_split']['test_size'])
else:
    X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size=cfg['train_test_split']['test_size'], random_state=cfg['train_test_split']['random_state'])
    print("\nTrain-test split randomly.")
    print("Test size:",cfg['train_test_split']['test_size'])

## Building models and generating eval metrics
if cfg['models']['LinearRegression']:
    print("\n--Linear Regression--")
    model = LinearRegression()
    model.set_params(**cfg['LinearRegression'])
    model.fit(X_train,y_train)
    generate_eval_metrics(model,y_test,X_test)

if cfg['models']['RandomForestRegressor']:
    print("\n--Random Forest--")
    model = RandomForestRegressor()
    model.set_params(**cfg['RandomForest'])
    model.fit(X_train,y_train)
    generate_eval_metrics(model,y_test,X_test)

if cfg['models']['LGBMRegressor']:
    print("\n--LightGBM--")
    model = LGBMRegressor()
    model.set_params(**cfg['LightGBM'])
    model.fit(X_train,y_train)
    generate_eval_metrics(model,y_test,X_test)