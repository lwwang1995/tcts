import numpy as np
import pandas as pd
import lightgbm as lgb
import time
from hyperopt import hp, fmin, tpe, partial

task = 0 # task1
df = pd.read_pickle('./feature.pkl')
df['label'] = pd.read_pickle('./init_label_daily.pkl')
df['label'] = df['label'].groupby(level='instrument').apply(lambda x:x.shift(-task))
label = df['label']
df = df.iloc[:,0:360]

train_start_date='2008-01-01'
train_end_date='2013-12-31'
valid_start_date='2014-01-01'
valid_end_date='2015-12-31'
test_start_date='2016-01-01'
test_end_date='2020-08-31'
train_slc = slice(pd.Timestamp(train_start_date), pd.Timestamp(train_end_date))
valid_slc = slice(pd.Timestamp(valid_start_date), pd.Timestamp(valid_end_date))
test_slc = slice(pd.Timestamp(test_start_date), pd.Timestamp(test_end_date))

x_train = np.array(df.loc[train_slc])
x_valid = np.array(df.loc[valid_slc])
x_test = np.array(df.loc[test_slc])

y_train = np.array(label.loc[train_slc])
y_valid = np.array(label.loc[valid_slc])
y_test = np.array(label.loc[test_slc])

from pathlib import Path
def make_path(path):
    import os
    path = Path(path)
    if not path.parent.exists():
        os.makedirs(path.parent)
    return path

def get_gbdt_param_path():
    return make_path('./baseline_lgb/lgb_param_label%s.npy'%(task))

def get_gbdt_model_path():
    return './baseline_lgb/lgb_model_label%s.bin'%(task)

def scorer(params, return_model=False):
    print(params)
    params.update({'metric': 'mse', 'num_threads': 30, 'verbosity': 0})
    params['num_leaves'] = int(params['num_leaves'])
    hist = dict()
    model = lgb.train(
        params, dtrain, num_boost_round=1000,
        valid_sets=[dtrain, dvalid],
        valid_names=['train','valid'],
        early_stopping_rounds=50,
        verbose_eval=10,
        evals_result=hist
    )
    if return_model: return model
    return model.best_score['valid']['l2']

def search(scorer, max_evals=10):

    space = {
        'learning_rate': hp.uniform("learning_rate", 0.01, 0.05),
        'num_leaves': hp.quniform("num_leaves", 30, 250, 23),
        'subsample': hp.uniform("subsample", 0.6, 1),
        'colsample_bytree': hp.uniform("colsample_bytree", 0.6, 1),
        'lambda_l2': hp.uniform("lambda_l2", 0, 600),
    }
    tic = time.time()
    algo = partial(tpe.suggest, n_startup_jobs=1)
    best = fmin(scorer, space, algo=algo, max_evals=max_evals)
    print('>> time cost: %.3f'%(time.time()-tic))
    return best

dtrain = lgb.Dataset(x_train, y_train)
dvalid = lgb.Dataset(x_valid, y_valid)

params = search(scorer, max_evals=10)
np.save(get_gbdt_param_path(), params)
model = scorer(params, return_model=True)
model.save_model(get_gbdt_model_path())

model = lgb.Booster(model_file=get_gbdt_model_path())
label = label.to_frame()
label['pred'] = model.predict(np.concatenate([x_train, x_valid, x_test], axis = 0))
label.columns = ['label', 'score']
label['label'] = label['label'].groupby(level='datetime').apply(lambda x:(x-x.mean())/x.std())
label['score'] = label['score'].groupby(level='datetime').apply(lambda x:(x-x.mean())/x.std())

train_res = label.loc[train_slc]
valid_res = label.loc[valid_slc]
test_res = label.loc[test_slc]

res = dict()

ic = valid_res.groupby(level='datetime').apply(
    lambda x: x.label.corr(x.score, method='pearson'))
mse = np.nanmean((valid_res['score']-valid_res['label'])**2)
print(('Valid: mse:%.3lf, Rank IC %.3f, Rank ICIR %.3f')%(
    mse, ic.mean(), ic.mean()/ic.std()))
res['valid'] = {}
res['valid']['mse'] = mse
res['valid']['ic'] = ic.mean()
res['valid']['icir'] = ic.mean() / ic.std()

ic = test_res.groupby(level='datetime').apply(
    lambda x: x.label.corr(x.score, method='pearson'))
mse = np.nanmean((test_res['score']-test_res['label'])**2)
print(('Test: mse:%.3lf, Rank IC %.3f, Rank ICIR %.3f')%(
    mse, ic.mean(), ic.mean()/ic.std()))
res['test'] = {}
res['test']['mse'] = mse
res['test']['ic'] = ic.mean()
res['test']['icir'] = ic.mean() / ic.std()
res_df = pd.DataFrame.from_dict(res)
res_df.to_csv('./baseline_lgb/metrics_label%d.csv'%(task))
valid_res.to_pickle('./baseline_lgb/label%d_pred.pkl.valid'%(task))
test_res.to_pickle('./baseline_lgb/label%d_pred.pkl.test'%(task))
print('finished.')