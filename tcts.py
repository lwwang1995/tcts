import torch
import torch.nn as nn
import torch.optim as optim

import os
import copy
import argparse
import datetime
import numpy as np
import pandas as pd
from copy import deepcopy

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

EPS = 1e-12
e = 2.718281828459


class DataLoader:

    def __init__(self, df_feature, df_label, device, batch_size=800, pin_memory=False):

        assert len(df_feature) == len(df_label)

        self.df_feature = df_feature.values
        self.df_label = df_label.values
        self.device =  device

        if pin_memory:
            self.df_feature = torch.tensor(self.df_feature, dtype=torch.float, device=self.device)
            self.df_label = torch.tensor(self.df_label, dtype=torch.float, device=self.device)

        self.index = df_label.index

        self.batch_size = batch_size
        self.pin_memory = pin_memory

        self.daily_count = df_label.groupby(level='datetime').size().values
        self.daily_index = np.roll(np.cumsum(self.daily_count), 1)
        self.daily_index[0] = 0

    @property
    def batch_length(self):

        if self.batch_size <= 0:
            return self.daily_length

        return len(self.df_label) // self.batch_size

    @property
    def daily_length(self):

        return len(self.daily_count)

    def iter_batch(self):

        if self.batch_size <= 0:
            yield from self.iter_daily()
            return

        indices = np.arange(len(self.df_label))
        np.random.shuffle(indices)

        for i in range(len(indices))[::self.batch_size]:
            if len(indices) - i < self.batch_size:
                break
            yield indices[i:i+self.batch_size] # NOTE: advanced indexing will cause copy

    def iter_daily(self):

        for idx, count in zip(self.daily_index, self.daily_count):
            yield slice(idx, idx + count) # NOTE: slice index will not cause copy

    def get(self, slc):

        outs = self.df_feature[slc], self.df_label[slc]

        if not self.pin_memory:
            outs = tuple(torch.tensor(x, dtype=torch.float, device=self.device) for x in outs)

        return outs + (self.index[slc],)


class GRU(nn.Module):

    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0, output_dim=1):
        super().__init__()

        self.rnn = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc_out = nn.Linear(hidden_size, output_dim)
        self.softmax = nn.Softmax(dim=1)

        self.d_feat = d_feat
        self.output_dim = output_dim

    def forward(self, x):
        # x: [N, F*T]
        x = x.reshape(len(x), self.d_feat, -1) # [N, F, T]
        x = x.permute(0, 2, 1) # [N, T, F]
        out, _ = self.rnn(x)
        out = self.fc_out(out[:, -1, :]).squeeze()
        self.embedding = out
        if self.output_dim > 1:
            out = self.softmax(out)
        return out


class MLP(nn.Module):

    def __init__(self, d_feat, hidden_size=256, num_layers=3, dropout=0.0, output_dim=1):
        super().__init__()

        self.mlp = nn.Sequential()
        self.softmax = nn.Softmax(dim=1)

        for i in range(num_layers):
            if i > 0:
                self.mlp.add_module('drop_%d'%i, nn.Dropout(dropout))
            self.mlp.add_module('fc_%d'%i, nn.Linear(
                d_feat if i == 0 else hidden_size, hidden_size))
            self.mlp.add_module('relu_%d'%i, nn.ReLU())

        self.mlp.add_module('fc_out', nn.Linear(hidden_size, output_dim))

    def forward(self, x):
        # feature
        # [N, F]
        out = self.mlp(x).squeeze()
        out = self.softmax(out)
        return out

class LSTM(nn.Module):

    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0):
        super().__init__()

        self.rnn = nn.LSTM(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc_out = nn.Linear(hidden_size, 1)

        self.d_feat = d_feat

    def forward(self, x):
        # x: [N, F*T]
        x = x.reshape(len(x), self.d_feat, -1) # [N, F, T]
        x = x.permute(0, 2, 1) # [N, T, F]
        out, _ = self.rnn(x)
        return self.fc_out(out[:, -1, :]).squeeze()

def get_model(model_name):

    if model_name.upper() == 'LSTM':
        return LSTM

    if model_name.upper() == 'GRU':
        return GRU

    if model_name.upper() == 'MLP':
        return MLP

    raise ValueError('unknown model name `%s`'%model_name)


def mse(pred, label, weight):
    loc = torch.argmax(weight, 1)
    loss = (pred - label[np.arange(weight.shape[0]),loc])**2
    return torch.mean(loss)

def loss_fn(pred, label, weight, args):

    if args.loss == 'mse':
        return mse(pred, label, weight)

    raise ValueError('unknown loss `%s`'%args.loss)

def calculate_valid_mse(pred, label, weight, args):
    loc = torch.argmax(weight, 1)
    valid_score = -torch.mean((pred - label[:,args.label])**2)
    loss = -valid_score*torch.log(weight[np.arange(weight.shape[0]), loc])
    return torch.mean(loss)

global_log_file = None
def pprint(*args):
    # print with UTC+8 time
    time = '['+str(datetime.datetime.utcnow()+
                   datetime.timedelta(hours=8))[:19]+'] -'
    print(time, *args, flush=True)

    if global_log_file is None:
        return
    with open(global_log_file, 'a') as f:
        print(time, *args, flush=True, file=f)

global_step = -1
def train_epoch(epoch, fore_model, weight_model, fore_optimizer, weight_optimizer, \
                train_loader, valid_loader, writer, args):

    global global_step

    init_fore_model = deepcopy(fore_model)
    for p in init_fore_model.parameters():
        p.init_fore_model = False

    fore_model.train()
    weight_model.train()

    for p in weight_model.parameters():
        p.requires_grad = False
    for p in fore_model.parameters():
        p.requires_grad = True

    # fix weight model and train forecasting model
    for _ in range(args.steps):
        for slc in tqdm(train_loader.iter_batch(), total=train_loader.batch_length):

            global_step += 1
            feature, label, _ = train_loader.get(slc)

            init_pred = init_fore_model(feature)
            pred = fore_model(feature)

            dis = init_pred - label.transpose(0,1)
            weight_feature = torch.cat((feature,dis.transpose(0,1),label,init_pred.view(-1,1)), 1)
            weight = weight_model(weight_feature)

            loss = loss_fn(pred, label, weight, args) # hard
            fore_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(fore_model.parameters(), 3.)
            fore_optimizer.step()

            writer.add_scalar('Train/RunningLoss', loss.item(), global_step)

    for p in weight_model.parameters():
        p.requires_grad = True
    for p in fore_model.parameters():
        p.requires_grad = False

    # fix forecasting model and train weight model
    for slc in tqdm(valid_loader.iter_batch(), total=valid_loader.batch_length):

        global_step += 1

        feature, label, _ = valid_loader.get(slc)

        pred = fore_model(feature)
        dis = pred - label.transpose(0,1)
        weight_feature = torch.cat((feature,dis.transpose(0,1),label,pred.view(-1,1)), 1)
        weight = weight_model(weight_feature)

        loss = calculate_valid_mse(pred, label, weight, args)
        weight_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(weight_model.parameters(), 3.)
        weight_optimizer.step()

        writer.add_scalar('Train/RunningLoss', loss.item(), global_step)


def test_epoch(epoch, model, test_loader, writer, args, prefix='Test'):

    model.eval()

    losses = []

    for slc in tqdm(test_loader.iter_daily(), desc=prefix, total=test_loader.daily_length):

        feature, label, _ = test_loader.get(slc)
        with torch.no_grad():
            pred = model(feature)

        loss = torch.mean((pred - label[:,abs(args.label)])**2)
        losses.append(loss.item())

    writer.add_scalar(prefix+'/Loss', np.mean(losses), epoch)
    writer.add_scalar(prefix+'/std(Loss)', np.std(losses), epoch)

    return np.mean(losses)


def inference(model, data_loader, args):

    model.eval()

    preds = []
    for slc in tqdm(data_loader.iter_daily(), total=data_loader.daily_length):

        feature, label, index = data_loader.get(slc)
        with torch.no_grad():
            pred = model(feature)

        preds.append(pd.DataFrame({
            'score': pred.cpu().numpy(),
            'label': label.cpu().numpy()[:,abs(args.label)], # first columns is the label
            }, index=index))

    preds = pd.concat(preds, axis=0)

    return preds


def create_loaders(args, device):

    df = pd.read_pickle('./feature.pkl')
    df = df.iloc[:, 0:360]
    df['label0'] = pd.read_pickle('./%s.pkl'%(args.label_name))

    for i in range(1,args.output_dim):
        df['label%d'%(-i)] = df['label0'].groupby(level='instrument').apply(lambda x:x.shift(-i))

    df.dropna(subset=['label%d'%(-i) for i in range(args.output_dim)], inplace=True)

    # NOTE: we always assume the last column is label
    df_feature = df.iloc[:, 0:360]
    df_label = df.iloc[:, 360:]

    slc = slice(pd.Timestamp(args.train_start_date), pd.Timestamp(args.train_end_date))
    train_loader = DataLoader(df_feature.loc[slc], df_label.loc[slc], device=device,
                              batch_size=args.batch_size, pin_memory=args.pin_memory)

    slc = slice(pd.Timestamp(args.valid_start_date), pd.Timestamp(args.valid_end_date))
    valid_loader = DataLoader(df_feature.loc[slc], df_label.loc[slc], device=device, pin_memory=False)

    slc = slice(pd.Timestamp(args.test_start_date), pd.Timestamp(args.test_end_date))
    test_loader = DataLoader(df_feature.loc[slc], df_label.loc[slc], device=device, pin_memory=False)

    return train_loader, valid_loader, test_loader


def main(args):

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    suffix = "%s_dh%s_dn%s_drop%s_lr%s%s_bs%s_seed%s%s_%s_step%s"%(
        args.model_name, args.hidden_size, args.num_layers, args.dropout,
        args.fore_lr, args.weight_lr, args.batch_size, args.seed, args.annot, args.label_name, args.steps
    )

    output_path = args.outdir
    if not output_path:
        output_path = './rl_enhance_label%d_%d/'%(args.label, args.output_dim) + suffix
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if not args.overwrite and os.path.exists(output_path+'/'+'pred.pkl.test'):
        print('already runned, exit.')
        return

    writer = SummaryWriter(log_dir=output_path)

    global global_log_file
    global_log_file = output_path + '/' + 'run.log'

    pprint('create model...')
    global device
    device = 'cuda:%d'%(args.cuda) if torch.cuda.is_available() else 'cpu'
    # initialize models
    fore_model = get_model(args.model_name)(d_feat=6, hidden_size=64, num_layers=2, dropout=0)
    if args.pretrain:
        pretrain_param = torch.load('%s/model.bin'%(args.pretrain_path), map_location=device)
        fore_model.load_state_dict(pretrain_param)
    weight_model = MLP(args.d_feat+2*args.output_dim+1, args.hidden_size, args.num_layers, args.dropout, args.output_dim)

    fore_model.to(device)
    weight_model.to(device)

    fore_optimizer = optim.Adam(fore_model.parameters(), lr=args.fore_lr)
    weight_optimizer = optim.Adam(weight_model.parameters(), lr=args.weight_lr)

    pprint('create loaders...')
    train_loader, valid_loader, test_loader = create_loaders(args=args, device=device)

    best_loss = np.inf
    best_epoch = 0
    stop_round = 0
    fore_best_param = copy.deepcopy(fore_optimizer.state_dict())
    weight_best_param = copy.deepcopy(weight_optimizer.state_dict())

    for epoch in range(args.n_epochs):
        pprint('Epoch:', epoch)

        pprint('training...')
        train_epoch(epoch, fore_model, weight_model, fore_optimizer, weight_optimizer, train_loader, valid_loader, writer, args)

        pprint('evaluating...')
        val_loss  = test_epoch(epoch, fore_model, valid_loader, writer, args, prefix='Valid')
        test_loss = test_epoch(epoch, fore_model, test_loader, writer, args, prefix='Test')

        pprint('valid %.6f, test %.6f'%(val_loss, test_loss))

        if val_loss < best_loss:
            best_loss = val_loss
            stop_round = 0
            best_epoch = epoch
            torch.save(copy.deepcopy(fore_model.state_dict()), output_path+'/fore_model.bin')
            torch.save(copy.deepcopy(weight_model.state_dict()), output_path+'/weight_model.bin')

        else:
            stop_round += 1
            if stop_round >= args.early_stop:
                pprint('early stop')
                break

    pprint('best loss:', best_loss, '@', best_epoch)
    best_param = torch.load(output_path + '/fore_model.bin')
    fore_model.load_state_dict(best_param)
    best_param = torch.load(output_path + '/weight_model.bin')
    weight_model.load_state_dict(best_param)

    pprint('inference...')
    res = dict()
    for name in ['train', 'valid', 'test']:
        res[name] = {}

        pred = inference(fore_model, eval(name+'_loader'), args)
        pred.to_pickle(output_path+'/pred.pkl.'+name)

        pred['label'] = pred['label'].groupby(level='datetime').apply(lambda x:(x-x.mean())/x.std())
        pred['score'] = pred['score'].groupby(level='datetime').apply(lambda x:(x-x.mean())/x.std())

        ic = pred.groupby(level='datetime').apply(
            lambda x: x.label.corr(x.score, method='pearson'))
        mse = np.nanmean((pred['score']-pred['label'])**2)

        pprint(('%s: mse:%.3lf, Rank IC %.3f, Rank ICIR %.3f')%(
            name, mse, ic.mean(), ic.mean()/ic.std()))

        res[name]['mse'] = mse
        res[name]['ic'] = ic.mean()
        res[name]['icir'] = ic.mean() / ic.std()

    res_df = pd.DataFrame.from_dict(res)
    res_df.to_csv(output_path+'/metrics.csv')
    pprint('finished.')


def parse_args():

    parser = argparse.ArgumentParser()

    # model
    
    parser.add_argument('--model_name', default='GRU')
    parser.add_argument('--d_feat', type=int, default=360)
    parser.add_argument('--hidden_size', type=int, default=32)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--output_dim', type=int, default=5)

    # training
    parser.add_argument('--pretrain', type=bool, default=True)
    parser.add_argument('--pretrain_path', default='')
    parser.add_argument('--n_epochs', type=int, default=500)
    parser.add_argument('--fore_lr', type=float, default=5e-7)
    parser.add_argument('--weight_lr', type=float, default=5e-7)
    parser.add_argument('--steps', type=int, default=3)
    parser.add_argument('--early_stop', type=int, default=20)
    parser.add_argument('--metric', default='') # '' refers to loss
    parser.add_argument('--loss', default='mse')

    # data
    parser.add_argument('--pin_memory', action='store_false', default=True)
    parser.add_argument('--batch_size', type=int, default=800) # -1 indicate daily batch
    parser.add_argument('--label_name', default='init_label_daily') # specify other labels
    parser.add_argument('--label', type=int, default=0)
    parser.add_argument('--train_start_date', default='2008-01-01')
    parser.add_argument('--train_end_date', default='2013-12-31')
    parser.add_argument('--valid_start_date', default='2014-01-01')
    parser.add_argument('--valid_end_date', default='2015-12-31')
    parser.add_argument('--test_start_date', default='2016-01-01')
    parser.add_argument('--test_end_date', default='2020-08-01')

    # other
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--annot', default='')
    parser.add_argument('--outdir', default='')
    parser.add_argument('--overwrite', action='store_true', default=False)
    parser.add_argument('--cuda', type=int, default=0)

    args = parser.parse_args()

    return args

if __name__ == '__main__':

    args = parse_args()
    main(args)
