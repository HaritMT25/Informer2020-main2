Skip to main content
latestinformer.ipynb
latestinformer.ipynb_
Files
..
Drop files to upload them to session storage.
Disk
81.51 GB available

[76]
0s
import os
import shutil

# Delete all .pyc files
for root, dirs, files in os.walk("."):
    for file in files:
        if file.endswith(".pyc"):
            os.remove(os.path.join(root, file))

# Delete all __pycache__ directories
for root, dirs, files in os.walk("."):
    for dir in dirs:
        if dir == "__pycache__":
            shutil.rmtree(os.path.join(root, dir))

print("All .pyc files and __pycache__ directories have been removed.")

All .pyc files and __pycache__ directories have been removed.
Informer Demo
Download code and dataset

[77]
1s
%reset -f
import os
os.chdir("/content")
!pwd
!rm -r ./Informer2020-main2
!rm -r ./ETDataset
!rm -r ./results
!rm -r ./Informer2020
!rm -r .
!ls
/content
rm: cannot remove './results': No such file or directory
rm: cannot remove './Informer2020': No such file or directory
rm: refusing to remove '.' or '..' directory: skipping '.'
informer_checkpoints  sample_data

[78]
2s
!git clone -b HaritMT25-patch-1 https://github.com/HaritMT25/Informer2020-main2.git
!git clone https://github.com/zhouhaoyi/ETDataset.git
!ls
Cloning into 'Informer2020-main2'...
remote: Enumerating objects: 260, done.
remote: Counting objects: 100% (52/52), done.
remote: Compressing objects: 100% (49/49), done.
remote: Total 260 (delta 36), reused 3 (delta 3), pack-reused 208 (from 2)
Receiving objects: 100% (260/260), 1.14 MiB | 2.97 MiB/s, done.
Resolving deltas: 100% (148/148), done.
Cloning into 'ETDataset'...
remote: Enumerating objects: 187, done.
remote: Counting objects: 100% (28/28), done.
remote: Compressing objects: 100% (8/8), done.
remote: Total 187 (delta 25), reused 20 (delta 20), pack-reused 159 (from 1)
Receiving objects: 100% (187/187), 3.86 MiB | 6.78 MiB/s, done.
Resolving deltas: 100% (62/62), done.
ETDataset  Informer2020-main2  informer_checkpoints  sample_data

[79]
0s
!pwd
!ls
import os
os.chdir('Informer2020-main2')
!ls
/content
ETDataset  Informer2020-main2  informer_checkpoints  sample_data
data	    environment.yml  img      main_informer.py	models	   requirements.txt  utils
Dockerfile  exp		     LICENSE  Makefile		README.md  scripts

[80]
import sys
import matplotlib.pyplot as plt
if not 'Informer2020-main2' in sys.path:
    sys.path += ['Informer2020-main2']

[81]
# !pip install -r ./Informer2020-main2/requirements.txt
Experiments: Train and Test

[82]
0s
from utils.tools import dotdict
from exp.exp_informer import Exp_Informer
import torch
import pandas as pd
import numpy as np
!ls
data	    environment.yml  img      main_informer.py	models	   requirements.txt  utils
Dockerfile  exp		     LICENSE  Makefile		README.md  scripts

[83]
args = dotdict()

args.model = 'informer' # model of experiment, options: [informer, informerstack, informerlight(TBD)]

args.data = 'ETTh1' # data
args.root_path = '../ETDataset/ETT-small/' # root path of data file
args.data_path = 'ETTh1.csv' # data file
args.features = 'M' # forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate
args.target = 'OT' # target feature in S or MS task
args.freq = 'h' # freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h
args.checkpoints = '../informer_checkpoints' # location of model checkpoints

args.seq_len = 96 # input sequence length of Informer encoder
args.label_len = 48 # start token length of Informer decoder
args.pred_len = 24 # prediction sequence length
# Informer decoder input: concat[start token series(label_len), zero padding series(pred_len)]

args.enc_in = 7 # encoder input size
args.dec_in = 7 # decoder input size
args.c_out = 7 # output size
args.factor = 5 # probsparse attn factor
args.d_model = 512 # dimension of model
args.n_heads = 8 # num of heads
args.e_layers = 2 # num of encoder layers
args.d_layers = 1 # num of decoder layers
args.d_ff = 2048 # dimension of fcn in model
args.dropout = 0.05 # dropout
args.attn = 'prob' # attention used in encoder, options:[prob, full]
args.embed = 'timeF' # time features encoding, options:[timeF, fixed, learned]
args.activation = 'gelu' # activation
args.distil = True # whether to use distilling in encoder
args.output_attention = False # whether to output attention in ecoder
args.mix = True
args.padding = 0
args.freq = 'h'

args.batch_size = 128
args.learning_rate = 0.0001
args.loss = 'mse'
args.lradj = 'type1'
args.use_amp = False # whether to use automatic mixed precision training

args.num_workers = 0
args.itr = 1
args.train_epochs = 5
args.patience = 3
args.des = 'exp'

args.use_gpu = True if torch.cuda.is_available() else False
args.gpu = 0

args.use_multi_gpu = False
args.devices = '0,1,2,3'


[84]
args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ','')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

[85]
# Set augments by using data name
data_parser = {
    'ETTh1':{'data':'ETTh1.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
    'ETTh2':{'data':'ETTh2.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
    'ETTm1':{'data':'ETTm1.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
    'ETTm2':{'data':'ETTm2.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
}
if args.data in data_parser.keys():
    data_info = data_parser[args.data]
    args.data_path = data_info['data']
    args.target = data_info['T']
    args.enc_in, args.dec_in, args.c_out = data_info[args.features]

[86]
args.detail_freq = args.freq
args.freq = args.freq[-1:]

[87]
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred
from exp.exp_basic import Exp_Basic
from models.model import Informer, InformerStack
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric
import warnings

warnings.filterwarnings('ignore')

# Placeholder for missing function
def apply_fft_in_pieces(data, seq_len, enc_in):
    # Assuming this function applies FFT preprocessing on the data
    # Here, we return the data as-is. Replace with your FFT logic.
      """
    Apply FFT on the input data in chunks and concatenate the results.

    Args:
        data (numpy.ndarray): The input data array of shape (num_samples, seq_len, enc_in).
        seq_len (int): Sequence length for FFT.
        enc_in (int): Number of input features.

    Returns:
        numpy.ndarray: The FFT-transformed and concatenated data.
    """
      num_samples = data.shape[0]
      fft_transformed = []

      for i in range(num_samples):
          sample = data[i]
          # Apply FFT along the sequence length axis
          fft_result = np.fft.fft(sample, axis=0)
          fft_magnitude = np.abs(fft_result)
          fft_transformed.append(fft_magnitude)

      # Convert the list back to a numpy array
      fft_transformed = np.stack(fft_transformed, axis=0)
      print(f"FFT-Transformed Data Shape: {fft_transformed.shape}")
      return fft_transformed



class Exp_Informer(Exp_Basic):
    def __init__(self, args):
        super().__init__(args)

    def _build_model(self):
        model_dict = {
            'informer': Informer,
            'informerstack': InformerStack,
        }

        if self.args.model in model_dict:
            e_layers = self.args.e_layers if self.args.model == 'informer' else self.args.s_layers
            model = model_dict[self.args.model](
                self.args.enc_in,
                self.args.dec_in,
                self.args.c_out,
                self.args.seq_len,
                self.args.label_len,
                self.args.pred_len,
                self.args.factor,
                self.args.d_model,
                self.args.n_heads,
                e_layers,
                self.args.d_layers,
                self.args.d_ff,
                self.args.dropout,
                self.args.attn,
                self.args.embed,
                self.args.freq,
                self.args.activation,
                self.args.output_attention,
                self.args.distil,
                self.args.mix,
                self.device
            ).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)

        return model

    def _get_data(self, flag):
          args = self.args
          data_dict = {
              'ETTh1': Dataset_ETT_hour,
              'ETTh2': Dataset_ETT_hour,
              'ETTm1': Dataset_ETT_minute,
              'ETTm2': Dataset_ETT_minute,
              'WTH': Dataset_Custom,
              'ECL': Dataset_Custom,
              'Solar': Dataset_Custom,
              'custom': Dataset_Custom,
          }

          Data = data_dict[self.args.data]
          timeenc = 0 if args.embed != 'timeF' else 1

          shuffle_flag = flag != 'test' and flag != 'pred'
          drop_last = flag == 'test'
          batch_size = 1 if flag == 'pred' else args.batch_size
          freq = args.detail_freq if flag == 'pred' else args.freq

          # Adjusted dataset initialization
          data_set = Data(
              root_path=args.root_path,
              data_path=args.data_path,
              flag=flag,
              size=[args.seq_len, args.label_len, args.pred_len],  # Combine sizes into `size`
              features=args.features,
              target=args.target,
              inverse=args.inverse,
              timeenc=timeenc,
              freq=freq,
              cols=args.cols,
              #args=self.args
          )

          data_loader = DataLoader(
              data_set,
              batch_size=batch_size,
              shuffle=shuffle_flag,
              num_workers=args.num_workers,
              drop_last=drop_last,
              pin_memory=True  # Add pin_memory for faster data transfer to GPU
          )

          # Iterate through the data loader and move data to the target device

          # ... (rest of the existing code) ...
          return data_set, data_loader


    def _select_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    def _select_criterion(self):
        return nn.MSELoss()

    def vali(self, vali_data, vali_loader, criterion):
          """
          Validation function.
          """
          self.model.eval()
          total_loss = []

          # Use the provided vali_loader for iteration, not self._get_data()
          with torch.no_grad():
              for batch_x, batch_y, batch_x_mark, batch_y_mark in vali_loader:
                  pred, true = self._process_one_batch(vali_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
                  loss = criterion(pred.detach().cpu(), true.detach().cpu())
                  total_loss.append(loss.item())

          avg_vali_loss = np.mean(total_loss)
          return avg_vali_loss


    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        train_losses = []
        vali_losses = []

        for epoch in range(self.args.train_epochs):
            train_loss = []
            self.model.train()

            for batch_x, batch_y, batch_x_mark, batch_y_mark in train_loader:
                model_optim.zero_grad()

                with torch.cuda.amp.autocast():  # Enable mixed precision if needed
                    pred, true = self._process_one_batch(train_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
                    loss = criterion(pred, true)

                train_loss.append(loss.item())

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            train_loss = np.mean(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)

            train_losses.append(train_loss)
            vali_losses.append(vali_loss)

            print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Validation Loss: {vali_loss:.4f}")

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping triggered.")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        return train_losses, vali_losses







    def test(self, setting):
        _, test_loader = self._get_data(flag='test')

        self.model.eval()
        preds, trues = [], []

        for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
            pred, true = self._process_one_batch(_, batch_x, batch_y, batch_x_mark, batch_y_mark)
            preds.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())

        preds, trues = np.array(preds), np.array(trues)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        mae, mse, _, _, _ = metric(preds, trues)
        print(f'MSE: {mse}, MAE: {mae}')

    def _process_one_batch(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark):
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)
            batch_x_mark = batch_x_mark.float().to(self.device)
            batch_y_mark = batch_y_mark.float().to(self.device)

            # Decoder input
            dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]], device=self.device)
            dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1)

            # Forward pass
            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            # Adjust the target variable to match the prediction length
            true = batch_y[:, -self.args.pred_len:, :].to(self.device)
            return outputs, true


[88]
Exp = Exp_Informer

[89]
class FourierTransform:
    def __call__(self, sample):
        data = sample['data']
        data_fft = apply_fft_in_pieces(data,96,7)  # Apply Fourier Transform
        sample['data'] = torch.tensor(data_fft, dtype=torch.float32)
        return sample



[ ]

Start coding or generate with AI.

[90]
import os
import numpy as np
import pandas as pd

import numpy.fft as fft

import torch
from torch.utils.data import Dataset, DataLoader
# from sklearn.preprocessing import StandardScaler

from utils.tools import StandardScaler
from utils.timefeatures import time_features

import warnings
warnings.filterwarnings('ignore')

class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='h', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path

        self.__read_data__()



    def __read_data__(self):


        self.scaler = StandardScaler()


        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12*30*24 - self.seq_len, 12*30*24+4*30*24 - self.seq_len]
        border2s = [12*30*24, 12*30*24+4*30*24, 12*30*24+8*30*24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)

        self.data_x = data[border1:border2]

        #self.data_x = apply_fft_in_pieces(self.data_x, self.seq_len, 7)
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = np.concatenate([self.data_x[r_begin:r_begin+self.label_len], self.data_y[r_begin+self.label_len:r_end]], 0)
        else:
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len- self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='t', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path

        self.__read_data__()



    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12*30*24*4 - self.seq_len, 12*30*24*4+4*30*24*4 - self.seq_len]
        border2s = [12*30*24*4, 12*30*24*4+4*30*24*4, 12*30*24*4+8*30*24*4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)

        self.data_x = data[border1:border2]
        #self.data_x = apply_fft_in_pieces(self.data_x, self.seq_len,7)
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:borenc_inder2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = np.concatenate([self.data_x[r_begin:r_begin+self.label_len], self.data_y[r_begin+self.label_len:r_end]], 0)
        else:
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='h', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols=cols
        self.root_path = root_path
        self.data_path = data_path

        self.__read_data__()


    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        # cols = list(df_raw.columns);
        if self.cols:
            cols=self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns); cols.remove(self.target); cols.remove('date')
        df_raw = df_raw[['date']+cols+[self.target]]

        num_train = int(len(df_raw)*0.7)
        num_test = int(len(df_raw)*0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train-self.seq_len, len(df_raw)-num_test-self.seq_len]
        border2s = [num_train, num_train+num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)

        self.data_x = data[border1:border2]
        #self.data_x = apply_fft_in_pieces(self.data_x, self.seq_len, 7)
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = np.concatenate([self.data_x[r_begin:r_begin+self.label_len], self.data_y[r_begin+self.label_len:r_end]], 0)
        else:
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len- self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols=cols
        self.root_path = root_path
        self.data_path = data_path

        self.__read_data__()


    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.cols:
            cols=self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns); cols.remove(self.target); cols.remove('date')
        df_raw = df_raw[['date']+cols+[self.target]]

        border1 = len(df_raw)-self.seq_len
        border2 = len(df_raw)

        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len+1, freq=self.freq)

        df_stamp = pd.DataFrame(columns = ['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq[-1:])

        self.data_x = data[border1:border2]
        #self.data_x = apply_fft_in_pieces(self.data_x, self.seq_len,7)
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin+self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin+self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

[95]
0s
for ii in range(args.itr):
    # setting record of experiments
    setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}_mx{}_{}_{}'.format(args.model, args.data, args.features,
                args.seq_len, args.label_len, args.pred_len,
                args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff, args.attn, args.factor, args.embed, args.distil, args.mix, args.des, ii)

    # set experiments
    exp = Exp(args)

    # train


Next steps:

[ ]
import matplotlib.pyplot as plt

for i in range(args.itr):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(all_train_losses[i]) + 1), all_train_losses[i], label=f'Train Loss (Iter {i+1})')
    plt.plot(range(1, len(all_vali_losses[i]) + 1), all_vali_losses[i], label=f'Validation Loss (Iter {i+1})')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Training Curve for Iteration {i+1}')
    plt.legend()


[ ]
all_train_losses = []
all_vali_losses = []

for ii in range(args.itr):
    setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}_mx{}_{}_{}'.format(args.model, args.data, args.features,
                args.seq_len, args.label_len, args.pred_len,
                args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff, args.attn, args.factor, args.embed, args.distil, args.mix, args.des, ii)

    exp = Exp(args)

    print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
    train_losses, vali_losses = exp.train(setting)

    all_train_losses.append(train_losses)
    all_vali_losses.append(vali_losses)

    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting)

    torch.cuda.empty_cache()

Prediction

[ ]
import os

# set saved model path
setting = 'informer_ETTh1_ftM_sl96_ll48_pl24_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_exp_0'
# path = os.path.join(args.checkpoints,setting,'checkpoint.pth')

[ ]
# If you already have a trained model, you can set the arguments and model path, then initialize a Experiment and use it to predict
# Prediction is a sequence which is adjacent to the last date of the data, and does not exist in the data
# If you want to get more information about prediction, you can refer to code `exp/exp_informer.py function predict()` and `data/data_loader.py class Dataset_Pred`

exp = Exp(args)

exp.predict(setting, True)

[ ]
# the prediction will be saved in ./results/{setting}/real_prediction.npy
import numpy as np

prediction = np.load('./results/'+setting+'/real_prediction.npy')

prediction.shape
More details about Prediction - prediction function

[ ]
# here is the detailed code of function predict

def predict(exp, setting, load=False):
    pred_data, pred_loader = exp._get_data(flag='pred')

    if load:
        path = os.path.join(exp.args.checkpoints, setting)
        best_model_path = path+'/'+'checkpoint.pth'
        exp.model.load_state_dict(torch.load(best_model_path))

    exp.model.eval()

    preds = []

    for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(pred_loader):
        batch_x = batch_x.float().to(exp.device)
        batch_y = batch_y.float()
        batch_x_mark = batch_x_mark.float().to(exp.device)
        batch_y_mark = batch_y_mark.float().to(exp.device)

        # decoder input
        if exp.args.padding==0:
            dec_inp = torch.zeros([batch_y.shape[0], exp.args.pred_len, batch_y.shape[-1]]).float()
        elif exp.args.padding==1:
            dec_inp = torch.ones([batch_y.shape[0], exp.args.pred_len, batch_y.shape[-1]]).float()
        else:
            dec_inp = torch.zeros([batch_y.shape[0], exp.args.pred_len, batch_y.shape[-1]]).float()
        dec_inp = torch.cat([batch_y[:,:exp.args.label_len,:], dec_inp], dim=1).float().to(exp.device)
        # encoder - decoder
        if exp.args.use_amp:
            with torch.cuda.amp.autocast():
                if exp.args.output_attention:
                    outputs = exp.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = exp.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        else:
            if exp.args.output_attention:
                outputs = exp.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                outputs = exp.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        f_dim = -1 if exp.args.features=='MS' else 0
        batch_y = batch_y[:,-exp.args.pred_len:,f_dim:].to(exp.device)

        pred = outputs.detach().cpu().numpy()#.squeeze()

        preds.append(pred)

    preds = np.array(preds)
    preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

    # result save
    folder_path = './results/' + setting +'/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    np.save(folder_path+'real_prediction.npy', preds)

    return preds


[ ]
# you can also use this prediction function to get result
prediction = predict(exp, setting, True)

[ ]
import matplotlib.pyplot as plt

plt.figure()
plt.plot(prediction[0,:,-1])
plt.show()
More details about Prediction - prediction dataset
You can give a root_path and data_path of the data you want to forecast, and set seq_len, label_len, pred_len and other arguments as other Dataset. The difference is that you can set a more detailed freq such as 15min or 3h to generate the timestamp of prediction series.

Dataset_Pred only has one sample (including encoder_input: [1, seq_len, dim], decoder_token: [1, label_len, dim], encoder_input_timestamp: [1, seq_len, date_dim], decoder_input_timstamp: [1, label_len+pred_len, date_dim]). It will intercept the last sequence of the given data (seq_len data) to forecast the unseen future sequence (pred_len data).


[ ]
from data.data_loader import Dataset_Pred
from torch.utils.data import DataLoader

[ ]
Data = Dataset_Pred
timeenc = 0 if args.embed!='timeF' else 1
flag = 'pred'; shuffle_flag = False; drop_last = False; batch_size = 1

freq = args.detail_freq

data_set = Data(
    root_path=args.root_path,
    data_path=args.data_path,
    flag=flag,
    size=[args.seq_len, args.label_len, args.pred_len],
    features=args.features,
    target=args.target,
    timeenc=timeenc,
    freq=freq
)
data_loader = DataLoader(
    data_set,
    batch_size=batch_size,
    shuffle=shuffle_flag,
    num_workers=args.num_workers,
    drop_last=drop_last)

[ ]
import torch
import torch.nn as nn

self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                    kernel_size=3, padding=padding, padding_mode='circular')

[ ]
len(data_set), len(data_loader)
Visualization

[ ]
# When we finished exp.train(setting) and exp.test(setting), we will get a trained model and the results of test experiment
# The results of test experiment will be saved in ./results/{setting}/pred.npy (prediction of test dataset) and ./results/{setting}/true.npy (groundtruth of test dataset)

preds = np.load('./results/'+setting+'/pred.npy')
trues = np.load('./results/'+setting+'/true.npy')

# [samples, pred_len, dimensions]
preds.shape, trues.shape

[ ]
import matplotlib.pyplot as plt
import seaborn as sns

[ ]
# draw OT prediction
plt.figure()
plt.plot(trues[0,:,-1], label='GroundTruth')
plt.plot(preds[0,:,-1], label='Prediction')
plt.legend()
plt.show()

[ ]
# draw HUFL prediction
plt.figure()
plt.plot(trues[0,:,0], label='GroundTruth')
plt.plot(preds[0,:,0], label='Prediction')
plt.legend()
plt.show()

[ ]
from data.data_loader import Dataset_ETT_hour
from torch.utils.data import DataLoader

Data = Dataset_ETT_hour
timeenc = 0 if args.embed!='timeF' else 1
flag = 'test'; shuffle_flag = False; drop_last = True; batch_size = 1

data_set = Data(
    root_path=args.root_path,
    data_path=args.data_path,
    flag=flag,
    size=[args.seq_len, args.label_len, args.pred_len],
    features=args.features,
    timeenc=timeenc,
    freq=args.freq
)
data_loader = DataLoader(
    data_set,
    batch_size=batch_size,
    shuffle=shuffle_flag,
    num_workers=args.num_workers,
    drop_last=drop_last)

[ ]
import os

args.output_attention = True

exp = Exp(args)

model = exp.model

setting = 'informer_ETTh1_ftM_sl96_ll48_pl24_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_exp_0'
path = os.path.join(args.checkpoints,setting,'checkpoint.pth')
model.load_state_dict(torch.load(path))

[ ]
# attention visualization
idx = 0
for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(data_loader):
    if i!=idx:
        continue
    batch_x = batch_x.float().to(exp.device)
    batch_y = batch_y.float()

    batch_x_mark = batch_x_mark.float().to(exp.device)
    batch_y_mark = batch_y_mark.float().to(exp.device)

    dec_inp = torch.zeros_like(batch_y[:,-args.pred_len:,:]).float()
    dec_inp = torch.cat([batch_y[:,:args.label_len,:], dec_inp], dim=1).float().to(exp.device)

    outputs,attn = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

[ ]
attn[0].shape, attn[1].shape #, attn[2].shape

[ ]
layer = 0
distil = 'Distil' if args.distil else 'NoDistil'
for h in range(0,8):
    plt.figure(figsize=[10,8])
    plt.title('Informer, {}, attn:{} layer:{} head:{}'.format(distil, args.attn, layer, h))
    A = attn[layer][0,h].detach().cpu().numpy()
    ax = sns.heatmap(A, vmin=0, vmax=A.max()+0.01)
    plt.show()

[ ]
layer = 1
distil = 'Distil' if args.distil else 'NoDistil'
for h in range(0,8):
    plt.figure(figsize=[10,8])
    plt.title('Informer, {}, attn:{} layer:{} head:{}'.format(distil, args.attn, layer, h))
    A = attn[layer][0,h].detach().cpu().numpy()
    ax = sns.heatmap(A, vmin=0, vmax=A.max()+0.01)
    plt.show()
Custom Data
Custom data (xxx.csv) has to include at least 2 features: date(format: YYYY-MM-DD hh:mm:ss) and target feature.


[ ]
from data.data_loader import Dataset_Custom
from torch.utils.data import DataLoader
import pandas as pd
import os

[ ]
# custom data: xxx.csv
# data features: ['date', ...(other features), target feature]

# we take ETTh2 as an example
args.root_path = './ETDataset/ETT-small/'
args.data_path = 'ETTh2.csv'

df = pd.read_csv(os.path.join(args.root_path, args.data_path))

[ ]
df.head()

[ ]
'''
We set 'HULL' as target instead of 'OT'

The following frequencies are supported:
        Y   - yearly
            alias: A
        M   - monthly
        W   - weekly
        D   - daily
        B   - business days
        H   - hourly
        T   - minutely
            alias: min
        S   - secondly
'''

args.target = 'HULL'
args.freq = 'h'

Data = Dataset_Custom
timeenc = 0 if args.embed!='timeF' else 1
flag = 'test'; shuffle_flag = False; drop_last = True; batch_size = 1

data_set = Data(
    root_path=args.root_path,
    data_path=args.data_path,
    flag=flag,
    size=[args.seq_len, args.label_len, args.pred_len],
    features=args.features,
    timeenc=timeenc,
    target=args.target, # HULL here
    freq=args.freq # 'h': hourly, 't':minutely
)
data_loader = DataLoader(
    data_set,
    batch_size=batch_size,
    shuffle=shuffle_flag,
    num_workers=args.num_workers,
    drop_last=drop_last)

[ ]
batch_x,batch_y,batch_x_mark,batch_y_mark = data_set[0]

[ ]

Start coding or generate with AI.
Colab paid products - Cancel contracts here
383940414235363733343243444546474849505152535455565758596061
        padding = m  # Ensures output length matches input length
        self.tokenConv = nn.Conv1d(
            in_channels=c_in, 
            out_channels=d_model, 
            kernel_size=kernel_size, 
            padding=padding, 
            padding_mode='circular'  # Circular padding for sequence continuity
        )
        
        # Initialize weights


  0s
completed at 2:55 PM
