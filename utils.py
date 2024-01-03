import pandas as pd
import numpy as np
import torch
import time


class DateDataset(torch.utils.data.Dataset):

    def __init__(self,date : pd.DataFrame, window_size : int = 60, batch_size : int = 64):
        date = date[['Date']].to_numpy()[:-1]
        ls = []
        for i in range(window_size - 1, len(date)):
            data_window = date[i-window_size+1:i+1]
            ls.append(data_window)
        self.correct_form = np.stack(ls).tolist()

    def __len__(self):
        return len(self.correct_form)
    
    def __getitem__(self, idx):
        date = self.correct_form[idx]
        return {'date' : date}


def create_dataloader_date(date : pd.DataFrame, window_size : int = 60, batch_size : int = 64):
    
    date_dataset = DateDataset(date, window_size, batch_size)
    return DataLoader(date_dataset, batch_size, num_workers=4, pin_memory= True, collate_fn=date_collate_fn)

class StringDataset(torch.utils.data.Dataset):

    def __init__(self, news:pd.DataFrame, window_size : int = 60, batch_size : int = 64):
        news_numpy = news[['News']].to_numpy().flatten()[:-1]
        ls = []
        for i in range(window_size - 1, len(news_numpy)):
            data_window = news_numpy[i-window_size+1:i+1]
            ls.append(data_window)
        self.correct_form = np.stack(ls)

    def __len__(self):
        return len(self.correct_form)

    def __getitem__(self, idx):
        text = self.correct_form[idx].tolist()
        return {'text': text}

def create_dataloader_news(news:pd.DataFrame, window_size : int = 60, batch_size : int = 64):
    """
    return numpy array of it
    """
    
    str_dataset = StringDataset(news, window_size, batch_size)
    return DataLoader(str_dataset, batch_size, num_workers =4,pin_memory=True, collate_fn=news_collate_fn)
    

def news_collate_fn(sample):
    ls = []
    for samp in sample:
        ls.append(samp['text'])
    return np.stack(ls).tolist()

def date_collate_fn(sample):
    ls = []
    for samp in sample:
        ls.append(samp['date'])
    return np.stack(ls).tolist()

def create_dataloader_ohlcv(df : pd.DataFrame, window_size : int = 60, batch_size = 64, is_val:bool = False):
    labels = torch.tensor(df[['High', 'Low']].to_numpy(), dtype=torch.float32)[window_size:]
    data = df[['Open', 'High', 'Low', 'Close', 'Volume']].iloc[:-1,:]
    minmax= np.array([[3.6579000e+02, 3.6601300e+02, 3.6527300e+02, 3.6578300e+02, 2.7176684e+07],
                        [11.128, 11.161, 11.092, 11.131,  1.   ],
                        [3.5466200e+02, 3.5485200e+02, 3.5418100e+02, 3.5465200e+02, 2.7176683e+07,]])
    if not is_val:
        scaler = MinMaxScaler()
        data = pd.DataFrame(scaler.fit_transform(data))
    else:
        data = pd.DataFrame((np.array(data) - minmax[1]) / minmax[2])
    # labels = torch.tensor(pd.DataFrame(scaler.fit_transform(labels)).to_numpy(), dtype=torch.float32)
    ls = []
    for i in range(window_size-1, len(data)):
        data_window = data.loc[i - window_size+1: i]    
        ls.append(data_window.to_numpy())
    correct_form = torch.tensor(ls, dtype=torch.float32)
    # print(correct_form.shape)
    dataset = TensorDataset(correct_form, labels)
    dataloader = DataLoader(dataset, batch_size, num_workers=4, pin_memory=True)
    return dataloader
def to_device(data, device):
  if isinstance(data, (list, tuple)):
    return [to_device(x, device) for x in data]
  return data.to(device, non_blocking = True)

class DeviceDataLoader():
  def __init__(self, data, device):
    self.data = data
    self.device = device

  def __iter__(self):
    for b in self.data:
      yield to_device(b, self.device)

  def __len__(self):
    return len(self.data)

