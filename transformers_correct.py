import torch
import torch.nn as nn
import torch.nn.functional as F 
import warnings
from torch.optim import Adam
import pandas as pd
from datetime import datetime
import numpy as np
import time
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset 
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from torcheval.metrics.functional import r2_score
warnings.simplefilter('ignore')

class PositionalEmbedding(nn.Module):
    def __init__(self, dropout, hyperparameter : int = 2100, embed_model_dim : int= 512, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.embed_model_dim = embed_model_dim
        self.hyperparameter = hyperparameter
        self.params = torch.empty(6, 1)
        self.dropout = dropout
        if self.dropout:
            self.dropout_layer = nn.Dropout(self.dropout)
        torch.nn.init.xavier_normal(self.params)
        self.params = torch.nn.Parameter(self.params)
        self.register_buffer('pe', self.params)

    def forward(self, date_time_str : list, training : bool = True):
        """
            date_time_str : Enter in the form of batch dimension as (batch_dim, 1)
        """
        value = torch.empty(len(date_time_str), self.embed_model_dim)
        self.params.requires_grad = training
        for i in range(len(date_time_str)):
            for j in range(self.embed_model_dim , 2):
                date = datetime.strptime(date_time_str[i][0], r'%Y-%m-%d %H:%M:%S')
                year = date.year
                month = date.month
                day = date.day
                hour = date.hour
                minute = date.minute
                second = date.second
                value[i][j] = (torch.sin(self.params[0][0] * year / (self.hyperparameter ** (2*(j+1) / self.embed_model_dim))) + 
                                torch.sin(self.params[1][0] * month / (self.hyperparameter ** (2*(j+1) / self.embed_model_dim))) + 
                                torch.sin(self.params[3][0] * day  / (self.hyperparameter ** (2*(j+1) / self.embed_model_dim))) +
                                torch.sin(self.params[4][0] * hour / (self.hyperparameter ** (2*(j+1) / self.embed_model_dim))) +
                                torch.sin(self.params[5][0] * minute / (self.hyperparameter ** (2*(j+1) / self.embed_model_dim))) +
                                torch.sin(self.params[6][0] * second / (self.hyperparameter ** (2*(j+1) / self.embed_model_dim)))) / 6
                
                value[i][j + 1] = (torch.cos(self.params[0][0] * year / (self.hyperparameter ** (2*(j+1) / self.embed_model_dim))) + 
                                torch.cos(self.params[1][0] * month / (self.hyperparameter ** (2*(j+1) / self.embed_model_dim))) + 
                                torch.cos(self.params[3][0] * day  / (self.hyperparameter ** (2*(j+1) / self.embed_model_dim))) +
                                torch.cos(self.params[4][0] * hour / (self.hyperparameter ** (2*(j+1) / self.embed_model_dim))) +
                                torch.cos(self.params[5][0] * minute / (self.hyperparameter ** (2*(j+1) / self.embed_model_dim))) +
                                torch.cos(self.params[6][0] * second / (self.hyperparameter ** (2*(j+1) / self.embed_model_dim)))) / 6
            
        return self.dropout_layer(value)
        # return value

class MultiHeadAttention(nn.Module):
    def __init__(self, in_features = 64, embed_model_dim = 64, n_heads = 8,*args, dropout,**kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dropout = dropout
        self.embed_model_dim = embed_model_dim
        self.n_heads = n_heads
        assert self.embed_model_dim % self.n_heads == 0
        self.single_head_dim = self.embed_model_dim // self.n_heads
        
        self.dropout_layer = nn.Dropout(self.dropout)
        self.query_matrix = nn.Linear(in_features = in_features, out_features = self.embed_model_dim, bias=False)
        self.key_matrix = nn.Linear(in_features = in_features, out_features = self.embed_model_dim, bias=False)
        self.value_matrix = nn.Linear(in_features = in_features, out_features = self.embed_model_dim, bias=False)
        self.output_matrix = nn.Linear(in_features=self.embed_model_dim, out_features=self.embed_model_dim, bias = True)

    @staticmethod
    def attention(key : torch.Tensor, query : torch.Tensor, value : torch.Tensor, mask : bool = False):
        """
            query, key, value -> (batch, n_heads, seq_len, 64)
        """

        attention = torch.matmul(query, key.transpose(-1, -2)) / torch.sqrt(torch.tensor(query.shape[-1])) ##YAHA PE 512 LIKHA H
        
        if mask:
            attention.masked_fill_(mask == 0, -1e9)  # mask hum bhejenge
        attention = torch.softmax(attention, axis = -1)
        attention = torch.matmul(attention, value).transpose(1, 2)
        attention = attention.reshape(attention.shape[0], attention.shape[1], -1)
        return attention


    def forward(self, q, k, v, mask = False):
        # q, k, v -> (batch, 60, 512)
        query = self.query_matrix(q).view(q.shape[0], q.shape[1], self.n_heads, -1)
        key = self.key_matrix(k).view(k.shape[0], k.shape[1], self.n_heads, -1)
        value = self.value_matrix(v).view(v.shape[0], v.shape[1], self.n_heads, -1)  # batch, 60, 8, 64

        attention = MultiHeadAttention.attention(key = key.transpose(1, 2), query = query.transpose(1, 2) ,value = value.transpose(1, 2), mask = mask)
        
        attention = self.dropout_layer(attention)
        
        attention = self.output_matrix(attention)
        
        return attention

class LayerNormalization(nn.Module):

    def __init__(self, eps : float = 1e-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim = True)
        std = x.std(dim = -1, keepdim = True)

        return self.alpha * (x - mean) / (std + self.eps) + self.beta
    

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff,dropout,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.linear1 = nn.Linear(in_features=d_model, out_features=d_ff)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(in_features=d_ff, out_features=d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        (Batch, seq_len, d_model) -> (Batch, seq_len, d_ff) -> (Batch, seq_len, d_model)
        """
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(self.dropout(x))

        return self.dropout(x)

class ResidualConnections(nn.Module):

    def __init__(self, dropout : float, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    
class EncoderBlock(nn.Module):
    def __init__(self, multihead_attention : MultiHeadAttention, feed_forward_block : FeedForward, dropout : float,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.attention_block = multihead_attention
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnections(dropout) for _ in range(2)])

    def forward(self, x, source_mask):
        x = self.residual_connections[0](x, lambda x: self.attention_block(x, x, x, source_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x
    

class Encoder(nn.Module):

    def __init__(self, layers : nn.ModuleList ,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    

class DecoderBlock(nn.Module):

    def __init__(self, self_attention_block : MultiHeadAttention, cross_attention_block: MultiHeadAttention, feed_forward_block : FeedForward, d_model : int, dropout : float,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.dropout = nn.Dropout(dropout)
        self.residual_block = nn.ModuleList([ResidualConnections(dropout) for _ in range(3)])
        self.linear1 = nn.Linear(in_features=d_model, out_features = d_model)
        self.linear2 = nn.Linear(in_features=d_model, out_features = d_model)
        self.linear3 = nn.Linear(in_features=d_model, out_features = d_model)
        self.linear4 = nn.Linear(in_features=d_model, out_features = d_model)
        self.linear5 = nn.Linear(in_features=d_model, out_features = d_model)


    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_block[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_block[1](x, lambda x: self.cross_attention_block(F.tanh(self.linear3(x)), self.linear1(encoder_output) + F.relu(self.linear4(x)), self.linear2(encoder_output) + F.relu(self.linear5(x)), src_mask))
        x = self.residual_block[2](x, self.feed_forward_block)

        return x
    

class Decoder(nn.Module):

    def __init__(self, layers : nn.ModuleList, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)

        return self.norm(x)
    

class ProjectionLayer(nn.Module):

    def __init__(self, d_model :int, d_output : int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.linear1 = nn.Linear(in_features=1280, out_features=1024, device='cuda')
        self.linear2 = nn.Linear(in_features = 1024, out_features = d_output, device='cuda')
    def forward(self, x):
        # batch_len, seq_len, d_model -> batch_len, seq_len, d_output
        x = x.view(x.shape[0], -1)
        print
        x = self.linear1(x)
        x = self.linear2(x)

        return x
    
class InputEmbeddings(nn.Module):

    def __init__(self, d_input : int, d_model : int = 512, dropout:int = 0.2, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.linear1 = nn.Linear(in_features = d_input, out_features = 32).to('cuda')
        self.linear4 = nn.Linear(in_features=32, out_features=d_model).to('cuda')
        self.dropout = nn.Dropout(dropout).to('cuda')

    def forward(self, x):
        # batch_len, seq_len, d_input -> batch_len, seq_len, d_model
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.linear4(x)
        return self.dropout(x)
    
class FinbertNew(nn.Module):

    def __init__(self, finbert,d_model,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
        self.finbert_bert = finbert.bert.to('cuda')

    def forward(self, x):
        x = self.tokenizer(x, return_tensors="pt", padding=True).to('cuda')
        return self.finbert_bert(**x)['pooler_output']

class Finbert(nn.Module):
    def __init__(self, d_model:int):
      super().__init__()
      self.finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone',num_labels=3)
      self.finbert_model = FinbertNew(self.finbert, d_model)
      self.finbert_dropout = self.finbert.dropout.to('cuda')
      self.finbert_classifier = self.finbert.classifier.to('cuda')
      self.linear = nn.Linear(in_features = 768, out_features = d_model, device='cuda')

    def forward(self, x):
      x = self.finbert_model(x)
      x = self.finbert_dropout(x)
      embeddings = self.linear(x)
      sentiments = self.finbert_classifier(x)
      return F.softmax(sentiments, dim = -1), embeddings
# DO REMEMBER TO SEND THE DATA IN THIS FORM ONLYYYYYY!!!!!!
class L2_Regularized_Loss(nn.Module):
    def __init__(self, lamda, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.lamda = lamda

    def forward(self, y_pred : torch.tensor, y_true : torch.tensor, model, mode = 'train'):
        """
        Computes L2 loss with L2 Regularization

        Args:
            y_pred (torch.tensor) : Model predictions (Unnormalized Logits)
            y_true (torch.tensor) : True labels
            model (nn.Module) : The Neural Network Model
            mode (string) : if train, L2 regularization will work else Not

        Return:
            torch.Tensor : L2 Loss with L2 Regularization
        """

        mse_loss = nn.MSELoss()(y_pred, y_true)
        reg_loss = 0.0
        if mode == 'train':
            for param in  model.parameters():
                reg_loss += torch.norm(param, p = 2) ** 2
                # print('hi')
                # print('Reg Loss: ', reg_loss)

        total_loss = mse_loss + self.lamda * reg_loss

        return total_loss

class Transformer(nn.Module):
    #  encoder : Encoder, decoder : Decoder, projection_layer : ProjectionLayer, finbert : Finbert,
    def __init__(self, positional_embeddings:PositionalEmbedding, input_embeddings : InputEmbeddings,encoder : Encoder, finbert : Finbert,decoder : Decoder,projection_layer : ProjectionLayer, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.encoder =  encoder.to('cuda')
        self.decoder = decoder.to('cuda')
        self.input_embeddings = input_embeddings
        self.projection_layer = projection_layer.to('cuda')
        self.finbert = finbert
        self.positional_embeddings = positional_embeddings
        self.loss = nn.MSELoss()
        #self.optim = Adam(self.parameters(), lr = 0.000001)
        self.optim=torch.optim.Adam(self.parameters(), lr=0.0001)
        # self.scheduler = CosineAnnealingLR(self.optim, T_max=10, eta_min=0.00001)

    def forward(self, ohlcv_batch : torch.tensor, news_batch : list, date_batch : list):
        """
        ohlcv_batch is tensor of batch_len, seq_len, feature_length
        news_batch is a list of batch_length, seq_lenor window_size, 
        date_batch is a numpy array of seq_len, 1 which is a string
        """
        
        ls_date = []
        ls_sentiments = []
        ls_embeddings = []
        for news, date in zip(news_batch, date_batch):
            for j in range(len(news)):
                if len(news[j]) > 512:
                    news[j] = news[j][:512]
            
            finbert_output = self.finbert(news)
                
            ls_sentiments.append(finbert_output[0])
            ls_embeddings.append(finbert_output[1])
            positional_encoding = self.positional_embeddings(date)
            ls_date.append(positional_encoding)
            

        embeddings = torch.stack(ls_embeddings).to('cuda')
        sentiments = torch.stack(ls_sentiments).to('cuda')
        
        ohlcv_batch = torch.cat([ohlcv_batch, sentiments], dim=-1)
        positional_encoding = torch.stack(ls_date).to('cuda')
        input_embedding = self.input_embeddings(ohlcv_batch)
        input = input_embedding + positional_encoding
        encoder_output = self.encoder(input, mask=False)
        decoder_output = self.decoder(embeddings,encoder_output, False, False)
        return self.projection_layer(decoder_output)
    
    def calculate_loss(self, preds : torch.tensor, true : torch.tensor):
        return self.loss(preds, true)

    # def get_directional_accuracy(self, current:torch.tensor, previous:torch.tensor):

    def fit(self, ohlcv_train_dl : DataLoader, news_train_dl : DataLoader, date_train_dl : DataLoader, ohlcv_val_dl : DataLoader, news_val_dl : DataLoader, date_val_dl : DataLoader,epochs : int = 10):
        history = pd.DataFrame(columns = ['Epoch', 'Train Accuracy', 'Val Accuracy'])
        for epoch in tqdm(range(epochs)):
            final_training_loss = 0
            no_of_batches = len(ohlcv_train_dl)
            i=0
            for (ohlcv, news, date) in tqdm(zip(ohlcv_train_dl, news_train_dl, date_train_dl)):
                self.train()
                ohlcv_x, ohlcv_y = ohlcv
                preds = self(ohlcv_x, news, date)
                batch_loss = L2_Regularized_Loss(0.01)(preds, ohlcv_y, self, 'train')
                
                if not pd.isna(batch_loss.detach().cpu().numpy()):
                    self.optim.zero_grad()
                    batch_loss.backward()
                    self.optim.step()
                    # self.scheduler.step()
                    i += 1
                    final_training_loss += batch_loss.detach().cpu().item()
                    print('Epoch: '+ f'{epoch+1}/{epochs}','     Training Loss: ', '{:15.8f}'.format(batch_loss.detach().cpu().item()), '     Batch Number', f'{i+1}/{no_of_batches}', end = '\r')
            
            final_training_loss /= i
            
            final_val_loss = 0 
            
            no_of_batches=len(ohlcv_val_dl)
            i = 0
            for (ohlcv, news, date) in tqdm(zip(ohlcv_val_dl, news_val_dl, date_val_dl)):
                self.eval()
                ohlcv_x, ohlcv_y = ohlcv
                preds = self(ohlcv_x, news, date)
                batch_loss = L2_Regularized_Loss(0.01)(preds, ohlcv_y, self, 'val')
                if not pd.isna(batch_loss.detach().cpu().numpy()):
                    final_val_loss += batch_loss.detach().cpu().item()
                    i+=1
            		
                    print('Epoch: '+ f'{epoch+1}/{epochs}','     Training Loss: ', '{:15.8f}'.format(final_training_loss),'     Validation Loss: ', '{:15.8f}'.format(batch_loss.detach().cpu().item()), '     Batch number: ', f'{i+1}/{no_of_batches}', end = '\r')
                # else :
                #     print(ohlcv)
                #     print(date)
                #     for sentence in news:
                #         for wrd in sentence:
                #             print(wrd)

            
            final_val_loss /= i
            print('Epoch: '+ f'{epoch+1}/{epochs}', '     Training Loss: ', '{:15.8f}'.format(final_training_loss), '     Validation_Loss', '{:15.8f}'.format(final_val_loss), "                                                                          ")
            history.loc[epoch] = [epoch+1, final_training_loss, final_val_loss]
            history.to_csv(f'Models/{epoch+1}.csv', index=False)
            torch.save(self, f'Models/{epoch+1}.pt')
            print('Completed')
            exit(0)
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

def main():
    import pandas as pd
    d_model = 64
    encoder_list = nn.ModuleList()
    for i in range(6):
        encoder_list.append(EncoderBlock(MultiHeadAttention(d_model, d_model, n_heads=8, dropout=0.5), FeedForward(d_model, 3092, 0.5), dropout=0.5))
    encoder = Encoder(encoder_list)
    decoder_list = nn.ModuleList()
    for i in range(6):
        decoder_list.append(DecoderBlock(self_attention_block = MultiHeadAttention(d_model, d_model, n_heads = 8, dropout = 0.5), cross_attention_block = MultiHeadAttention(d_model, d_model, n_heads = 8, dropout = 0.5),feed_forward_block = FeedForward(d_model, 3092, 0.5),d_model = d_model,dropout = 0.5))
    decoder = Decoder(decoder_list)
    # for ohlcv, news, date in zip(ohlcv_dl, news_dl, date_dl):
    #model = Transformer(PositionalEmbedding(dropout=0.1, embed_model_dim=d_model), InputEmbeddings(8, d_model), encoder, Finbert(d_model), decoder, ProjectionLayer(d_model, d_output=2))
    model = torch.load('Model.pt')
    ohlcv_df = pd.read_csv('FINAL/meta_ohlcv.csv')
    news_df = pd.read_csv('FINAL/meta_news.csv')
    
    
    
    ohlcv_train_df, ohlcv_val_df = train_test_split(ohlcv_df, test_size=0.05, shuffle = False)
    news_train_df, news_val_df= train_test_split(news_df, test_size=0.05, shuffle = False)
    date_train_df, date_val_df = train_test_split(ohlcv_df[['Date']], test_size=0.05, shuffle = False)
    ohlcv_train_dl = create_dataloader_ohlcv(ohlcv_train_df, 20, batch_size=8)
    ohlcv_val_dl = create_dataloader_ohlcv(ohlcv_val_df, 20, batch_size=8, is_val=True)
    ohlcv_train_dl = DeviceDataLoader(ohlcv_train_dl, 'cuda')
    ohlcv_val_dl = DeviceDataLoader(ohlcv_val_dl, 'cuda')
    date_train_dl = create_dataloader_date(date_train_df, 20, 8)
    date_val_dl = create_dataloader_date(date_val_df, 20, 8)
    news_train_dl = create_dataloader_news(news_train_df, 20, 8)
    news_val_dl = create_dataloader_news(news_val_df, 20, 8)
    
    	
    model.fit(ohlcv_train_dl, news_train_dl, date_train_dl, ohlcv_val_dl, news_val_dl, date_val_dl, epochs = 100)
                
   





if __name__ == '__main__':
    main()
    




