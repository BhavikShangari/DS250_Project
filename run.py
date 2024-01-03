import torch
import pandas as pd
from modules import *
from utils import *

d_model = 64

# for ohlcv, news, date in zip(ohlcv_dl, news_dl, date_dl):
try:
    model = torch.load('Model.pt')
except:
    encoder_list = nn.ModuleList()
    for i in range(6):
        encoder_list.append(EncoderBlock(MultiHeadAttention(d_model, d_model, n_heads=8, dropout=0.5), FeedForward(d_model, 3092, 0.5), dropout=0.5))
    encoder = Encoder(encoder_list)
    decoder_list = nn.ModuleList()
    for i in range(6):
        decoder_list.append(DecoderBlock(self_attention_block = MultiHeadAttention(d_model, d_model, n_heads = 8, dropout = 0.5), cross_attention_block = MultiHeadAttention(d_model, d_model, n_heads = 8, dropout = 0.5),feed_forward_block = FeedForward(d_model, 3092, 0.5),d_model = d_model,dropout = 0.5))
    decoder = Decoder(decoder_list)
    
    model = Transformer(PositionalEmbedding(dropout=0.1, embed_model_dim=d_model), InputEmbeddings(8, d_model), encoder, Finbert(d_model), decoder, ProjectionLayer(d_model, d_output=2))
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
