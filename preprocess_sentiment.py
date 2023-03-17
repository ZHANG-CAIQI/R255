from transformers import BertForSequenceClassification
from transformers import BertTokenizer
import torch
import numpy as np
from tqdm import tqdm
import os

tokenizer=BertTokenizer.from_pretrained('IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment')
model=BertForSequenceClassification.from_pretrained('IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment')

biased_data = np.load('/home/telinwu/research/caiqi/CORGI-PM/dataset/CORGI-PC_splitted_biased_corpus_v1.npy',allow_pickle=True).item()
non_biased_data = np.load('/home/telinwu/research/caiqi/CORGI-PM/dataset/CORGI-PC_splitted_non-bias_corpus_v1.npy',allow_pickle=True).item()

# for split in ['train', 'valid', 'test']:
#     sentiment = []
#     for sent in tqdm(biased_data[split]['ori_sentence']):
#         try:
#             output=model(torch.tensor([tokenizer.encode(sent)]))
#             sentiment.append(torch.nn.functional.softmax(output.logits,dim=-1))
#         except:
#             print(sent)
#             sentiment.append(torch.tensor([0.5, 0.5]))
#     biased_data[split]['sentiment'] = sentiment

# np.save("/home/telinwu/research/caiqi/CORGI-PM/dataset/biased_corpus_sentiment.npy", biased_data)

for split in ['train', 'valid', 'test']:
    path = '/home/telinwu/research/caiqi/CORGI-PM/dataset/non_biased_corpus_sentiment.npy'
    if os.path.exists(path):
        non_biased_data = np.load(path,allow_pickle=True).item()
        if 'sentiment' in non_biased_data[split]:
            sentiment = non_biased_data[split]['sentiment']
        else:
            sentiment = []
    else:
        sentiment = []

    for num, sent in tqdm(enumerate(non_biased_data[split]['text'])):
        if num < len(sentiment):
            pass
        else:
            try:
                output=model(torch.tensor([tokenizer.encode(sent)]))
                sentiment.append(torch.nn.functional.softmax(output.logits,dim=-1))
            except KeyboardInterrupt:
                exit()
            except Exception as e:
                print(sent)
                print(e)
                sentiment.append(torch.tensor([0.5, 0.5]))
        if num != 0 and num % 1000 == 0:
            non_biased_data[split]['sentiment'] = sentiment
            np.save(path, non_biased_data)
        if num == len(non_biased_data[split]['text']) - 1:
            non_biased_data[split]['sentiment'] = sentiment
            np.save(path, non_biased_data)

