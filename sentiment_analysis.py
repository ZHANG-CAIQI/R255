from transformers import BertForSequenceClassification
from transformers import BertTokenizer
import torch
import numpy as np
from tqdm import tqdm
import os

biased_data_sentiment = np.load('/home/telinwu/research/caiqi/CORGI-PM/dataset/biased_corpus_sentiment.npy',allow_pickle=True).item()
for split in ['train', 'valid', 'test']:
    count = 0
    for i in biased_data_sentiment[split]['sentiment']:
        #print(i.tolist()[0][0])
        try:
            if i.tolist()[0][0]> 0.5:
                count += 1
        except:
            print(i)
    print("In biased data, {} set, there are {}% negative sentences. ".format(split, count/len(biased_data_sentiment[split]['sentiment'])))

non_biased_data_sentiment = np.load('/home/telinwu/research/caiqi/CORGI-PM/dataset/non_biased_corpus_sentiment.npy',allow_pickle=True).item()
for split in ['train', 'valid', 'test']:
    count = 0
    for i in non_biased_data_sentiment[split]['sentiment']:
        #print(i.tolist()[0][0])
        try:
            if i.tolist()[0][0]> 0.5:
                count += 1
        except:
            print(i)
    print("In non biased data, {} set, there are {}% negative sentences. ".format(split, count/len(non_biased_data_sentiment[split]['sentiment'])))