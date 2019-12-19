# %%
import sys

import keras
import now
from kashgari.callbacks import EvalCallBack

# from CustomModel import DoubleBLSTMModel
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from pytorch_pretrained_bert import BertTokenizer, BertConfig, CONFIG_NAME, WEIGHTS_NAME
from pytorch_pretrained_bert import BertAdam, BertForSequenceClassification
import pandas as pd
import io
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import torch

from Word2Vec_LSTM import createWord2VecModel
from custom_model import DoubleBLSTMModel

from preprocessing import doPreprocessing
import pandas as pd
from sklearn.model_selection import train_test_split

from keras_preprocessing.text import Tokenizer, text_to_word_sequence

import kashgari
from kashgari.tasks.classification import BiLSTM_Model
from kashgari.embeddings import BERTEmbedding
import os
import logging
import argparse
from datetime import datetime

# Use GPU Flag
from pytorchBERT import train_Pytorch_BERT

#%%

kashgari.config.use_cudnn_cell = True
# use GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


parser = argparse.ArgumentParser(description='Model Parametrization')
parser.add_argument('--embedding', default='pytorchbert', help='embedding to use. Either BERT, pytorchBERT or W2V')
parser.add_argument('--model', default='custom', help='For BERT either BiLSTM or CUSTOM. Not relevant for W2V')
parser.add_argument('--action', default='new',
                    help='If "new" create new model or "load" for continuing to train old model(Only for BERT)')
parser.add_argument('--bertpath', default='models/uncased_L-12_H-768_A-12',
                    help='Path to the pretrained BERT Model')
parser.add_argument('--existingModelPath', default='data/modelName', help='INPUT FILE PATH')
parser.add_argument('--epochs', default=2, help='INPUT FILE PATH')
parser.add_argument('--batchsize', default=32, help='INPUT FILE PATH')
parser.add_argument('--doPreProcessing', default=True, help='INPUT FILE PATH')
args = parser.parse_args()


MODELFILENAME = 'DEFAULT'

if bool(args.doPreProcessing) == True:
    if str(args.embedding).lower() == 'bert':
        doPreprocessing(forWordToVec=False)
    else:
        doPreprocessing(forWordToVec=True)

if (str(args.embedding).lower() == 'bert' or str(args.embedding).lower() == 'pytorchbert')  and os.path.exists('data/preprocessedTweetsBERT.csv'):
    input_data = pd.read_csv('data/preprocessedTweetsBERT.csv', sep=',', header=0, encoding='latin-1')
elif args.embedding.lower() == 'w2v' and os.path.exists('data/preprocessedTweetsW2Vec.csv'):
    input_data = pd.read_csv('data/preprocessedTweetsW2Vec.csv', sep=',', header=0, encoding='latin-1')
else:
    raise ValueError("Inputfile is missing! Invalid action or not existing \"existingModelPath\" need!")




print(input_data['target'].value_counts())

#Tokenize Tweets
tweets = input_data['text'].tolist()


train_sequences = []
for tweet in tweets:
    train_sequences.append(text_to_word_sequence(str(tweet)))




if str(args.embedding).lower() == 'bert':


    print('Using BERT embedding')
    if args.action == 'new':
        print('Training new Model')

        train_x, test_x, train_y, test_y = train_test_split(train_sequences, input_data['target'], test_size=0.1,
                                                            random_state=1234)
        train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.1, random_state=1234)

        # Kashgari seems to like lists and no Series/Panda types
        train_y = train_y.tolist()
        test_y = test_y.tolist()
        valid_y = valid_y.tolist()

        ##Create String labels.. kashgari cannot handle binary numeric targets after reloading the model..
        # for i in range(0, len(train_y)):
        #    train_y[i] = str(train_y[i])
        # for i in range(0, len(valid_y)):
        #    valid_y[i] = str(valid_y[i])
        # for i in range(0, len(test_y)):
        #    test_y[i] = str(test_y[i])
        if os.path.exists(args.bertpath):
            print('Using BERT: ' + args.bertpath)
            bert_embed = BERTEmbedding(args.bertpath,
                                   task=kashgari.CLASSIFICATION,
                                   # trainable=True,
                                   sequence_length=30)
        else:
            raise ValueError("Invalid BertPath! " + args.bertpath )

        if str(args.model).lower() == 'bilstm':
            print('Use BiLSTM Architecture')
            MODELFILENAME = 'BiLSTM'
            hyper = BiLSTM_Model.get_default_hyper_parameters()
            hyper['layer_bi_lstm']['units'] = 64
            model = BiLSTM_Model(bert_embed, hyper_parameters=hyper)

        elif str(args.model).lower() == 'custom':
            print('Use BiLSTM Architecture')
            MODELFILENAME = 'CUSTOM'
            hyper = DoubleBLSTMModel.get_default_hyper_parameters()
            #started working directly in model definition
            model = DoubleBLSTMModel(bert_embed, hyper_parameters=hyper)

        else:
            raise ValueError("So far only bilstm or custom model available for BERT embeddings!")


        model.fit(train_x, train_y, valid_x, valid_y, epochs=int(args.epochs),
                  batch_size=(args.batchsize))  # , fit_kwargs={'initial_epoch': 4}
        model.evaluate(test_x, test_y)

    elif args.action == 'load' and os.path.exists(args.existingModelPath):
        print('Continuing Traing with ' + args.existingModelPath)
        model = kashgari.utils.load_model(args.existingModelPath)
        model.compile_model()

        MODELFILENAME = 'CONTINUATION_MODEL'

    else:
        raise ValueError("Invalid action or not existing \"existingModelPath\" need!")




elif args.embedding.lower() == 'w2v':
    print('Using Word2Vec Embeddings')

    model = createWord2VecModel(train_sequences, input_data['target'].tolist(), epochs_model=int(args.epochs))
    MODELFILENAME = 'WORD2VEC'

elif args.embedding.lower() == 'pytorchbert':
    model, tokenizer = train_Pytorch_BERT(tweets, input_data['target'].tolist())
    MODELFILENAME = 'pytorch_BERT'

else:
    raise ValueError("Parameter --embedding for the model type must be either \"BERT\" or \"W2V\"!")




#Save Model
timestamp = str(datetime.now().strftime("%Y%m%d_%H-%M-%S"))
modelfilePath = 'models/' + MODELFILENAME + '-' + timestamp

if args.embedding.lower() == 'pytorchbert':
    os.mkdir(modelfilePath)
    output_model_file = os.path.join(modelfilePath, WEIGHTS_NAME)
    output_config_file = os.path.join(modelfilePath, CONFIG_NAME)
    torch.save(model.state_dict(), output_model_file)
    model.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(modelfilePath)
else:
    model.save(modelfilePath)

#%%
#CODE FOR Reloading PYTORCH MODEL --> WILL BE IMPLEMENTED IN THE FINAL APPLICATION SINCE ITS THE BEST MODEL
model_state_dict = torch.load(output_model_file)
loaded_model = BertForSequenceClassification.from_pretrained(modelfilePath, state_dict=model_state_dict, num_labels=2)









