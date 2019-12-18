#%%
import os
import pandas as pd
from nltk import SnowballStemmer
from nltk.corpus import stopwords

def stemAndRemoveStopwords(text, stem=False):
    stop_words = stopwords.words("english")
    stemmer = SnowballStemmer("english")
    tokens = []
    for token in text.split():
        if token not in stop_words:
            if stem:
                tokens.append(stemmer.stem(token))
            else:
                tokens.append(token)
    return " ".join(tokens)

def doPreprocessing(forWordToVec=False):
    FILEPATH = 'data/training.1600000.processed.noemoticon.csv'

    data = pd.read_csv(FILEPATH, sep=',', header=None, names=['target', 'id', 'date', 'flag', 'user', 'text'],
                       usecols=['target', 'text'], encoding='latin-1')#
    # multiple points, exclamation mark and question marks
    data['text'] = data['text'].str.replace(r'[.]+', '.', regex=True)
    data['text'] = data['text'].str.replace(r'[?]+', '?', regex=True)
    data['text'] = data['text'].str.replace(r'[!]+', '!', regex=True)
    #  all single numbers, but leave things like 8am
    data['text'] = data['text'].str.replace(r' [123456789]+ ', ' ')

    # twitter handles
    data['text'] = data['text'].str.replace(r'@[^\s]+', ' ')
    #Remove textual smileys apart so there are no single "D" left over from ":D"
    data['text'] = data['text'].str.replace(':D', ' ')
    data['text'] = data['text'].str.replace(':P', ' ')
    data['text'] = data['text'].str.replace(':p', ' ')
    data['text'] = data['text'].str.replace(':d', ' ')

    # links
    data['text'] = data['text'].str.replace(r'http\S+', ' ', case=False)
    data['text'] = data['text'].str.replace(r'www.\S+', ' ', case=False)
    #And the links where they forgot a space to separate them..
    data['text'] = data['text'].str.replace(r'\S+http\S+', ' ', case=False)
    data['text'] = data['text'].str.replace(r'\S+http\S+', ' ', case=False)

    # all hashtags but not the words after it and the rests of special chars
    data['text'] = data['text'].str.replace('[@#\"\(\)=:;]', ' ')

    # remove special chars
    data['text'] = data['text'].str.replace(r'[-%&;§=*~#]+', '', regex=True)
    data['text'] = data['text'].str.strip()
    data['target'] = data['target'].map({0: 0, 4: 1})

    if forWordToVec:
        data['text'] = data['text'].apply(lambda x: stemAndRemoveStopwords(x, stem=True))
        OutPutPathTraining = 'data/preprocessedTweetsW2Vec.csv'
    else:
        OutPutPathTraining = 'data/preprocessedTweetsBERT.csv'

    #in the end remove double spaces
    data['text'] = data['text'].str.replace(r'[ ]+', ' ', regex=True)
    data['text'] = data['text'].str.replace('  ', ' ', regex=True)

    if os.path.exists(OutPutPathTraining):
        os.remove(OutPutPathTraining)
    data.to_csv(OutPutPathTraining, header=True, index=False, encoding='')

