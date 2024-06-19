#### Class for Dictionary
## Receives: 
import pickle
import nlp
import pandas as pd
import gensim.corpora as corpora

class Dictionary: # dictionary

    def __init__(self, 
                 dicpath,
                min_count = 5,
                thr = 100,
                scorfun = 'default',
                no_below = 5,
                no_above = 0.5,
                keep_n = 100000,
                lemmatized_texts = [],
                from_pickle = True):
        
        self.from_pickle = from_pickle

        if not from_pickle:
            data_bigrams_trigrams = nlp.make_multigrams(lemmatized_texts, min_count = min_count, threshold = thr, scoring = scorfun)
            self.id2word = nlp.make_id2word(data_bigrams_trigrams, dicpath, no_below, no_above, keep_n)            
        else:
            self.id2word = corpora.Dictionary.load_from_text(dicpath)

        print(f'I have just created a dictionary with length {len(self.id2word)}.')