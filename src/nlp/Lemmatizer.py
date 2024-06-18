import multiprocessing as mp
import os
import pickle
import re
import pandas as pd
import numpy as np
import string
import glob

class Lemmatizer:
    def __init__(self, 
                 yearlist = range(2006,2023),
                 cequity_mapper = '/Users/pedrovallocci/Documents/PhD (local)/Research/By Topic/Measuring knowledge capital risk/output/cequity_mapper.csv',
                 source_1a = '/Users/pedrovallocci/Documents/PhD (local)/Research/By Topic/Measuring knowledge capital risk/output/1A files',
                 min10kwords = 200, 
                 load_from_pickle = True,
                 num_processes=mp.cpu_count()):
        self.yearlist = yearlist
        self.min10kwords = min10kwords
        self.cequity_mapper = cequity_mapper
        self.num_processes = num_processes
        self.source_1a = source_1a
        self.load_from_pickle = load_from_pickle
        if not self.load_from_pickle:
            self.lemmatized_texts, self.idxs_to_keep, self.ciks_to_keep = self.create_lemmatized_texts()
        else:
            self.lemmatized_texts, self.idxs_to_keep, self.ciks_to_keep, self.yr_vec = self.retrieve_lemmatized_texts()
            
    def create_lemmatized_texts(self):
        for yr in self.yearlist:
            print(f"Starting year {yr}")
            texts, filename_list = self.clean_files_mp(yr)
            selection, idxs_to_keep, ciks_to_keep = self.filter_corpus(texts, filename_list, yr)
            lemmatized_texts = self.lemmatization(texts, selection, yr)
            filelist = []
            for qtr in range(1, 5):
                pattern = f'{self.source_1a}/{yr}/Q{qtr}/*.txt'
                filelist.extend(glob.glob(pattern))
            self.create_crosswalks(filelist, yr)
        return lemmatized_texts, idxs_to_keep, ciks_to_keep
    
    def retrieve_lemmatized_texts(self):
        lemmatized_texts = []
        yr_vec = []
        idxs_to_keep = pd.Series()
        ciks_to_keep = pd.Series()
        for yr in self.yearlist:
            file_path = f"/Users/pedrovallocci/Documents/PhD (local)/Research/By Topic/Measuring knowledge capital risk/output/lemmatized_texts/{yr}/lemmatized_texts{yr}.pkl"
            # Load the file using pickle
            with open(file_path, 'rb') as f:
                lemmatized_texts = lemmatized_texts + pickle.load(f)
            filter_path = f"/Users/pedrovallocci/Documents/PhD (local)/Research/By Topic/Measuring knowledge capital risk/output/lemmatized_texts/{yr}/lem_filter{yr}.pkl"
            with open(filter_path, 'rb') as f:
                selection = pickle.load(f)
            idxs_to_keep = idxs_to_keep.append(selection['order_in_cik'])
            ciks_to_keep = ciks_to_keep.append(selection['cik'])
            yr_vec = yr_vec + [yr for _ in selection['cik']]
        print(f"Length of lemmatized_texts is: {len(lemmatized_texts)}")
        return lemmatized_texts, idxs_to_keep, ciks_to_keep, yr_vec
    
    def lemmatization(self, old_texts, selection, yr):
        idxs_to_keep = selection['order_in_cik']
        print(f"Starting lemmatization")
        pool = mp.Pool(self.num_processes)
        results = []
        texts = [old_texts[i] for i in idxs_to_keep]
        for i, text in enumerate(texts):
            results.append(pool.apply_async(self.lemmatize_text, args=(text,)))
            # Print progress: every 500 iterations print the iteration number
            if (i+1) % 500 == 0:
                print(f"Lemmatized {i+1} texts so far")
        pool.close()
        pool.join()
        texts_out = [r.get() for r in results]
        path = f"/Users/pedrovallocci/Documents/PhD (local)/Research/By Topic/Measuring knowledge capital risk/output/lemmatized_texts/{yr}"
    # Create the directory if it doesn't exist
        if not os.path.exists(path):
            os.makedirs(path)
        # Save the vector as a pickle file
        with open(os.path.join(path, f"lemmatized_texts{yr}.pkl"), "wb") as f:
            pickle.dump(texts_out, f)
        
        with open(os.path.join(path, f"lem_filter{yr}.pkl"), "wb") as f:
            pickle.dump(selection, f)
        
        return texts_out
    
    @staticmethod
    def clean_file(filename):
        with open(filename) as f:
            doc = f.read().splitlines() 
        doc = filter(None, doc) # remove empty string
        doc = '. '.join(doc)
        doc = doc.replace("\\n", " ")
        doc = doc.translate(str.maketrans('','',string.punctuation))
        doc = doc.translate(str.maketrans('','','1234567890'))      
        doc = doc.encode('ascii', 'ignore') # ignore fancy unicode chars
        doc = str(doc)
        return (filename, doc)

    def clean_files_mp(self, yr):
        print(f"Creating texts")
        texts = []
        i = 0
        filename_list = []
        results = []
        for qtr in [1,2,3,4]:
            pool = mp.Pool(mp.cpu_count())
            print(f"Pool started with {mp.cpu_count()} cores. Qtr {qtr}")
            for filename in glob.glob(f'{self.source_1a}/{yr}/Q{qtr}/*.txt'):
                results.append(pool.apply_async(self.clean_file, args=(filename,)))
                i += 1
                if i % 500 == 0:
                    print(f"Created {i} texts so far")
            pool.close()
            pool.join()
        for r in results:
            result = r.get()
            filename, doc = result
            filename_list.append(filename)    
            texts.append(doc)                 
        return texts, filename_list

    def filter_corpus(self, texts, filename_list, yr):

        cequity_mapper = pd.read_csv(self.cequity_mapper)
        
        text_length = [len(text) for text in texts]
        
        cik = [int(re.search(r'/(\d+)_', fn).group(1)) for fn in filename_list]
        
        if yr <= 2020:
            cequity_mapper = cequity_mapper[cequity_mapper['year'] == yr]
        else:
            cequity_mapper = cequity_mapper[cequity_mapper['year'] == 2020]        
        
        order_in_cik = list(range(len(cik)))
        stats_texts = pd.DataFrame({"order_in_cik": order_in_cik, "cik": cik, "text_length": text_length})
        fullfilter = pd.merge(stats_texts, cequity_mapper, on="cik", how="inner")
        fullfilter['crit_LEN'] = fullfilter['text_length'] > self.min10kwords
        fullfilter['crit_ALL'] = fullfilter['crit_ALL'] == 1
        fullfilter['crit_ALL2'] = list(np.logical_and(np.array(fullfilter['crit_ALL']),np.array(fullfilter['crit_LEN'])))
        selection = fullfilter[fullfilter['crit_ALL2']]
        selection = selection.drop_duplicates(subset = "cik", keep = "first")
        idxs_to_keep = selection['order_in_cik']
        ciks_to_keep = selection['cik']    
        
        return selection, idxs_to_keep, ciks_to_keep
    
    @staticmethod
    def lemmatize_text(text):
        if text is None:
            return None
        else:
            text = text.replace("\\n", "  ")
            allowed_postags=["NOUN", "ADJ", "VERB", "ADV"]
            nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
            nlp.max_length = 2000000 
            doc = nlp(text)
            new_text = []
            for token in doc:
                if token.pos_ in allowed_postags:
                    new_text.append(token.lemma_)
            return " ".join(new_text)
    
    @staticmethod
    def create_crosswalks(filename_list, yr):
        print(f"Creating cross walks for year {yr}")
        fn_list = [fn.split('/')[-1] for fn in filename_list]
        idx_list = list(range(len(filename_list)))
        fn2idx = pd.DataFrame({"idx": idx_list, "filename": fn_list})
        fn2cp = []
        for qtr in [1,2,3,4]:
            fn2cp.append(pd.read_csv(f'/Users/pedrovallocci/Documents/PhD (local)/Research/By Topic/Measuring knowledge capital risk/output/firmdict/{yr}Q{qtr}.csv'))
        fn2cp = pd.concat(fn2cp)
        merged_df = pd.merge(fn2idx, fn2cp, on="filename", how="outer")
        merged_df.to_csv(f'/Users/pedrovallocci/Documents/PhD (local)/Research/By Topic/Measuring knowledge capital risk/output/cp2idx/{yr}.csv')
        return None    