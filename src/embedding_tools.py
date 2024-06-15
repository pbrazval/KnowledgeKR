import mpfiles
import multiprocessing as mp
import glob
import importlib
import utilities as ut
importlib.reload(ut)
import pandas as pd
import os
import pickle
import openai
import tiktoken
import numpy as np
from concurrent.futures import ProcessPoolExecutor
# Make sure to replace 'your-api-key' with your actual OpenAI API key

key_file_path = os.path.join('..', 'data', 'apikey.txt')

# Read the key from the file
with open(key_file_path, 'r') as file:
    api_key = file.read().strip()

client = openai.OpenAI(api_key=api_key)

def get_text_embeddings(text):
    text = text.replace("\n", " ")
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    max_tokens = 8192
    truncated_tokens = tokens[:max_tokens]
    truncated_text = tokenizer.decode(truncated_tokens)
    return client.embeddings.create(input = [truncated_text], model="text-embedding-3-small").data[0].embedding
    
def create_texts_shortmp(yr):
    print(f"Creating texts")
    texts = []
    i = 0
    filename_list = []
    results = []
    for qtr in [1,2,3,4]:
        #pool = mp.Pool(6)
        #print(f"Pool started with {6} cores. Qtr {qtr}")
        for filename in glob.glob(f'/Users/pedrovallocci/Documents/PhD (local)/Research/By Topic/Measuring knowledge capital risk/output/1A files/{yr}/Q{qtr}/*.txt'):
            #results.append(pool.apply_async(mpfiles.process_file_short, args=(filename,)))
            results.append(mpfiles.process_file_short(filename))
            i += 1
            if i % 500 == 0:
                print(f"Created {i} texts so far")
            # Save to the same filename, but to directory 1A_files_cleaned:
            # new_filename = filename.replace('1A files', '1A_files_cleaned')
            # #print(f"New filename: {new_filename}")
            # with open(new_filename, 'w') as f:
            #    f.write(doc)
            
        #pool.close()
        #pool.join()
    for r in results:
        filename, doc = r
        filename_list.append(filename)    
        texts.append(doc)   
        

    return texts, filename_list


def filter(elements, idxs_to_keep):
    #print(f"Starting to filter texts")
    #results = []
    filtered_elements = [elements[i] for i in idxs_to_keep]
    # for i, text in enumerate(texts):
    #     results.append(pool.apply_async(mpfiles.lemmatize_text, args=(text,)))
    #     # Print progress: every 500 iterations print the iteration number
    #     if (i+1) % 500 == 0:
    #         print(f"Lemmatized {i+1} texts so far")

    # texts_out = [r.get() for r in results]
    # path = f"/Users/pedrovallocci/Documents/PhD (local)/Research/By Topic/Measuring knowledge capital risk/output/lemmatized_texts/{yr}"
    # # Create the directory if it doesn't exist
    # if not os.path.exists(path):
    #     os.makedirs(path)
    # # Save the vector as a pickle file
    # with open(os.path.join(path, f"lemmatized_texts{yr}.pkl"), "wb") as f:
    #     pickle.dump(texts_out, f)
    
    # with open(os.path.join(path, f"lem_filter{yr}.pkl"), "wb") as f:
    #     pickle.dump(selection, f)
    return filtered_elements

# For a given text, return the embeddings from OpenAI's text-embedding-3-small model:


class Dataset:
    def __init__(self):
        self.cequity_mapper = pd.read_csv("/Users/pedrovallocci/Documents/PhD (local)/Research/By Topic/Measuring knowledge capital risk/input/cequity_mapper.csv")
        self.min10kwords = 200
        self.filtered_texts = []
        self.ciks_to_keep = []
        self.year = []
        self.embeddings = []
        #self.filename_list = []
        #self.texts = []
        # Load all the dirty 1As from the folder:
    
    def from_pkls(self):
        with open("/Users/pedrovallocci/Documents/PhD (local)/Research/By Topic/Measuring knowledge capital risk/output/filtered_texts.pkl", "rb") as f:
            self.filtered_texts = pickle.load(f)
        with open("/Users/pedrovallocci/Documents/PhD (local)/Research/By Topic/Measuring knowledge capital risk/output/ciks_to_keep.pkl", "rb") as f:
            self.ciks_to_keep = pickle.load(f)
        with open("/Users/pedrovallocci/Documents/PhD (local)/Research/By Topic/Measuring knowledge capital risk/output/year.pkl", "rb") as f:
            self.year = pickle.load(f)
        print("Loaded all the filtered texts, CIKs and years from the pickle files")
        return self

    def from_texts(self):
        for yr in range(2006, 2023):
            print(f"Loading all the dirty 1As from the year {yr}")
            texts, filename_list = create_texts_shortmp(yr)
            selection, idxs_to_keep, ciks_to_keep = ut.filter_corpus(texts, filename_list, self.cequity_mapper, yr, self.min10kwords)
            filtered_texts = [texts[i] for i in idxs_to_keep]
            filtered_filenames = [filename_list[i] for i in idxs_to_keep]
            self.save_filtered_texts_as_txt(yr, filtered_texts, filtered_filenames)
            self.save_filtered_texts_and_selection_as_pkl(yr, filtered_texts, selection)
            #self.texts.extend(texts)            
            self.filtered_texts.extend(filtered_texts)
            self.ciks_to_keep.extend(ciks_to_keep)
            self.year.extend([yr]*len(texts))
        # Save self.filtered_texts, self.ciks_to_keep and self.year to a pickle file
        with open("/Users/pedrovallocci/Documents/PhD (local)/Research/By Topic/Measuring knowledge capital risk/output/filtered_texts.pkl", "wb") as f:
            pickle.dump(self.filtered_texts, f)
        with open("/Users/pedrovallocci/Documents/PhD (local)/Research/By Topic/Measuring knowledge capital risk/output/ciks_to_keep.pkl", "wb") as f:
            pickle.dump(self.ciks_to_keep, f)
        with open("/Users/pedrovallocci/Documents/PhD (local)/Research/By Topic/Measuring knowledge capital risk/output/year.pkl", "wb") as f:
            pickle.dump(self.year, f)
        return self
        
        print("Everything saved to pickle files")
    
    def create_embeddings(self, init_year, end_year):
        #embeddings = []
        for yr in range(init_year, end_year):
            
            with open(f"/Users/pedrovallocci/Documents/PhD (local)/Research/By Topic/Measuring knowledge capital risk/output/1A_files_cleaned/{yr}/clean_texts{yr}.pkl", "rb") as f:
                texts = pickle.load(f)
            
            num_texts = len(texts)
            
            embeddings = [None] * num_texts
            for j, text in enumerate(texts):
                embeddings[j] = get_text_embeddings(text)
                # Report every 500 texts:
                if j % 200 == 0:
                    print(f"Created {j} embeddings so far out of {num_texts} for the year {yr}")
            # Save the embeddings to a pickle file
            embeddings_matrix = np.array(embeddings)
            np.save(f"/Users/pedrovallocci/Documents/PhD (local)/Research/By Topic/Measuring knowledge capital risk/output/1A_files_cleaned/{yr}/embeddings{yr}.npy", embeddings_matrix)
            # with open(f"/Users/pedrovallocci/Documents/PhD (local)/Research/By Topic/Measuring knowledge capital risk/output/1A_files_cleaned/{yr}/embeddings{yr}.pkl", "wb") as f:
            #     pickle.dump(embeddings, f)
        # for text, year in zip(self.filtered_texts, self.year):
        #     embeddings.append(get_text_embeddings(text))
        #     # Save the embeddings to a pickle file
        #     with open(f"/Users/pedrovallocci/Documents/PhD (local)/Research/By Topic/Measuring knowledge capital risk/output/1A_files_cleaned/{yr}/embeddings{yr}.pkl", "wb") as f:
        #         pickle.dump(embeddings, f)
    
    # from concurrent.futures import ProcessPoolExecutor

    # def create_embeddings_parallel(self, init_year, end_year):
    #     def process_text(text):
    #         return get_text_embeddings(text)

    #     i = 1
    #     for yr in range(init_year, end_year):
    #         embeddings = []
    #         with open(f"/Users/pedrovallocci/Documents/PhD (local)/Research/By Topic/Measuring knowledge capital risk/output/1A_files_cleaned/{yr}/clean_texts{yr}.pkl", "rb") as f:
    #             texts = pickle.load(f)
            
    #         # Use ProcessPoolExecutor to parallelize the embedding generation
    #         with ProcessPoolExecutor() as executor:
    #             results = list(executor.map(process_text, texts))
            
    #         embeddings.extend(results)

    #         # Report progress
    #         i += len(texts)
    #         print(f"Created {i} embeddings so far")
            
    #         # Convert the list of embeddings to a NumPy array
    #         embeddings_matrix = np.array(embeddings)
            
    #         # Save the NumPy array to a file
    #         np.save(f"/Users/pedrovallocci/Documents/PhD (local)/Research/By Topic/Measuring knowledge capital risk/output/1A_files_cleaned/{yr}/embeddings{yr}.npy", embeddings_matrix)

    def save_filtered_texts_as_txt(self, yr, filtered_texts, filtered_filenames):
        if not os.path.exists(f"/Users/pedrovallocci/Documents/PhD (local)/Research/By Topic/Measuring knowledge capital risk/output/1A_files_cleaned/{yr}"):
            os.makedirs(f"/Users/pedrovallocci/Documents/PhD (local)/Research/By Topic/Measuring knowledge capital risk/output/1A_files_cleaned/{yr}")
            os.makedirs(f"/Users/pedrovallocci/Documents/PhD (local)/Research/By Topic/Measuring knowledge capital risk/output/1A_files_cleaned/{yr}/Q1")
            os.makedirs(f"/Users/pedrovallocci/Documents/PhD (local)/Research/By Topic/Measuring knowledge capital risk/output/1A_files_cleaned/{yr}/Q2")
            os.makedirs(f"/Users/pedrovallocci/Documents/PhD (local)/Research/By Topic/Measuring knowledge capital risk/output/1A_files_cleaned/{yr}/Q3")
            os.makedirs(f"/Users/pedrovallocci/Documents/PhD (local)/Research/By Topic/Measuring knowledge capital risk/output/1A_files_cleaned/{yr}/Q4")   
            # Save each filtered text to the same path but in folder 1A_files_cleaned:
        for i, text in enumerate(filtered_texts):
            with open(filtered_filenames[i].replace('1A files', '1A_files_cleaned'), 'w') as f:
                f.write(text)

    def save_filtered_texts_and_selection_as_pkl(self, yr, texts, selection):
        path = f"/Users/pedrovallocci/Documents/PhD (local)/Research/By Topic/Measuring knowledge capital risk/output/1A_files_cleaned/{yr}"
            # Create the directory if it doesn't exist
        if not os.path.exists(path):
            os.makedirs(path)
            # Save the vector as a pickle file
        with open(os.path.join(path, f"clean_texts{yr}.pkl"), "wb") as f:
            pickle.dump(texts, f)
            
        with open(os.path.join(path, f"clean_texts_filter{yr}.pkl"), "wb") as f:
            pickle.dump(selection, f)
    
    def run_fa():
        n_factors = 10
        fa = FactorAnalysis(n_components=n_factors, random_state=42, rotation="varimax")  # Adjust the number of factors based on your needs
        fa_result = fa.fit_transform(embeddings)


        # Examine the factor loadings
        # print("Factor Loadings:")
        # print(fa.components_)
        # fa.noise_variance_

    # Variance explained by each factor is the sum of the squares of the loadings of that factor
        variance_explained_by_factor = np.var(fa_result, axis=0)

    # Total variance explained by the factors
        total_variance_explained = np.sum(variance_explained_by_factor)

    # Compute the percentage of variance explained by each factor
        percentage_variance_explained = (variance_explained_by_factor / total_variance_explained) * 100

    # Output the percentage of variance explained by each factor
        for i, variance in enumerate(percentage_variance_explained):
            print(f"Factor #{i+1} explains {variance:.2f}% of the variance")

    # Output the total percentage of variance explained by all factors
        total_percentage_variance_explained = np.sum(percentage_variance_explained)
        print(f"Total variance explained by the {n_factors} factors: {total_percentage_variance_explained:.2f}%")
