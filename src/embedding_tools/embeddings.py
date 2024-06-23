import pandas as pd
import numpy as np
import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import embedding_tools as et
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt

class Embeddings:
    def __init__(self, from_pickle = True):
        
        self.from_pickle = from_pickle
        if from_pickle:
            self.data = Embeddings.from_pickle()
        else:
            self.data = Embeddings.rebuild()
        # Preserve the columns 'year' and 'CIK'

    @staticmethod
    def rebuild():
        years = range(2006, 2022 + 1)

        # Initialize empty lists to store embeddings, years, and CIKs
        all_embeddings = []
        all_years = []
        all_ciks = []

        # Loop through each year to load the embeddings and CIKs
        for year in years:
            # Load embeddings
            if year == 2012 or year == 2006:
                embeddings_file_path = f"/Users/pedrovallocci/Documents/PhD (local)/Research/By Topic/Measuring knowledge capital risk/output/1A_files_cleaned/{year}/embeddings{year}.pkl"
                embeddings = pd.read_pickle(embeddings_file_path)
                embeddings = np.array(embeddings)
                all_embeddings.append(embeddings)
            elif year > 2012:
                embeddings_file_path = f"/Users/pedrovallocci/Documents/PhD (local)/Research/By Topic/Measuring knowledge capital risk/output/1A_files_cleaned/{year}/embeddings{year}.npy"
                embeddings = np.load(embeddings_file_path)
                all_embeddings.append(embeddings)
            
            # Load CIKs
            pickle_file_path = f"/Users/pedrovallocci/Documents/PhD (local)/Research/By Topic/Measuring knowledge capital risk/output/1A_files_cleaned/{year}/clean_texts_filter{year}.pkl"
            clean_texts_filter = pd.read_pickle(pickle_file_path)
            ciks = clean_texts_filter['cik']
            
            # Append embeddings, years, and CIKs to the lists
            
            all_years.append(np.full(ciks.shape[0], year))
            all_ciks.append(ciks)

        # Concatenate all embeddings, years, and CIKs along the first axis (rows)
        all_embeddings_concatenated = np.concatenate(all_embeddings, axis=0)
        all_years_concatenated = np.concatenate(all_years, axis=0)
        all_ciks_concatenated = np.concatenate(all_ciks, axis=0)

        # Create a DataFrame from the concatenated embeddings
        embedding_df = pd.DataFrame(all_embeddings_concatenated)

        # Add the year and CIK columns to the DataFrame
        embedding_df['year'] = all_years_concatenated
        embedding_df['CIK'] = all_ciks_concatenated
        # Save the DataFrame to a pickle file in the data folder, sister of the src folder:
        save_path = os.path.join('..', 'data', 'embeddings.pkl')
        embedding_df.to_pickle(save_path)
        print("Embeddings saved to data/embeddings.pkl")
        return embedding_df
    
    @staticmethod
    def from_pickle():
        # Load the embeddings DataFrame from the pickle file
        load_path = os.path.join('..', 'data', 'embeddings.pkl')
        embeddings = pd.read_pickle(load_path)
        print("Embeddings loaded from data/embeddings.pkl")
        return embeddings

