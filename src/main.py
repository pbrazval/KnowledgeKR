import data_loading as dl
import nlp
import importlib
yearrange = range(2006,2007)
download_10ks_again = False

if __name__ == '__main__':
    # dl.The10KDownloader(yearrange = yearrange, redo = download_10ks_again)
    # dl.The1AConverter(yearrange = yearrange, redo = download_10ks_again)
    ngr = nlp.NGrammer(yearrange=yearrange, from_pickle=False) 
    dic = nlp.Dictionary(ngr, dicpath = 'output/dictionary.txt', from_pickle=False)
    bow = nlp.BagOfWords(ngr, dic)
    lda = nlp.LDA(bow, num_topics = 10, redo = False)
    lda.visualize_topics()
    lda.create_topic_map()
    lda.print_coherence()