import data_loading as dl
import nlp
import importlib
yearrange = range(2006,2007)
download_10ks_again = False

# dl.The10KDownloader(yearrange = yearrange, redo = download_10ks_again)
# dl.The1AConverter(yearrange = yearrange, redo = download_10ks_again)
if __name__ == '__main__':
    ngr = nlp.NGrammer(yearrange=yearrange, from_pickle=False) 
    dic = nlp.Dictionary(ngr, dicpath = 'output/dictionary.txt', from_pickle=False)
    bow = nlp.BagOfWords(ngr, dic)