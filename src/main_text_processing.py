import data_loading as dl
import nlp
import importlib
yearrange = range(2006,2023)
download_10ks_again = False

if __name__ == '__main__':
    if False:
        dl.The10KDownloader(yearrange = yearrange, redo = download_10ks_again)
        dl.The1AConverter(yearrange = yearrange, redo = download_10ks_again)
        ngr = nlp.NGrammer(yearrange=yearrange, from_pickle=True) 
        print("NGrammer initialized")
        dic = nlp.Dictionary(ngr, dicpath = 'output/dictionary.txt', from_pickle=False) # Need to test
    bow = nlp.BagOfWords(dicname = "dicfullmc10thr10defnob40noa0_8", from_pickle = True)
    lda = nlp.LDA(bow, None, modelname = "dicfullmc10thr10defnob40noa0_8_10_t", redo = False)
    # lda.visualize_topics()
    # lda.create_topic_map()
    # lda.print_coherence()