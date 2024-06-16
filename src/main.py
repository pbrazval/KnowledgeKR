import data_loading as dl
import importlib
importlib.reload(ut)
yearrange = range(2004,2005)
download_10ks_again = False

dl.The10KDownloader(yearrange = yearrange, redo = download_10ks_again)
dl.The1AConverter(yearrange = yearrange, redo = download_10ks_again)