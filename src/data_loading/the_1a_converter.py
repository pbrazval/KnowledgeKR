import multiprocessing as mp
import pandas as pd
import re
import os
from bs4 import BeautifulSoup
from os import listdir
from os.path import isfile, join

class The1AConverter:
    def __init__(self, 
                 yearrange, 
                 redo = False,
                 my10kpath_template = "/Users/pedrovallocci/Library/Mobile Documents/com~apple~CloudDocs/Documentos/PhD/Research/By Topic/Interpreting KR using 10-Ks/output/10-K files iCloud storage/{yr}/Q{qtr}/", 
                 my1apath_template = "/Users/pedrovallocci/Documents/PhD (local)/Research/By Topic/Measuring knowledge capital risk/output/1A files/{yr}/Q{qtr}/", 
                 qtrrange = [1,2,3,4], 
                 num_processes=mp.cpu_count()):
        self.yearrange = list(yearrange)
        self.qtrrange = qtrrange
        self.num_processes = num_processes
        self.mypath_template = my10kpath_template
        self.my1apath_template = my1apath_template
        self.redo = redo
        if self.redo:
            self.process_files()

    def convert10kcorpus(self, onlyfiles, mypath, my1apath):
        pool = mp.Pool(self.num_processes)
        results = []
        for file in onlyfiles:
            results.append(pool.apply_async(self.convertto1a, args=(file, mypath, my1apath)))
        pool.close()
        pool.join()
        return True

    def process_files(self):
        mypath_template = self.mypath_template
        my1apath_template = self.my1apath_template
        for yr in self.yearrange:
            for qtr in self.qtrrange:
                mypath = mypath_template.format(yr=yr, qtr=qtr)
                my1apath = my1apath_template.format(yr=yr, qtr=qtr)
                onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f)) and not f.startswith('._')]
                self.convert10kcorpus(onlyfiles, mypath, my1apath)

    @staticmethod
    def return1atext(fileaddress):
        file = open(fileaddress, "r")
        raw_10k = file.read()
        doc_start_pattern = re.compile(r'<SEC-DOCUMENT>')
        doc_end_pattern = re.compile(r'</SEC-DOCUMENT>')
        type_pattern = re.compile(r'<TYPE>[^\n]+')

        doc_start_is = [x.end() for x in doc_start_pattern.finditer(raw_10k)]
        doc_end_is = [x.start() for x in doc_end_pattern.finditer(raw_10k)]

        doc_types = [x[len('<TYPE>'):] for x in type_pattern.findall(raw_10k)]

        document = {}
        doc_start = doc_start_is[0]
        doc_end = doc_end_is[0]
        document['10-K'] = raw_10k[doc_start:doc_end]
        
        regex = re.compile(r'>\s*[Ii][Tt][Ee][Mm](\s|&#160;|&nbsp;|\\n)*[12]\s*[AaBb]?\.{0,1}')
        
        matches = regex.finditer(document['10-K'])
        test_df = pd.DataFrame([(x.group(), x.start(), x.end()) for x in matches])

        test_df.columns = ['item', 'start', 'end']
        test_df['item'] = test_df.item.str.lower()

        # Display the dataframe

        test_df.replace('&#160;',' ',regex=True,inplace=True)
        test_df.replace('&nbsp;',' ',regex=True,inplace=True)
        test_df.replace('\\\\n',' ',regex=True,inplace=True)
        test_df.replace('\\n',' ',regex=True,inplace=True)
        test_df.replace(' ','',regex=True,inplace=True)
        test_df.replace('\.','',regex=True,inplace=True)
        test_df.replace('>','',regex=True,inplace=True)


        if any(test_df['item'] == 'item1a'):
            pass
            #print("The column 'item' contains the value 'item1a'")
        else:
            raise AssertionError("There's no 1a here.")


        if any(test_df['item'] == 'item1b'):
            after1a = 'item1b'
        elif any(test_df['item'] == 'item2'):
            after1a = 'item2'
        else:
            raise AssertionError("There's an 1a here, but no 1b or 2 here.")

        pos_dat = test_df.sort_values('start', ascending=True).drop_duplicates(subset=['item'], keep='last')

        pos_dat.set_index('item', inplace=True)

        # # Get Item 1a
        item_1a_raw = document['10-K'][pos_dat['start'].loc['item1a']:pos_dat['start'].loc[after1a]]

        item_1a_content = BeautifulSoup(item_1a_raw,  "html.parser")

        text = item_1a_content.get_text()
        return text
    
    @staticmethod
    def convertto1a(shortfile, mypath, my1apath):
        file = shortfile
        try:
            thistext = The1AConverter.return1atext(mypath+file)
            print("Success")
        except Exception as e:
            print(f'Exception in file {file} occurred.')
            print("An error occurred: {}".format(e))
            return None
        else:
            if not os.path.exists(my1apath):
                os.makedirs(my1apath)
            with open(my1apath+file, 'w') as f:
                f.write(thistext)
        return None

if __name__ == '__main__':    
    converter = The1AConverter(yearrange=range(2013, 2014, 1), 
                            qtrrange=[2],
                            my1apath_template = "/Users/pedrovallocci/Documents/PhD (local)/Research/By Topic/Measuring knowledge capital risk/output/1A files_test_class/{yr}/Q{qtr}/")
    converter.process_files()

