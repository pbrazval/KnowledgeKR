
import os
from pathlib import Path
import requests
import random
import time
import pandas as pd

class The10KDownloader():
    def __init__(self, yearrange = range(2000, 2023), redo=False):
        self.redo = redo
        self.yearrange = list(yearrange)
        self.qtrrange = [1,2,3,4]
        if self.redo:
            self.downloadIndexFiles()
            self.loopThruYearLists()
        print("FileLoader initialized")
    
    def downloadIndexFiles(self):
        for year in self.yearrange:
            for qtr in self.qtrrange:
                wait_time = random.uniform(5, 15)
                time.sleep(wait_time)
                print(f"Time to retrieve {year}Q{qtr}")
                thisURL = self.urlname(year, qtr)
                response = requests.get(thisURL, headers={'User-Agent': 'Mozilla/5.0'})

                filename = self.address2save(year, qtr)
                if not os.path.isfile(filename):
                    # File doesn't exist, create it
                    with open(filename, "wb") as file:
                        file.write(response.content)
                else:
                    # File exists, overwrite content
                    with open(filename, "w") as file:
                        file.write(response.content.decode())                  
    
    @staticmethod
    def urlname(year, qtr):
        url = f'https://www.sec.gov/Archives/edgar/full-index/{year}/QTR{qtr}/company.idx'
        return url
    
    @staticmethod
    def address2save(year, qtr):
        address = f"/Users/pedrovallocci/Documents/PhD (local)/Research/By Topic/Measuring knowledge capital risk/input/company/{year}Q{qtr}.idx"
        return address
    
    @staticmethod
    def createfile(filename, content):
            name= filename + ".txt"  # Here we define the name of the file
            with open(name, "w") as file:
                    file.write(str(content)) # Here we define its content, which will be the textual content from the 10-K files.
                    file.close()
                    print(f"Success! We're in {year}Q{qtr}. Iteration {a_index}. Firm {company_names[a_index]}. ")
    
    def loopThruYearLists(self):
        for year in self.yearrange:
            for qtr in self.qtrrange:
                parent_dir = os.path.dirname(os.getcwd())
                filename = f'/Users/pedrovallocci/Documents/PhD (local)/Research/By Topic/Measuring knowledge capital risk/input/company/{year}Q{qtr}.idx'
                with open(filename, 'r', encoding='utf-8', errors='replace') as file:
                    index_file = file.readlines()
                find_list = []
                item = 0
                line = 0

                while True:
                    i = index_file[line]
                    loc1 = i.find('10-K')
                    loc2 = i.find("NT 10-K") 
                    loc3 = i.find("10-K/A")

                #We strictly keep 10-K files, not NT 10-K or 10-K/A
                    if (loc1 != -1) and (loc2 == -1)  and (loc3 == -1):
                        find_list.append(i)
                    line+=1
                    item = len(find_list)
                    if line >= len(index_file):
                        break

                self.saveResultsToCSV(year, qtr, find_list)
                
                self.download10Ks(year, qtr, find_list)

    def download10Ks(self, year, qtr, find_list):
        if self.redo:
            folder_name = f"/Users/pedrovallocci/Documents/PhD (local)/Research/By Topic/Measuring knowledge capital risk/output/10-K files/{year}/Q{qtr}/"
            Path(folder_name).mkdir(parents=True, exist_ok=True)
            ReportList = []
            Company_No = []

            for i in find_list:
                split_i = i.split()
                ReportList.append("https://www.sec.gov/Archives/" + split_i[-1])
                Company_No.append(split_i[-3] + "_" + split_i[-2])

            os.chdir(folder_name)
            company_order = 0
            unable_request = 0

            for a_index in range(len(ReportList)):
                web_add = ReportList[a_index]
                filename = Company_No[a_index]

                webpage_response = requests.get(web_add, headers={'User-Agent': 'Mozilla/5.0'}) 
                        # It is very important to use the header, otherwise the SEC will block the requests after the first 5.

                if webpage_response.status_code == 200: 
                            # The HTTP 200 OK success status response code indicates that the request has succeeded. 
                    body = webpage_response.content
                    self.createfile(filename, body)
                else:
                    print ("Unable to get response with Code : %d " % (webpage_response.status_code))
                    unable_request += 1

                a_index +=1

            print(unable_request)

    def saveResultsToCSV(self, year, qtr, find_list):
        filenames = []
        sec_filenames = []
        company_names = []
        cik = []
        for line in find_list:
            split_i = line.split()
            cik.append(split_i[-3])
            filenames.append(split_i[-3] + "_" + split_i[-2]+ ".txt")
            sec_filenames.append("https://www.sec.gov/Archives/" + line.split()[-1])
            company_names.append(line.split('10-K')[0].strip())

        df = pd.DataFrame({'company': company_names, 'filename': filenames, 'sec_filename': sec_filenames, 'cik': cik})
        df.to_csv(f'/Users/pedrovallocci/Documents/PhD (local)/Research/By Topic/Measuring knowledge capital risk/output/firmdict/{year}Q{qtr}.csv')