import pandas as pd
import sys
from pathlib import Path
import os
import csv
import shutil

DATASET_LIST = [
    'NCBI-disease',
    'BC2GM',
    'BC4CHEMD',
    'BC5CDR-chem',
    'BC5CDR-disease',
    'JNLPBA',
    'linnaeus',
    's800'
]

def make_dir():
    for dataset in DATASET_LIST:
        if not os.path.isdir("input/NERdataset_preproc/"+dataset+"/"):
            os.mkdir("input/NERdataset_preproc/"+dataset+"/")
        if not os.path.isdir("input/NERdataset_preproc_temp/"+dataset+"/"):
            os.mkdir("input/NERdataset_preproc_temp/"+dataset+"/")

def replace_blank_lines_with_token(FILEPATH, TEMPPATH):
    '''
    Blank lines here denote the ends of sentences. They are taken as NaN by 
    the pd.read_csv. But missing data is also NaN. Replace blanklines with a special token
     = [newline] to aid preprocessing
    '''
    file = open(FILEPATH, "r")
    temp = open(TEMPPATH, "w+")
    for line in file:
        if line.isspace():
            temp.write('[newline]\tX\n')
        else:
            temp.write(line)
    file.close()
    temp.close()


def preprocess():
    for dataset in DATASET_LIST:
        SOURCE = Path("input/NERdataset/"+dataset)
        DEST = Path("input/NERdataset_preproc/"+dataset)
        TEMP = Path("input/NERdataset_preproc_temp/"+dataset)
        pathlist = Path(SOURCE).glob('*.tsv')
        for path in pathlist:
            PATH = str(SOURCE/path.name)
            NEWPATH = str(DEST/path.name)
            TEMPPATH = str(TEMP/path.name)
            replace_blank_lines_with_token(PATH,TEMPPATH)
            print(PATH, '->' ,NEWPATH)
            do_preprocess(TEMPPATH,NEWPATH)
        print(dataset+' Done!')

def do_preprocess(PATH,NEWPATH):
    '''
    Preprocessing script for NCBI-disease data
    The original dataset is just a csv of words and tags
    we convert it into sentences and the tags of that sentence
    Output df: 
                Sentence              |       Tags
    ['Bert','is','a','good','model'] : ['B','O','O','O','O']
    '''
    df = pd.read_csv(PATH,delimiter='\t',names=['word','label'],quoting=csv.QUOTE_NONE)
    new_data = []
    sent, label_list = [],[]
    for r in df.itertuples(): 
        if r[1]!=r[1]: #Skip NaNs
            continue
        if r[1]=='[newline]': #Check for newlines
            if(len(sent)==0):
                continue
            if(sent[-1]!='.'):
                sent.append('.')
                label_list.append('O')
            new_data.append([sent,label_list])
            # To check that sentence length and label_list length are consistent
            assert(len(sent)==len(label_list))
            if(label_list[0]=='I'):
                print('Should not happend found - > ',sent,label_list)
                raise ValueError('The first label of a sentence in I')
    
            # To check there are no Named entities overlapping with a sentence end
            assert(label_list[0]!='I')
            sent, label_list = [],[]
        else:
            sent.append(r[1])
            try:
                label_list.append(r[2])
            except:
                print(r)
                raise ValueError('out of range?')
        
    df_new = pd.DataFrame(new_data,columns = ['sentence','labels'])
    df_new.to_csv(NEWPATH, index=False)

if __name__ == "__main__":
    if not os.path.isdir("input/NERdataset_preproc_temp/"):
        os.mkdir("input/NERdataset_preproc_temp/")
    make_dir()
    preprocess()
    if os.path.isdir("input/NERdataset_preproc_temp/"):
        shutil.rmtree("input/NERdataset_preproc_temp/")
