import pandas as pd
import sys
from pathlib import Path

def preprocess():
    SOURCE = Path("input/NERdataset/NCBI-disease/")
    DEST = Path("input/NERdataset_preproc/NCBI-disease/")
    pathlist = Path(SOURCE).glob('*.tsv')
    for path in pathlist:
        PATH = str(SOURCE/path.name)
        NEWPATH = str(DEST/path.name)
        print(PATH, NEWPATH)
        do_preprocess(PATH,' -> ',NEWPATH)

def do_preprocess(PATH,NEWPATH):
    '''
    Preprocessing script for NCBI-disease data
    The original dataset is just a csv of words and tags
    we convert it into sentences and the tags of that sentence
    Output df: 
                Sentence              |       Tags
    ['Bert','is','a','good','model'] : ['B','O','O','O','O']
    '''
    df = pd.read_csv(PATH,delimiter='\t',names=['word','label'],skip_blank_lines=False)
    new_data = []
    sent, label_list = [],[]
    for r in df.itertuples(): 
        if r[1]!=r[1]: #Check for NaN which are present at newlines
            if(sent[-1]!='.'):
                sent.append('.')
                label_list.append('O')
            new_data.append([sent,label_list])
            # To check that sentence length and label_list length are consistent
            assert(len(sent)==len(label_list))
            if(label_list[0]=='I'):
                print(sent,label_list)
            # To check there are no Named entities overlapping with a sentence end
            assert(label_list[0]!='I')
            sent, label_list = [],[]
        else:
            sent.append(r[1])
            label_list.append(r[2])
        
    df_new = pd.DataFrame(new_data,columns = ['sentence','labels'])
    df_new.to_csv(NEWPATH)

if __name__ == "__main__":
    preprocess()