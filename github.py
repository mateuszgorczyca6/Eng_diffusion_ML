import pandas as pd
from math import ceil
from os import remove

to_divide = {'data/part1/ML/features.csv': 2}

def split_pandas(path, num):
    df = pd.read_csv(path)
    print(len(df))
    del df['Unnamed: 0']
    size = len(df)
    new_size = ceil(size / num)
    done = 0
    n = 0
    fname = '.'.join(path.split('.')[:-1])
    while done < size:
        ffname = fname+str(n)+'.csv'
        new_df = df.loc[done : done+new_size]
        print(ffname)
        new_df.to_csv(ffname)
        n += 1
        done += new_size + 1
    remove(path)

def join_pandas(path, num):
    fname = '.'.join(path.split('.')[:-1])
    ffname = fname+str(0)+'.csv'
    print(ffname)
    df = pd.read_csv(ffname)
    remove(ffname)
    for n in range(1, num):
        ffname = fname+str(n)+'.csv'
        print(ffname)
        df = pd.concat([df, pd.read_csv(ffname)], ignore_index=True)
        remove(ffname)
    del df['Unnamed: 0']
    df.to_csv(path)
    print(len(df))
    print(df.head())
    print(df.loc[9990:10010])
    print(df.tail())

def split():
    for key,value in to_divide.items():
        split_pandas(key,value)

def join():
    for key,value in to_divide.items():
        join_pandas(key,value)