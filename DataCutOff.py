import os
import pandas as pd
from sklearn.utils import shuffle


if __name__ == '__main__':
    path = "data/"
    pd_all = pd.read_csv(os.path.join(path, "data_clean.csv"), header=None, sep='\t' )
    pd_all = shuffle(pd_all)    # 随机打乱
    pd_all.to_csv(path+"data_clean_shuffled.csv", encoding='utf-8', sep='\t', index=None, header=None)
    # pd_all = pd.read_csv(os.path.join(path, "data_clean_shuffled.csv"), header=None, sep='\t' )


    dev_set = pd_all.iloc[0:int(pd_all.shape[0]*0.2)]
    train_set = pd_all.iloc[int(pd_all.shape[0]*0.2): int(pd_all.shape[0])]
    dev_set.to_csv("data/val.csv", index=False, sep='\t', header=None)
    train_set.to_csv("data/train.csv", index=False, sep='\t', header=None)
    
    