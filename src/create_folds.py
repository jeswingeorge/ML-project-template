import pandas as pd
from sklearn import model_selection

if __name__ == "__main__":
    df = pd.read_csv("input/train.csv")
    # create a fake column kfold and assign value -1
    df['kfold'] = -1

    # shuffle the data and reset indices
    df = df.sample(frac = 1).reset_index(drop=True)

    # need the kfolds
    kf = model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X = df, y = df.target.values)):
        # kf.split(X = df, y = df.target.values) - returns 2 list of indices for train and validation sets
        print(len(train_idx), len(val_idx))
        df.loc[val_idx, 'kfold'] = fold

    
    df.to_csv("input/train_folds.csv", index=False)

    
# In spyder run this script in the console
# runfile('D:\\Github\\ML project template\\src\\create_folds.py')
# or you can use the anaconda prompt.
