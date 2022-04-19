import os
import pandas as pd
from sklearn import preprocessing
from sklearn import metrics 
import joblib

from . import dispatcher

# we will be taking the TRAINING_DATA and FOLD from environment variables (31:13) - defined in file run.sh
TRAINING_DATA = os.environ.get("TRAINING_DATA")    # "input/train_folds.csv"
FOLD = int(os.environ.get("FOLD"))
MODEL = os.environ.get("MODEL") 

# TRAINING_DATA = "input/train_folds.csv"
# FOLD = 0

# Fold mapping creating such that for fold values between 0 to 4 if any number is taken as key the other values will
# be taken up as values

FOLD_MAPPING = {
    0: [1, 2, 3, 4],
    1: [0, 2, 3, 4],
    2: [0, 1, 3, 4],
    3: [0, 1, 2, 4],
    4: [0, 1, 2, 3]
}

if __name__=="__main__":
    # read training data
    df = pd.read_csv(TRAINING_DATA)
    train_df = df[df.kfold.isin(FOLD_MAPPING.get(FOLD))]
    valid_df = df[df.kfold == FOLD]

    ytrain = train_df.target.values
    yvalid = valid_df.target.values

    train_df = train_df.drop(columns = ['id', 'target', 'kfold'])
    valid_df = valid_df.drop(columns = ['id', 'target', 'kfold'])
    
    # to make sure order of variables in train and validating data are same
    valid_df = valid_df[train_df.columns]

    label_encoders = []
    for c in train_df.columns:
        lbl = preprocessing.LabelEncoder()
        lbl.fit(train_df[c].values.tolist() + valid_df[c].values.tolist())
        train_df.loc[:,c] = lbl.transform(train_df[c].values.tolist())
        valid_df.loc[:,c] = lbl.transform(valid_df[c].values.tolist())
        label_encoders.append((c,lbl))


    # data is ready to train
    clf = dispatcher.MODELS[MODEL]
    clf.fit(train_df, ytrain)
    preds = clf.predict_proba(valid_df)[:,1]
    print(metrics.roc_auc_score(yvalid, preds))
    
    joblib.dump(value = label_encoders, filename = f"models/{MODEL}_label_encoder.pkl")
    joblib.dump(value = clf, filename = f"models/{MODEL}.pkl")

    









