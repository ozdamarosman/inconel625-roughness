"""RQ1 mechanism: can a classifier tell which cut a window comes from?"""
import json
import os
import sys
import warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")
from common import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, KFold

WS = cols(SIG_ACC + SIG_DYN, win)
acc = cross_val_score(RandomForestClassifier(300, random_state=0, n_jobs=-1), win[WS].values, win.cut_id.values,
                      cv=KFold(5, shuffle=True, random_state=0)).mean()
print("cut-identity accuracy", acc, "chance", 1 / 27)
json.dump({"cut_id_accuracy": acc, "chance": 1 / 27}, open("as1_fingerprint.json", "w"))
