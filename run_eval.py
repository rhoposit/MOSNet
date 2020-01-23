import glob, sys

import os
import time 
import numpy as np
from tqdm import tqdm
import scipy.stats
import pandas as pd
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import argparse
import tensorflow as tf
from tensorflow import keras
import model_rep
import utils
import random
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model

def get_test_results(BIN_DIR, data, test_list, modelfile, resultsfile, reg_class_flag):

    print('testing...', modelfile)
    model = load_model(modelfile)
    model.summary()
    MOS_Predict=np.zeros([len(test_list),])
    MOS_true   =np.zeros([len(test_list),])
    df = pd.DataFrame(columns=['audio', 'true_mos','predict_mos','system_ID','speaker_ID'])

    for i in tqdm(range(len(test_list))):

        if data == "VC":
            filepath=test_list[i].split(',')
            filename=filepath[0].split('.')[0]
            sysid = ""
            speakerid = ""
            mos=float(filepath[1])
        elif data == "LA":
            filepath=test_list[i].split(',')
            filename=filepath[2].split('.')[0]
            sysid = filepath[1]
            speakerid = filepath[0]
            mos=float(filepath[3])

        _DS = utils.read_rep(os.path.join(BIN_DIR,filename+'.npy'))
        
        _DS = np.expand_dims(_DS, axis=3)
        Average_score=model.predict(_DS, verbose=0, batch_size=1)

        if reg_class_flag == "R":
            MOS_Predict[i]=Average_score
            MOS_true[i] =mos
        elif reg_class_flag == "C":
            MOS_Predict[i]=np.argmax(Average_score[0])
            MOS_true[i] = mos
        

        
        df = df.append({'audio': filepath[0], 
                        'true_mos': MOS_true[i], 
                        'predict_mos': MOS_Predict[i], 
                        'system_ID': sysid, 
                        'speaker_ID': speakerid}, 
                       ignore_index=True)
        
    df.to_pickle(results_file)
    return


    
# move my orig folders somewhere else
F = glob.glob("./output*/")
print(F)
# output_CNN_16_LA_xvec_5_R_0.01_0.1_64_16
# output, nn, batch, data, feats, reg/class, l2, dr, nodes, batch
for folder in F:
    logname = "log."+folder
    results_file = folder+"/res_df.pkl"
    items = folder.split("_")
    data = items[3]
    if data == "LA":
        try:
            testfile = "data_LA/test_list.txt"
            feats = items[4]
            if feats == "xvec":
                feats = feats + "_" + items[5]
            input = open(testfile, "r")
            testlist = input.read().split("\n")[:-1]
            input.close()
            model = folder+"/mosnet.h5"
            bin_dir = "data_"+data+"/"+feats
            flag = items[6]
            # get the model name, pass to the test function
            get_test_results(bin_dir, data, testlist, model, results_file, flag)
            # aggregate_test_results(results_file, logname)
        except:
            print("Caught exception, continuing")
