import glob, sys

import os.path
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
#import model_rep
import model
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

        filepath=test_list[i].split(',')
        speakerid = filepath[0]
        sysid = filepath[1]
        filename=filepath[2].split('.')[0]
        mos=float(filepath[3])

        _feat = utils.read(os.path.join(BIN_DIR,filename+'.h5'))
        _mag = _feat['mag_sgram']    
        [Average_score, Frame_score]=model.predict(_mag, verbose=0, batch_size=1)
        MOS_Predict[i]=Average_score
        MOS_true[i]   =mos


#        _DS = utils.read_rep(os.path.join(BIN_DIR,filename+'.npy'))
#        _DS = np.expand_dims(_DS, axis=3)
#        Average_score=model.predict(_DS, verbose=0, batch_size=1)
#        MOS_Predict[i]=Average_score[0][0]
#        MOS_true[i] =mos
            
        df = df.append({'audio': filename, 
                        'true_mos': MOS_true[i], 
                        'predict_mos': MOS_Predict[i], 
                        'system_ID': sysid, 
                        'speaker_ID': speakerid}, 
                       ignore_index=True)
        
    df.to_pickle(results_file)
    return



def get_scores(OUTPUT_DIR, data, resultsfile, reg_class_flag, logname):


    out = open(logname, "w")
    print('scoring', resultsfile)
    df = pd.read_pickle(resultsfile)

    print("successfully read file")
    x = df['true_mos']
    y = df['predict_mos']
    systemID = df['system_ID']
    speakerID = df['speaker_ID']

    MOS_true = np.array(x)
    MOS_Predict = np.array(y)

    plt.style.use('seaborn-deep')
    x = df['true_mos']
    y = df['predict_mos']
    bins = np.linspace(1, 5, 40)
    plt.figure(2)
    plt.hist([x, y], bins, label=['true_mos', 'predict_mos'])
    plt.legend(loc='upper right')
    plt.xlabel('MOS')
    plt.ylabel('number') 
    plt.savefig('./'+OUTPUT_DIR+'/MOSNet_distribution.png', dpi=150)


    LCC=np.corrcoef(MOS_true, MOS_Predict)
    out.write('[UTTERANCE] Linear correlation coefficient= %f' % LCC[0][1]+"\n")
    SRCC=scipy.stats.spearmanr(MOS_true.T, MOS_Predict.T)
    out.write('[UTTERANCE] Spearman rank correlation coefficient= %f' % SRCC[0]+"\n")    
    MSE=np.mean((MOS_true-MOS_Predict)**2)
    out.write('[UTTERANCE] Test error= %f' % MSE+"\n")


    # Plotting scatter plot
    M=np.max([np.max(MOS_Predict),5])
    plt.figure(3)
    plt.scatter(MOS_true, MOS_Predict, s =15, color='b',  marker='o', edgecolors='b', alpha=.20)
    plt.xlim([0.5,M])
    plt.ylim([0.5,M])
    plt.xlabel('True MOS')
    plt.ylabel('Predicted MOS')
    plt.title('Utterance-Level')
    plt.savefig('./'+OUTPUT_DIR+'/MOSNet_scatter_plot.png', dpi=150)


    spk_result_mean = df[['speaker_ID', 'predict_mos', 'true_mos']].groupby(['speaker_ID']).mean()

    spk_true = spk_result_mean['true_mos']
    spk_predicted = spk_result_mean['predict_mos']
    print(spk_true)

    spk_true_mean = np.mean(spk_true)
    spk_predicted_mean = np.mean(spk_predicted)
    out.write('\t[SPEAKER-AGG] TRUE= %f' % (spk_true_mean)+"\n")
    out.write('\t[SPEAKER-AGG] Predicted= %f' % (spk_predicted_mean)+"\n")

    LCC=np.corrcoef(spk_true, spk_predicted)
    out.write('[SPEAKER-AGG] Linear correlation coefficient= %f' % LCC[0][1]+"\n")
    SRCC=scipy.stats.spearmanr(spk_true.T, spk_predicted.T)
    out.write('[SPEAKER-AGG] Spearman rank correlation coefficient= %f' % SRCC[0]+"\n")
    MSE=np.mean((spk_true-spk_predicted)**2)
    MAE=np.mean(np.absolute(spk_true-spk_predicted))
    out.write('[SPEAKER-AGG] MSE error= %f' % MSE+"\n")
    out.write('[SPEAKER-AGG] MAE error= %f' % (MAE)+"\n")
        
    # Plotting scatter plot
    M=np.max([np.max(spk_predicted),5])
    # m=np.max([np.min(spk_predicted)-1,0.5])
    plt.figure(4)
    plt.scatter(spk_true, spk_predicted, s =25, color='b',  marker='o', edgecolors='b')
    plt.xlim([1,M])
    plt.ylim([1,M])
    plt.xlabel('True MOS')
    plt.ylabel('Predicted MOS')
    plt.title('Speaker-Level')
    
    # # add system id
    # for i in range(len(spk_mer_df)):
    #     spk_ID = mer_df['speaker_ID'][i]
    #     x = mer_df['mean'][i]
    #     y = mer_df['predict_mos'][i]
    #     plt.text(x-0.05, y+0.1, spk_ID, fontsize=8)
    plt.savefig('./'+OUTPUT_DIR+'/MOSNet_speaker_scatter_plot.png', dpi=150)


    spk_resultP = df[['speaker_ID', 'predict_mos']].groupby(['speaker_ID'])['predict_mos']
    spk_resultT = df[['speaker_ID', 'true_mos']].groupby(['speaker_ID'])['true_mos']
    print(spk_resultP)
    out.write("SYSTEM-AGG-SPEAKERS")
    for speakerID,true in spk_resultT:
        spk_true = spk_resultT.get_group(speakerID)
        spk_predicted = spk_resultP.get_group(speakerID)
        spk_true_mean = np.mean(spk_true)
        spk_predicted_mean = np.mean(spk_predicted)
        abs_diff = np.absolute(spk_true_mean - spk_predicted_mean)
        out.write('\t[SPEAKER-%s] TRUE= %f' % (speakerID,spk_true_mean)+"\n")
        out.write('\t[SPEAKER-%s] Predicted= %f' % (speakerID,spk_predicted_mean)+"\n")
#        out.write('\t[SPEAKER-%s] Abs Diff= %f' % (speakerID,abs_diff)+"\n")
#        LCC=np.corrcoef(spk_true, spk_predicted)
#        out.write('\t[SPEAKER-%s] Linear correlation coefficient= %f' % (speakerID,LCC[0][1])+"\n")
#        SRCC=scipy.stats.spearmanr(spk_true.T, spk_predicted.T)
#        out.write('\t[SPEAKER-%s] Spearman rank correlation coefficient= %f' % (speakerID,SRCC[0])+"\n")
#        MSE=np.mean((spk_true-spk_predicted)**2)
#        out.write('\t[SPEAKER-%s] MSE error= %f' % (speakerID,MSE)+"\n")
#        MAE=np.mean(np.absolute(spk_true-spk_predicted))
#        out.write('\t[SPEAKER-%s] MAE error= %f' % (speakerID,MAE)+"\n")

           

folder = './results_R/output_CNN_1_LA_orig/'
logname = "log."+folder[12:-1]
results_file = folder+"/harvard100.pkl"
items = folder.split("/")[2].split("_")
data = "LA"
flag = "R"
testfile = "data_harvard100/test_list.txt"
feats = "orig"
input = open(testfile, "r")
testlist = input.read().split("\n")[:-1]
input.close()
print(items)
model = folder+"/mosnet.h5"
bin_dir = "data_harvard100/"+feats
# get the model name, pass to the test function
#get_test_results(bin_dir, data, testlist, model, results_file, flag)
get_scores(folder, data, results_file, flag, logname)

