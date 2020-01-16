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
import model_rep
import utils
import random
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model

def get_scores(OUTPUT_DIR, data, resultsfile, reg_class_flag, logname):

    print('scoring', resultsfile)
    df = pd.read_pickle(resultsfile)

    print("successfully read file")
    x = df['true_mos']
    y = df['predict_mos']
    systemID = df['system_ID']
    speakerID = df['speaker_ID']
    print(x, y, systemID, speakerID)

'''   
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


    if args.reg_class_flag == "R":
        LCC=np.corrcoef(MOS_true, MOS_Predict)
        print('[UTTERANCE] Linear correlation coefficient= %f' % LCC[0][1])
        SRCC=scipy.stats.spearmanr(MOS_true.T, MOS_Predict.T)
        print('[UTTERANCE] Spearman rank correlation coefficient= %f' % SRCC[0])    
        MSE=np.mean((MOS_true-MOS_Predict)**2)
        print('[UTTERANCE] Test error= %f' % MSE)
    elif args.reg_class_flag == "C":
        ACC = accuracy_score(MOS_true, MOS_Predict)
        print('[UTTERANCE] Accuracy = %f' % ACC)
        print(confusion_matrix(MOS_true, MOS_Predict))

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



    if args.data == "VC":
        # load vcc2018_system
        sys_df = pd.read_csv(os.path.join(DATA_DIR,'vcc2018_system.csv'))
        df['system_ID'] = df['audio'].str.split('_').str[-1].str.split('.').str[0] + '_' + df['audio'].str.split('_').str[0]
    elif args.data == "LA":
        # load LA 2019 system
        sys_df = pd.read_csv(os.path.join(DATA_DIR,'LA_mos_system.csv'))
     
    sys_result_mean = df[['system_ID', 'predict_mos']].groupby(['system_ID']).mean()
    sys_mer_df = pd.merge(sys_result_mean, sys_df, on='system_ID')                          

    if args.reg_class_flag == "R":
        sys_true = sys_mer_df['mean']
        sys_predicted = sys_mer_df['predict_mos']
        print(sys_true)
        print(sys_predicted)
        print(sys_true.shape)
        print(sys_predicted.shape)
        LCC=np.corrcoef(sys_true, sys_predicted)
        print('[SYSTEM] Linear correlation coefficient= %f' % LCC[0][1])
        SRCC=scipy.stats.spearmanr(sys_true.T, sys_predicted.T)
        print('[SYSTEM] Spearman rank correlation coefficient= %f' % SRCC[0])
        MSE=np.mean((sys_true-sys_predicted)**2)
        print('[SYSTEM] Test error= %f' % MSE)
    elif args.reg_class_flag == "C":
        sys_true = sys_mer_df['mean'].round(0).astype(int)
        sys_predicted = sys_mer_df['predict_mos'].round(0).astype(int)

        print(sys_true)
        print(sys_predicted)
        print(sys_true.shape)
        print(sys_predicted.shape)
        ACC = accuracy_score(sys_true, sys_predicted)
        print('[SYSTEM] Accuracy = %f' % ACC)
        print(confusion_matrix(sys_true, sys_predicted))

    # Plotting scatter plot
    M=np.max([np.max(sys_predicted),5])
    # m=np.max([np.min(sys_predicted)-1,0.5])
    plt.figure(4)
    plt.scatter(sys_true, sys_predicted, s =25, color='b',  marker='o', edgecolors='b')
    plt.xlim([1,M])
    plt.ylim([1,M])
    plt.xlabel('True MOS')
    plt.ylabel('Predicted MOS')
    plt.title('System-Level')

    # # add system id
    # for i in range(len(sys_mer_df)):
    #     sys_ID = mer_df['system_ID'][i]
    #     x = mer_df['mean'][i]
    #     y = mer_df['predict_mos'][i]
    #     plt.text(x-0.05, y+0.1, sys_ID, fontsize=8)
    plt.savefig('./'+OUTPUT_DIR+'/MOSNet_system_scatter_plot.png', dpi=150)


    
    if args.data == "LA":
        spk_df = pd.read_csv(os.path.join(DATA_DIR,'LA_mos_speaker.csv'))
        spk_result_mean = df[['speaker_ID', 'predict_mos']].groupby(['speaker_ID']).mean()
        spk_mer_df = pd.merge(spk_result_mean, spk_df, on='speaker_ID')                          
        spk_result_mean = df[['speaker_ID', 'predict_mos']].groupby(['speaker_ID']).mean()
        spk_mer_df = pd.merge(spk_result_mean, spk_df, on='speaker_ID')                                                                                                                 
        if args.reg_class_flag == "R":
            spk_true = spk_mer_df['mean']
            spk_predicted = spk_mer_df['predict_mos']
            LCC=np.corrcoef(spk_true, spk_predicted)
            print('[SPEAKER] Linear correlation coefficient= %f' % LCC[0][1])
            SRCC=scipy.stats.spearmanr(spk_true.T, spk_predicted.T)
            print('[SPEAKER] Spearman rank correlation coefficient= %f' % SRCC[0])
            MSE=np.mean((spk_true-spk_predicted)**2)
            print('[SPEAKER] Test error= %f' % MSE)
        elif args.reg_class_flag == "C":
            spk_true = spk_mer_df['mean'].round(0).astype(int)
            spk_predicted = spk_mer_df['predict_mos'].round(0).astype(int)
            print(spk_true.shape)
            print(spk_predicted.shape)
            ACC = accuracy_score(spk_true, spk_predicted)
            print('[SPEAKER] Accuracy = %f' % ACC)
            print(confusion_matrix(spk_true, spk_predicted))
           
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
'''

    
# move my orig folders somewhere else
F = glob.glob("./output*/")
print(F)
# output_CNN_16_LA_xvec_5_R_0.01_0.1_64_16
# output, nn, batch, data, feats, reg/class, l2, dr, nodes, batch
for folder in F:
    logname = "log."+folder
    results_file = folder+"/res_df.pkl"
    if os.path.isfile(results_file):
        print("Success")
    items = folder.split("_")
    data = items[3]
    if data == "LA":
        try:
            print("try getting scores")
            feats = items[4]
            if feats == "xvec":
                feats = feats + "_" + items[5]
            model = folder+"/mosnet.h5"
            bin_dir = "data_"+data+"/"+feats
            flag = items[6]
            # get the model name, pass to the test function
            get_scores(folder, data, results_file, flag, logname)
            sys.exit()
        except:
            continue