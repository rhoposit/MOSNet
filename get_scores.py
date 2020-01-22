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
#import model
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

#        _feat = utils.read(os.path.join(BIN_DIR,filename+'.h5'))
#        _mag = _feat['mag_sgram']    
#        [Average_score, Frame_score]=model.predict(_mag, verbose=0, batch_size=1)
#        MOS_Predict[i]=Average_score
#        MOS_true[i]   =mos


        _DS = utils.read_rep(os.path.join(BIN_DIR,filename+'.npy'))
        _DS = np.expand_dims(_DS, axis=3)
        Average_score=model.predict(_DS, verbose=0, batch_size=1)
        MOS_Predict[i]=Average_score[0][0]
        MOS_true[i] =mos
            
        df = df.append({'audio': filepath[0], 
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


    if data == "VC":
        # load vcc2018_system
        sys_df = pd.read_csv(os.path.join("data_VC",'vcc2018_system.csv'))
        df['system_ID'] = df['audio'].str.split('_').str[-1].str.split('.').str[0] + '_' + df['audio'].str.split('_').str[0]
    elif data == "LA":
        # load LA 2019 system
        sys_df = pd.read_csv(os.path.join("data_LA",'LA_mos_system.csv'))
     
    sys_result_mean = df[['system_ID', 'predict_mos']].groupby(['system_ID']).mean()
    sys_mer_df = pd.merge(sys_result_mean, sys_df, on='system_ID')                          

    sys_true = sys_mer_df['mean']
    sys_predicted = sys_mer_df['predict_mos']
    LCC=np.corrcoef(sys_true, sys_predicted)
    out.write('[SYSTEM-AGG] Linear correlation coefficient= %f' % LCC[0][1]+"\n")
    SRCC=scipy.stats.spearmanr(sys_true.T, sys_predicted.T)
    out.write('[SYSTEM-AGG] Spearman rank correlation coefficient= %f' % SRCC[0]+"\n")
    MSE=np.mean((sys_true-sys_predicted)**2)
    out.write('[SYSTEM-AGG] MSE error= %f' % MSE+"\n")
    MAE=np.mean(np.absolute(sys_true-sys_predicted))
    out.write('[SYSTEM-AGG] MAE error= %f' % (MAE)+"\n")


    sys_resultP = df[['system_ID', 'predict_mos']].groupby(['system_ID'])
    sys_resultT = df[['system_ID', 'true_mos']].groupby(['system_ID'])

    for systemID,true in sys_resultT:
        sys_true = sys_resultT.get_group(systemID)['true_mos']
        sys_predicted = sys_resultP.get_group(systemID)['predict_mos']
        sys_true_mean = np.mean(sys_true)
        sys_predicted_mean = np.mean(sys_predicted)
        abs_diff = np.absolute(sys_true_mean - sys_predicted_mean)
        out.write('\t[SYSTEM-%s] Mean True= %f' % (systemID,sys_true_mean)+"\n")
        out.write('\t[SYSTEM-%s] Mean Predicted= %f' % (systemID,sys_predicted_mean)+"\n")
        out.write('\t[SYSTEM-%s] Abs Diff= %f' % (systemID, abs_diff)+"\n")
        LCC=np.corrcoef(sys_true, sys_predicted)
        out.write('\t[SYSTEM-%s] Linear correlation coefficient= %f' % (systemID,LCC[0][1])+"\n")
        SRCC=scipy.stats.spearmanr(sys_true.T, sys_predicted.T)
        out.write('\t[SYSTEM-%s] Spearman rank correlation coefficient= %f' % (systemID,SRCC[0])+"\n")
        MSE=np.mean((sys_true-sys_predicted)**2)
        out.write('\t[SYSTEM-%s] MSE error= %f' % (systemID,MSE)+"\n")
        MAE=np.mean(np.absolute(sys_true-sys_predicted))
        out.write('\t[SYSTEM-%s] MAE error= %f' % (systemID,MAE)+"\n")


        
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


    
    spk_df = pd.read_csv(os.path.join("data_LA",'LA_mos_speaker.csv'))
    spk_result_mean = df[['speaker_ID', 'predict_mos']].groupby(['speaker_ID']).mean()
    spk_mer_df = pd.merge(spk_result_mean, spk_df, on='speaker_ID')
    spk_result_mean = df[['speaker_ID', 'predict_mos']].groupby(['speaker_ID']).mean()
    spk_mer_df = pd.merge(spk_result_mean, spk_df, on='speaker_ID')

    spk_true = spk_mer_df['mean']
    spk_predicted = spk_mer_df['predict_mos']
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
    for speakerID,true in spk_resultT:
        spk_true = spk_resultT.get_group(speakerID)
        spk_predicted = spk_resultP.get_group(speakerID)
        spk_true_mean = np.mean(spk_true)
        spk_predicted_mean = np.mean(spk_predicted)
        abs_diff = np.absolute(spk_true_mean - spk_predicted_mean)
        out.write('\t[SPEAKER-%s] Mean True= %f' % (speakerID,spk_true_mean)+"\n")
        out.write('\t[SPEAKER-%s] Mean Predicted= %f' % (speakerID,spk_predicted_mean)+"\n")
        out.write('\t[SPEAKER-%s] Abs Diff= %f' % (speakerID,abs_diff)+"\n")
        LCC=np.corrcoef(spk_true, spk_predicted)
        out.write('\t[SPEAKER-%s] Linear correlation coefficient= %f' % (speakerID,LCC[0][1])+"\n")
        SRCC=scipy.stats.spearmanr(spk_true.T, spk_predicted.T)
        out.write('\t[SPEAKER-%s] Spearman rank correlation coefficient= %f' % (speakerID,SRCC[0])+"\n")
        MSE=np.mean((spk_true-spk_predicted)**2)
        out.write('\t[SPEAKER-%s] MSE error= %f' % (speakerID,MSE)+"\n")
        MAE=np.mean(np.absolute(spk_true-spk_predicted))
        out.write('\t[SPEAKER-%s] MAE error= %f' % (speakerID,MAE)+"\n")


           
''' 
folder = 'results_R/output_CNN_64_LA_CNN_R_0.01_0.1_32_64'
data = "LA"
results_file = folder+"/res_df.pkl"
logname = "log."+folder[10:-1]
flag = "R"
testfile = "data_LA/test_list.txt"
input = open(testfile, "r")
testlist = input.read().split("\n")[:-1]
input.close()
model = folder+"/mosnet.h5"
bin_dir = "data_LA/CNN"
get_test_results(bin_dir, data, testlist, model, results_file, flag)
get_scores(folder, data, results_file, flag, logname)
sys.exit()
'''

# move my orig folders somewhere else
#F = glob.glob("./results_R/output*LA_orig/")
F = glob.glob("./results_O/output*/")
print(F)
# output_CNN_16_LA_xvec_5_R_0.01_0.1_64_16
# output, nn, batch, data, feats, reg/class, l2, dr, nodes, batch
for folder in F:
#    try:
        logname = "log."+folder[12:-1]
        results_file = folder+"/res_df.pkl"
        items = folder.split("/")[2].split("_")
        if len(items) < 11:
            continue
        data = items[3]
        flag = items[5]
#        data = "LA"
#        flag = "R"
        if data == "LA" and flag == "R":
            testfile = "data_LA/test_list.txt"
#            feats = "orig"
            feats = items[4]
#            if feats == "xvec":
#                feats = feats + "_" + items[5]
            input = open(testfile, "r")
            testlist = input.read().split("\n")[:-1]
            input.close()
            print(items)
            model = folder+"/mosnet.h5"
            bin_dir = "data_"+data+"/"+feats
            # get the model name, pass to the test function
#            get_test_results(bin_dir, data, testlist, model, results_file, flag)
            get_scores(folder, data, results_file, flag, logname)
#    except:
#        print("skipping: ", folder)
#        continue

