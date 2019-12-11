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
import keras
import model_keras
import utils
import random

import random
myseed = 1984
random.seed(myseed) 
np.random.seed(myseed)

import platform, sys
print(platform.python_version())
print(tf.__version__)


parser = argparse.ArgumentParser()
parser.add_argument("--model", help="model to train with, CNN, BLSTM or CNN-BLSTM")
parser.add_argument("--epoch", type=int, default=100, help="number epochs")
parser.add_argument("--batch_size", type=int, default=8, help="number batch_size")
parser.add_argument("--data", help="data: VC, LA")
parser.add_argument("--feats", help="feats: orig, deepspectrum")


args = parser.parse_args()

if not args.model:
    raise ValueError('please specify model to train with, CNN, BLSTM or CNN-BLSTM')


print('training with model architecture: {}'.format(args.model))   
print('epochs: {}\nbatch_size: {}'.format(args.epoch, args.batch_size))
print('training with data: {}'.format(args.data))   
print('training with features: {}'.format(args.feats))   

# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # or any {'0', '1', '2'}


# set dir
DATA_DIR = './data_'+args.data
BIN_DIR = os.path.join(DATA_DIR, args.feats)
OUTPUT_DIR = './output_'+args.model+"_"+args.batch_size+"_"+args.data+"_"+args.feats

EPOCHS = args.epoch
BATCH_SIZE = args.batch_size

NUM_TRAIN = 13580
NUM_TEST=4000
NUM_VALID=3000

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
            
mos_list = utils.read_list(os.path.join(DATA_DIR,'mos_list.txt'))
random.shuffle(mos_list)

train_list= mos_list[0:-(NUM_TEST+NUM_VALID)]
random.shuffle(train_list)
valid_list= mos_list[-(NUM_TEST+NUM_VALID):-NUM_TEST]
test_list= mos_list[-NUM_TEST:]

print('{} for training; {} for valid; {} for testing'.format(NUM_TRAIN, NUM_TEST, NUM_VALID))        
# init model
if args.model == 'CNN':
    MOSNet = model_keras.CNN()
elif args.model == 'BLSTM':
    MOSNet = model_keras.BLSTM()
elif args.model == 'CNN-BLSTM':
    MOSNet = model_keras.CNN_BLSTM()
else:
    raise ValueError('please specify model to train with, CNN, BLSTM or CNN-BLSTM')

model = MOSNet.build()

model.compile(
    optimizer=keras.optimizers.Adam(1e-4),
    loss={'avg':'mse',
          'frame':'mse'},)
    
CALLBACKS = [
    keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(OUTPUT_DIR,'mosnet.h5'),
        save_best_only=True,
        monitor='val_loss',
        verbose=1),
#    keras.callbacks.TensorBoard(
#        log_dir=os.path.join(OUTPUT_DIR,'tensorboard.log'),
#        update_freq='epoch'), 
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        mode='min',
        min_delta=0,
        patience=5,
        verbose=1)
]

if args.feats == 'orig':
    # data generator
    train_data = utils.data_generator(train_list, BIN_DIR, frame=True, batch_size=BATCH_SIZE)
    valid_data = utils.data_generator(valid_list, BIN_DIR, frame=True, batch_size=BATCH_SIZE)
if args.feats == 'deepspectrum':
    train_data = utils.data_generator_DS(train_list, BIN_DIR, batch_size=BATCH_SIZE)
    valid_data = utils.data_generator_DS(valid_list, BIN_DIR, batch_size=BATCH_SIZE)




tr_steps = int(NUM_TRAIN/BATCH_SIZE)
val_steps = int(NUM_VALID/BATCH_SIZE)



# start fitting model
hist = model.fit_generator(train_data,
                           steps_per_epoch=tr_steps,
                           epochs=EPOCHS,
                           callbacks=CALLBACKS,
                           validation_data=valid_data,
                           validation_steps=val_steps,
                           verbose=1,)
    

# plot testing result
model.load_weights(os.path.join(OUTPUT_DIR,'mosnet.h5'),)   # Load the best model   

print('testing...')
MOS_Predict=np.zeros([len(test_list),])
MOS_true   =np.zeros([len(test_list),])
df = pd.DataFrame(columns=['audio', 'true_mos', 'predict_mos'])

for i in tqdm(range(len(test_list))):
    
    filepath=test_list[i].split(',')
    filename=filepath[0].split('.')[0]
    
    _feat = utils.read(os.path.join(BIN_DIR,filename+'.h5'))
    _mag = _feat['mag_sgram']    
        
    mos=float(filepath[1])
    
    [Average_score, Frame_score]=model.predict(_mag, verbose=0, batch_size=1)
    MOS_Predict[i]=Average_score
    MOS_true[i]   =mos
    df = df.append({'audio': filepath[0], 
                    'true_mos': MOS_true[i], 
                    'predict_mos': MOS_Predict[i]}, 
                   ignore_index=True)
    
    

plt.style.use('seaborn-deep')
x = df['true_mos']
y = df['predict_mos']
bins = np.linspace(1, 5, 40)
plt.figure(2)
plt.hist([x, y], bins, label=['true_mos', 'predict_mos'])
plt.legend(loc='upper right')
plt.xlabel('MOS')
plt.ylabel('number') 
plt.show()
plt.savefig('./output/MOSNet_distribution.png', dpi=150)

MSE=np.mean((MOS_true-MOS_Predict)**2)
print('[UTTERANCE] Test error= %f' % MSE)
LCC=np.corrcoef(MOS_true, MOS_Predict)
print('[UTTERANCE] Linear correlation coefficient= %f' % LCC[0][1])
SRCC=scipy.stats.spearmanr(MOS_true.T, MOS_Predict.T)
print('[UTTERANCE] Spearman rank correlation coefficient= %f' % SRCC[0])    
    


# Plotting scatter plot
M=np.max([np.max(MOS_Predict),5])
plt.figure(3)
plt.scatter(MOS_true, MOS_Predict, s =15, color='b',  marker='o', edgecolors='b', alpha=.20)
plt.xlim([0.5,M])
plt.ylim([0.5,M])
plt.xlabel('True MOS')
plt.ylabel('Predicted MOS')
plt.title('LCC= {:.4f}, SRCC= {:.4f}, MSE= {:.4f}'.format(LCC[0][1], SRCC[0], MSE))
plt.show()
plt.savefig('./output/MOSNet_scatter_plot.png', dpi=150)


# load vcc2018_system
sys_df = pd.read_csv(os.path.join(DATA_DIR,'vcc2018_system.csv'))
df['system_ID'] = df['audio'].str.split('_').str[-1].str.split('.').str[0] + '_' + df['audio'].str.split('_').str[0]
result_mean = df[['system_ID', 'predict_mos']].groupby(['system_ID']).mean()
mer_df = pd.merge(result_mean, sys_df, on='system_ID')                                                                                                                 

sys_true = mer_df['mean']
sys_predicted = mer_df['predict_mos']

MSE=np.mean((sys_true-sys_predicted)**2)
print('[SYSTEM] Test error= %f' % MSE)
LCC=np.corrcoef(sys_true, sys_predicted)
print('[SYSTEM] Linear correlation coefficient= %f' % LCC[0][1])
SRCC=scipy.stats.spearmanr(sys_true.T, sys_predicted.T)
print('[SYSTEM] Spearman rank correlation coefficient= %f' % SRCC[0])

# Plotting scatter plot
M=np.max([np.max(sys_predicted),5])
# m=np.max([np.min(sys_predicted)-1,0.5])
plt.figure(4)
plt.scatter(sys_true, sys_predicted, s =25, color='b',  marker='o', edgecolors='b')
plt.xlim([1,M])
plt.ylim([1,M])
plt.xlabel('True MOS')
plt.ylabel('Predicted MOS')
plt.title('LCC= {:.4f}, SRCC= {:.4f}, MSE= {:.4f}'.format(LCC[0][1], SRCC[0], MSE))

# # add system id
# for i in range(len(mer_df)):
#     sys_ID = mer_df['system_ID'][i]
#     x = mer_df['mean'][i]
#     y = mer_df['predict_mos'][i]
#     plt.text(x-0.05, y+0.1, sys_ID, fontsize=8)
plt.show()
plt.savefig('./output/MOSNet_system_scatter_plot.png', dpi=150)
