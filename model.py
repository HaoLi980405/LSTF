import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.linear_model import LinearRegression

dataset = 'Exchange'
raw = pd.read_csv(dataset+'.csv')
result_mae, result_mse = {}, {}

model_type = "LinearRegression"

if dataset == 'ILI':
    input_size = 96
elif dataset == 'Exchange':
    input_size = 31
else:
    input_size = 672

pred_lens = {'ETTh1': [24,48,168,336,720],
            'ETTh2': [24,48,168,336,720],
            'ETTm1': [24,48,96,288,672],
            'ETTm2': [96,192,336,720],
            #'ETTm2': [24,48,96,288,672],
            'ECL': [48,168,336,720,960],
            'WTH': [24,48,168,336,720],
            'Weather': [96,192,336,720],
            'ILI': [24,36,48,60],
            'Exchange': [96,192,336,720]}


for pred_len in tqdm(pred_lens[dataset]):
    mses = []
    maes = []
    for target in tqdm(raw.columns[1:]):
        data = raw[[target]].values.reshape(-1)
            
        if dataset == 'ETTh1' or dataset == 'ETTh2':
            border1s = [0, 12*30*24 - input_size, 16*30*24-input_size]
            border2s = [12*30*24, 16*30*24, 20*30*24]
        elif dataset == 'ETTm1' or dataset == 'ETTm2':
            border1s = [0, 12*30*24*4 - input_size, 16*30*24*4-input_size]
            border2s = [12*30*24*4, 16*30*24*4, 20*30*24*4]
        else:
            border1s = [0, int(len(data)*0.7)-input_size, int(len(data)*0.8)-input_size]
            border2s = [int(len(data)*0.7), int(len(data)*0.8), len(data)]
            
        training_series = data[border1s[0]:border2s[0]]
        mean = training_series.mean()
        std = training_series.std()
        training_series = data[border1s[0]:border2s[1]]
        training_series = (training_series-mean)/std
        test_series = data[border1s[2]:border2s[2]]
        test_series = (test_series-mean)/std
        
        train_X = []
        train_Y = []
        for i in range(border2s[1]-pred_len-input_size+1):
            X = training_series[i:(i+input_size)]
            Y = training_series[(i+input_size):(i+input_size+pred_len)]
            train_X.append(X)
            train_Y.append(Y)
        
        train_X = np.array(train_X);train_Y = np.array(train_Y);
            
        test_X = []
        test_Y = []
        for i in range(0, border2s[2]-pred_len-input_size - border1s[2]+1):
            X = test_series[i:(i+input_size)]
            Y = test_series[(i+input_size):(i+input_size+pred_len)]
            test_X.append(X)
            test_Y.append(Y)
        test_X = np.array(test_X);test_Y = np.array(test_Y);
            
        model = LinearRegression()
                
        model.fit(train_X,train_Y)
        pred = model.predict(test_X)
            
        mse = np.mean((test_Y-pred)**2)
        mae = np.mean(abs(test_Y-pred))
        mses.append(mse)
        maes.append(mae)
    result_mae[pred_len] = np.mean(np.array(maes));result_mse[pred_len] = np.mean(np.array(mses));
    
#print results
print('\n')
print("{:<9} {:<6} {:<6}".format('pred_len','MAE','MSE'))
for pred_len in pred_lens[dataset]:
    print("{:<9} {:<6} {:<6}".format(pred_len,'%.3f'%result_mae[pred_len],'%.3f'%result_mse[pred_len]))
    