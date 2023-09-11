import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
import random as rand
from skopt import gp_minimize
from scipy import stats
import pandas as pd

np.random.seed(0)
random.seed(0)


##

def give_behave(coh, acc, rt):
    mean_acc = np.zeros(len(coherence))
    mean_rt = np.zeros(len(coherence))
    for i in range(len(coherence)):
        mean_acc[i] = np.mean(acc[np.where(coh == coherence[i])])
        mean_rt[i] = np.mean(rt[np.where(coh == coherence[i])])

    return mean_acc, mean_rt

def give_model_behave(params):
    A = params[0]
    K = params[1]
    ndt = params[2]

    P_An = 1 / (1 + np.exp(-2 * K * cxbin * A))
    RT_An = (A / (K * cxbin)) * np.tanh(K * cxbin * A) +ndt

    return P_An, RT_An


##
coherence = np.array([1.6, 3.2, 6.4, 12.8, 25.6])/100
cxbin = np.linspace(0, .3, 100)



##




# df = pd.read_csv(r'data.csv')
# NumSub = df['SubjectNumber'].unique()


df = pd.read_csv(r'./data.csv')
NumSub = df['SubjectNumber'].unique()
all_tr = np.zeros(len(NumSub))
for si in range(len(NumSub)):
    tmpsub = NumSub[si]
    tmpdata = df.loc[df['SubjectNumber'] == NumSub[si]]
    tr_num = int(len(tmpdata))
    all_tr[si] = tr_num
    tr_index = 0

del_indx = np.where(all_tr < 2900)[0]

for i in range(len(del_indx)):
    df = df.drop(index=df.loc[df['SubjectNumber'] == NumSub[del_indx[i]]].index)

NumSub = np.delete(NumSub, del_indx)
# NumSub = np.delete(NumSub, [15, 14, 11, 9, 2])

PATH = "D:/IPM_DDM/DDMParams.pkl"
with open(PATH, 'rb') as f:
     SubRes, NumQ_RT, NumQ_learning, RTMaxRange, Time, n_iter_init, n_iter_boost, n_calls, n_randomstate, numtr_percoh = pickle.load(f)


letmwknow=1
numsubs = len(SubRes)

params = np.zeros((numsubs, 3, 4))
for i in range(numsubs):
     for ci in range(3):
          params[i, ci, :] = SubRes[i][ci][0][0]   # 0 only init


params = np.delete(params, del_indx, axis=0)
# params = np.delete(params, [15, 14, 11, 9, 2], axis=0)


NumQ_learning = 3
NumTrs = np.zeros((len(NumSub), NumQ_learning))
SubRes = []
for si in range(len(NumSub)):
    tmpsub = NumSub[si]
    tmpdata = df.loc[df['SubjectNumber'] == NumSub[si]]
    numsessiosn = tmpdata['SessionNumber'].unique()
    cutsize = np.round(len(numsessiosn)/NumQ_learning).astype(int)
    condition_res = []
    plt.figure()
    plt.title('#' + str(si))
    for ti in range(NumQ_learning-1):
        print('################ Subject (' + str(si) + ') Condition (' + str(ti) + ') ####################')
        tbin = np.arange(ti*cutsize, (ti+1)*cutsize) + 1
        tmp_session_data = tmpdata.loc[(tmpdata['SessionNumber'] >= tbin[0]) & (tmpdata['SessionNumber'] <= tbin[-1])]
        NumTrs[si, ti] = len(tmp_session_data)
        #### Fit model ###########

        BehaveAllTr = (tmp_session_data["Coherence"] / 100).to_numpy()
        BehaveRT = tmp_session_data["ReactionTime"].to_numpy()
        BehaveACC = tmp_session_data["ACC"].to_numpy()
        acc_behave, rt_behave = give_behave(BehaveAllTr, BehaveACC, BehaveRT)
        acc_model, rt_model = give_model_behave(params[si, ti, :])

        plt.subplot(2, 3, ti+1)
        plt.plot(coherence, acc_behave, '.', ms=10)
        plt.plot(cxbin, acc_model)


        plt.subplot(2, 3, ti+4)
        plt.plot(coherence, rt_behave, '.', ms=10)
        plt.plot(cxbin, rt_model)




    tmp_session_data = tmpdata.loc[(tmpdata['SessionNumber'] > tbin[-1])]
    NumTrs[si, NumQ_learning-1] = len(tmp_session_data)
    print('################ Subject (' + str(si) + ') Condition (' + str(ti + 1) + ') ####################')
    BehaveAllTr = (tmp_session_data["Coherence"] / 100).to_numpy()
    BehaveRT = tmp_session_data["ReactionTime"].to_numpy()
    BehaveACC = tmp_session_data["ACC"].to_numpy()

    BehaveAllTr = (tmp_session_data["Coherence"] / 100).to_numpy()
    BehaveRT = tmp_session_data["ReactionTime"].to_numpy()
    BehaveACC = tmp_session_data["ACC"].to_numpy()
    acc_behave, rt_behave = give_behave(BehaveAllTr, BehaveACC, BehaveRT)
    acc_model, rt_model = give_model_behave(params[si, ti, :])

    plt.subplot(2, 3, ti + 2)
    plt.plot(coherence, acc_behave, '.', ms=10)
    plt.plot(cxbin, acc_model)

    plt.subplot(2, 3, ti + 5)
    plt.plot(coherence, rt_behave, '.', ms=10)
    plt.plot(cxbin, rt_model)




# PATH = "D:/IPM_DDM/DDMParams.pkl"
# with open(PATH, 'wb') as f:
#      pickle.dump([SubRes, NumQ_RT, NumQ_learning, RTMaxRange, Time, n_iter_init, n_iter_boost, n_calls, n_randomstate, numtr_percoh], f)




plt.show()