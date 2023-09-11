import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
import random as rand
from skopt import gp_minimize
from scipy import stats
import pandas as pd
import matplotlib
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 15}

matplotlib.rc('font', **font)


np.random.seed(0)
random.seed(0)


##


def DDMModel(Vars0):
    b_in = Vars0[0]
    k_in = Vars0[1]
    ndt_in = Vars0[2]
    sp_in = Vars0[3]

    timelap = np.linspace(0, Time, int(Time / dt))
    V = sp_in * np.ones((len(DDMTr), len(timelap)))
    Bound = b_in
    #### Main Loop ####
    for t in range(len(timelap) - 1):
        dX = np.random.normal(k_in * DDMTr, np.ones([1, len(DDMTr)]))
        V[:, t + 1] = V[:, t] + dX

    def GetBehave(V, DDMTr):
        RT = np.zeros(len(DDMTr))
        Choice = np.zeros(len(DDMTr))
        Indx_in = V > Bound
        Indx_out = V < -Bound
        for i in range(len(DDMTr)):
            tmpindx1 = np.where(Indx_in[i, :])[0]
            tmpindx2 = np.where(Indx_out[i, :])[0]

            if len(tmpindx1) == 0:
                tmpindx1 = np.array([len(timelap)])
            if len(tmpindx2) == 0:
                tmpindx2 = np.array([len(timelap)])

            if tmpindx1[0] < tmpindx2[0]:
                Choice[i] = 1
                RT[i] = tmpindx1[0]
            elif tmpindx1[0] > tmpindx2[0]:
                Choice[i] = -1
                RT[i] = tmpindx2[0]
            else:
                Choice[i] = np.random.choice([-1, 1])
                RT[i] = tmpindx2[0]
        return RT, Choice

    ModelRT, ModelChoice = GetBehave(V, DDMTr=DDMTr)
    ModelRT = ModelRT + ndt_in
    ModelACC = ModelChoice == np.sign(DDMTr)
    Model_ProbMat = calMat(ModelACC, ModelRT, DDMTr)
    Model_ProbMat += 1e-8
    CostVal = np.nansum(((Behave_ProbMat - Model_ProbMat) ** 2) / Model_ProbMat)
    CostVal = stats.chi2.cdf(CostVal, 1)


    return CostVal




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



PATH = "D:/IPM_DDM/DDMParams_onlyFit.pkl"

with open(PATH, 'rb') as f:
     SubRes, NumQ_RT, NumQ_learning, RTMaxRange, Time, n_init_point, n_calls, n_randomstate, numtr_percoh = pickle.load(f)


letmwknow=1
numsubs = len(SubRes)

params = np.zeros((numsubs, 3, 4))
DDMTr = 0

lw=4
msize=20
for i in range(numsubs):
     for ci in range(3):
          params[i, ci, :] = SubRes[i][ci][0]


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
    plt.figure(figsize=(10, 7))
    plt.suptitle('Sub #' + str(si+1))
    # plt.suptitle("Title centered above all subplots", fontsize=14)
    pltcounter = 0
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
        plt.plot(coherence, acc_behave, '.', ms=msize)
        plt.plot(cxbin, acc_model, lw=lw)
        if ti == 0:
            plt.ylabel('Acc (%)')



        plt.subplot(2, 3, ti+4)
        plt.plot(coherence, rt_behave, '.', ms=msize)
        plt.plot(cxbin, rt_model, lw=lw)
        pltcounter += 1


        if ti == 0:
            plt.xlabel('Coherence')
            plt.ylabel('RT (ms)')




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
    acc_model, rt_model = give_model_behave(params[si, ti+1, :])

    plt.subplot(2, 3, ti + 2)
    plt.plot(coherence, acc_behave, '.', ms=msize)
    plt.plot(cxbin, acc_model, lw=lw)

    plt.subplot(2, 3, ti + 5)
    plt.plot(coherence, rt_behave, '.', ms=msize)
    plt.plot(cxbin, rt_model, lw=lw)

    plt.savefig('./Pics/' + f'Sub{si + 1}')




# PATH = "D:/IPM_DDM/DDMParams.pkl"
# with open(PATH, 'wb') as f:
#      pickle.dump([SubRes, NumQ_RT, NumQ_learning, RTMaxRange, Time, n_iter_init, n_iter_boost, n_calls, n_randomstate, numtr_percoh], f)




plt.show()