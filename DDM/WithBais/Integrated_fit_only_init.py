import pickle
import random

import numpy as np
import random as rand
from skopt import gp_minimize
from scipy import stats
import pandas as pd

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


def simple_fit(ACC, RT, Trials,  Vars0):
    global DDMTr, Behave_ProbMat
    DDMTr = np.random.permutation(np.repeat(np.array([1.6, 3.2, 6.4, 12.8, 25.6]), numtr_percoh)/100)
    Direction = np.ones_like(DDMTr)
    Direction[int(len(Direction) / 2):] = -1
    Direction = np.random.permutation(Direction)
    DDMTr = DDMTr * Direction

    Behave_ProbMat = calMat(ACC, RT, Trials)
    res = gp_minimize(DDMModel,  # the function to minimize
                      [(5., 70.), (.05, .9), (50, 900), (-20.0, 20.0)],  # the bounds on each dimension of x
                      x0=Vars0,  # the starting point
                      n_calls=n_calls,  # the number of evaluations of f including at x0
                      n_random_starts=n_randomstate,  # the number of random initial points
                      random_state=777,
                      verbose=verbose)

    Params = res.x
    func_vals = res.func_vals

    return Params, func_vals, Behave_ProbMat


def fittmodel_init(ACC, RT, Trials):
    init_Params = []
    init_func_vals = []
    bb_mats = []
    for init_i in range(n_iter_init):
        print('Iteration ' + str(init_i) + ' of Random Initial Points')
        B0 = np.random.uniform(5.0, 70.0)
        K0 = np.random.uniform(.05, .9)
        ndt0 = np.random.uniform(50, 900)
        sp0 = np.random.uniform(-20.0, 20.0)

        Vars0 = [B0, K0, ndt0, sp0]
        res = simple_fit(ACC=ACC, RT=RT, Trials=Trials, Vars0=Vars0)
        init_Params.append(res[0])
        init_func_vals.append(res[1])
        bb_mats.append(res[2])


    indx = np.argmin(np.min(np.array(init_func_vals), axis=1))
    BestParams = init_Params[indx]
    print('Best Solution was found!')
    return BestParams, init_Params, init_func_vals, bb_mats

def Bootstrap_fitting(data, Best_Var):
    Params = []
    Func_vals = []
    bb_sample_mat = []
    for iter in range(n_iter_boost):  # Iter over Starting point
        print('Boostrap on best solution. Iteration number: ' + str(iter))

        # sampling data
        data_sampled = data.sample(n=data.shape[0], replace=True)

        sample_AllTr = (data_sampled["Coherence"] / 100).to_numpy()
        trueDir = data_sampled["TrueDirection"].to_numpy()
        trueDir[trueDir == 180] = -1
        trueDir[trueDir == 0] = 1
        sample_AllTr = trueDir * sample_AllTr
        sample_Choice = data_sampled["SubResponse"].to_numpy()
        sample_Choice[sample_Choice == 180] = -1
        sample_Choice[sample_Choice == 0] = 1
        sample_RT = data_sampled["ReactionTime"].to_numpy()
        sample_ACC = data_sampled["ACC"].to_numpy()



        ####### Params #####
        Vars0 = Best_Var
        res = simple_fit(sample_ACC, sample_RT, sample_AllTr, Vars0=Vars0)
        Params.append(res[0])
        Func_vals.append(res[1])
        bb_sample_mat.append(res[2])

    print('Boostrap Procedure is done!')
    return Params, Func_vals, bb_sample_mat



def calMat(ACC, RT, Trials):
    ## behave mat
    CutSize = RTMaxRange / NumQ_RT
    righIndex = np.where(Trials > 0)[0]
    leftIndex = np.where(Trials < 0)[0]
    Right_RT = RT[righIndex]
    Left_RT = RT[leftIndex]
    ProbMat = np.zeros([NumQ_RT, 2, 2])  # RT * Choice (Right, Left) * ACC (0, 1)
    for i in range(NumQ_RT):
        # Right
        NumRight_tmp = np.where(np.logical_and(Right_RT >= i * CutSize, Right_RT <= (i + 1) * CutSize))[0]
        corr_indx = np.where(ACC[righIndex[NumRight_tmp]] == 1)[0].shape[0]
        err_indx = np.where(ACC[righIndex[NumRight_tmp]] == 0)[0].shape[0]
        ProbMat[i, 0, 0] = corr_indx
        ProbMat[i, 0, 1] = err_indx

        # Left
        NumLeftRT_tmp = np.where(np.logical_and(Left_RT >= i * CutSize, Left_RT <= (i + 1) * CutSize))[0]
        corr_indx = np.where(ACC[leftIndex[NumLeftRT_tmp]] == 1)[0].shape[0]
        err_indx = np.where(ACC[leftIndex[NumLeftRT_tmp]] == 0)[0].shape[0]
        ProbMat[i, 1, 0] = corr_indx
        ProbMat[i, 1, 1] = err_indx

    ProbMat = ProbMat / len(Trials)

    return ProbMat


## Global Params

Params = []
Func_vals = []
n_iter_boost = 100  # Boost strap interation
n_iter_init = 30  # for working over best soution (number of initial points)
NumQ_RT = 5
SP = 0
dt = 1
RTMaxRange = 5500
Time = 5000
n_calls = 100  # 100
n_randomstate = 31  # 31
numtr_percoh = 500  # 500
verbose = 0

##  Reading data
df = pd.read_csv(r'data.csv')
NumSub = df['SubjectNumber'].unique()
NumQ_learning = 3
NumTrs = np.zeros((len(NumSub), NumQ_learning))
SubRes = []
for si in range(len(NumSub)):
    tmpsub = NumSub[si]
    tmpdata = df.loc[df['SubjectNumber'] == NumSub[si]]
    numsessiosn = tmpdata['SessionNumber'].unique()
    cutsize = np.round(len(numsessiosn)/NumQ_learning).astype(int)
    condition_res = []
    for ti in range(NumQ_learning-1):
        print('################ Subject (' + str(si) + ') Condition (' + str(ti) + ') ####################')
        tbin = np.arange(ti*cutsize, (ti+1)*cutsize) + 1
        tmp_session_data = tmpdata.loc[(tmpdata['SessionNumber'] >= tbin[0]) & (tmpdata['SessionNumber'] <= tbin[-1])]
        NumTrs[si, ti] = len(tmp_session_data)
        #### Fit model ###########

        BehaveAllTr = (tmp_session_data["Coherence"] / 100).to_numpy()
        trueDir = tmp_session_data["TrueDirection"].to_numpy()
        trueDir[trueDir == 180] = -1
        trueDir[trueDir == 0] = 1
        BehaveAllTr = trueDir * BehaveAllTr
        BehaveChoice = tmp_session_data["SubResponse"].to_numpy()
        BehaveChoice[BehaveChoice == 180] = -1
        BehaveChoice[BehaveChoice == 0] = 1
        BehaveRT = tmp_session_data["ReactionTime"].to_numpy()
        BehaveACC = tmp_session_data["ACC"].to_numpy()

        BehaveNumTr = len(BehaveAllTr)
        Behave_ProbMat = calMat(ACC=BehaveACC, RT=BehaveRT, Trials=BehaveAllTr)
        Bests, params_iter, f_eval_iter, behave_mats_iter = fittmodel_init(BehaveACC, BehaveRT, BehaveAllTr)
        # FinalParams, f_vals_sample, bb_mat_sample = Bootstrap_fitting(tmp_session_data, Bests)
        condition_res.append((Bests, params_iter, f_eval_iter, behave_mats_iter))


    tmp_session_data = tmpdata.loc[(tmpdata['SessionNumber'] > tbin[-1])]
    NumTrs[si, NumQ_learning-1] = len(tmp_session_data)
    print('################ Subject (' + str(si) + ') Condition (' + str(ti + 1) + ') ####################')
    BehaveAllTr = (tmp_session_data["Coherence"] / 100).to_numpy()
    trueDir = tmp_session_data["TrueDirection"].to_numpy()
    trueDir[trueDir == 180] = -1
    trueDir[trueDir == 0] = 1
    BehaveAllTr = trueDir * BehaveAllTr
    BehaveChoice = tmp_session_data["SubResponse"].to_numpy()
    BehaveChoice[BehaveChoice == 180] = -1
    BehaveChoice[BehaveChoice == 0] = 1
    BehaveRT = tmp_session_data["ReactionTime"].to_numpy()
    BehaveACC = tmp_session_data["ACC"].to_numpy()

    BehaveNumTr = len(BehaveAllTr)
    Behave_ProbMat = calMat(ACC=BehaveACC, RT=BehaveRT, Trials=BehaveAllTr)
    Bests, params_iter, f_eval_iter, behave_mats_iter = fittmodel_init(BehaveACC, BehaveRT, BehaveAllTr)
    # FinalParams, f_vals_sample, bb_mat_sample = Bootstrap_fitting(tmp_session_data, Bests)
    condition_res.append((Bests, params_iter, f_eval_iter, behave_mats_iter))

    SubRes.append(condition_res)


PATH = "D:/IPM_DDM/DDMParams_OnlyInit.pkl"
with open(PATH, 'wb') as f:
     pickle.dump([SubRes, NumQ_RT, NumQ_learning, RTMaxRange, Time, n_iter_init, n_iter_boost, n_calls, n_randomstate, numtr_percoh], f)




