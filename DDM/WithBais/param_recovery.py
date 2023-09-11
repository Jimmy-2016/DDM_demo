

import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy import stats
from skopt import gp_minimize



## Functions


def DDMSim(Vars):
    Bound = Vars[0]
    k = Vars[1]
    ndt = Vars[2]
    SP = Vars[3]

    V = SP * np.ones((len(DDMTr_sim), len(timelap)))
    #### Main Loop ####
    for t in range(len(timelap) - 1):
        dX = np.random.normal(k * DDMTr_sim, np.ones([1, len(DDMTr_sim)]))
        V[:, t + 1] = V[:, t] + dX

    def GetBehave(V, DDMTr_sim):
        RT = np.zeros(len(DDMTr_sim))
        Choice = np.zeros(len(DDMTr_sim))
        Indx_in = V > Bound
        Indx_out = V < -Bound
        for i in range(len(DDMTr_sim)):
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


    ModelRT, ModelChoice = GetBehave(V, DDMTr_sim)
    ModelRT = ModelRT + ndt
    ModelACC = ModelChoice == np.sign(DDMTr_sim)

    return ModelRT, ModelChoice, ModelACC


def DDMModel(Vars0):
    b_in = Vars0[0]
    k_in = Vars0[1]
    ndt_in = Vars0[2]
    sp_in = Vars0[3]

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

    CutSize = RTMaxRange / NumQ_RT
    righIndex = np.where(DDMTr > 0)[0]
    leftIndex = np.where(DDMTr < 0)[0]
    Right_RT = ModelRT[righIndex]
    Left_RT = ModelRT[leftIndex]
    Model_ProbMat = np.zeros([NumQ_RT, 2, 2])
    for i in range(NumQ_RT):
        NumRight_tmp = np.where(np.logical_and(Right_RT >= i * CutSize, Right_RT <= (i + 1) * CutSize))[0]
        corr_indx = np.where(ModelACC[righIndex[NumRight_tmp]] == 1)[0].shape[0]
        err_indx = np.where(ModelACC[righIndex[NumRight_tmp]] == 0)[0].shape[0]
        Model_ProbMat[i, 0, 0] = corr_indx
        Model_ProbMat[i, 0, 1] = err_indx

        # Left
        NumLeftRT_tmp = np.where(np.logical_and(Left_RT >= i * CutSize, Left_RT <= (i + 1) * CutSize))[0]
        corr_indx = np.where(ModelACC[leftIndex[NumLeftRT_tmp]] == 1)[0].shape[0]
        err_indx = np.where(ModelACC[leftIndex[NumLeftRT_tmp]] == 0)[0].shape[0]
        Model_ProbMat[i, 1, 0] = corr_indx
        Model_ProbMat[i, 1, 1] = err_indx

    Model_ProbMat = Model_ProbMat/len(DDMTr)
    Model_ProbMat = Model_ProbMat + 1e-8


    CostVal = np.nansum(((Behave_ProbMat - Model_ProbMat) ** 2) / Model_ProbMat)
    CostVal = stats.chi2.cdf(CostVal, 1)


    return CostVal



## Set Params
NumIter = 30
n_calls = 200
n_start = 100
NumQ_RT = 7
Time = 5000
RTMaxRange = 5500
numtr_percoh = 1000
numtr_percoh_sim = 250



DDMTr = np.random.permutation(np.repeat(np.array([1.6, 3.2, 6.4, 12.8, 25.6]), numtr_percoh)/100)
Direction = np.ones_like(DDMTr)
Direction[int(len(Direction)/2):] = -1
Direction = np.random.permutation(Direction)

DDMTr = DDMTr*Direction
timelap = np.linspace(0, Time, int(Time))
Behave_ProbMat = np.zeros((NumQ_RT, 2, 2))  # RT * Choice (Right, Left) * ACC(0, 1)


DDMTr_sim = np.random.permutation(np.repeat(np.array([1.6, 3.2, 6.4, 12.8, 25.6]), numtr_percoh_sim)/100)
Direction_sim = np.ones_like(DDMTr_sim)
Direction_sim[int(len(Direction_sim)/2):] = -1
Direction_sim = np.random.permutation(Direction_sim)

DDMTr_sim = DDMTr_sim*Direction_sim



for pi in range(NumIter):
    print(pi)
    B = np.random.uniform(5.0, 70.0)
    K = np.random.uniform(.05, .9)
    ndt = int(np.random.uniform(50, 900))
    SP = np.random.uniform(-20, 20)
    ParamVars = [B, K, ndt, SP]


    param_RT, param_Choice, param_ACC = DDMSim(ParamVars)

    CutSize = RTMaxRange / NumQ_RT
    righIndex = np.where(DDMTr_sim > 0)[0]
    leftIndex = np.where(DDMTr_sim < 0)[0]
    Right_RT = param_RT[righIndex]
    Left_RT = param_RT[leftIndex]
    Behave_ProbMat = np.zeros([NumQ_RT, 2, 2])  # RT * Choice (Right, Left) * ACC (0, 1)
    for i in range(NumQ_RT):
        # Right
        NumRight_tmp = np.where(np.logical_and(Right_RT >= i * CutSize, Right_RT <= (i + 1) * CutSize))[0]
        corr_indx = np.where(param_ACC[righIndex[NumRight_tmp]] == 1)[0].shape[0]
        err_indx = np.where(param_ACC[righIndex[NumRight_tmp]] == 0)[0].shape[0]
        Behave_ProbMat[i, 0, 0] = corr_indx
        Behave_ProbMat[i, 0, 1] = err_indx

        # Left
        NumLeftRT_tmp = np.where(np.logical_and(Left_RT >= i * CutSize, Left_RT <= (i + 1) * CutSize))[0]
        corr_indx = np.where(param_ACC[leftIndex[NumLeftRT_tmp]] == 1)[0].shape[0]
        err_indx = np.where(param_ACC[leftIndex[NumLeftRT_tmp]] == 0)[0].shape[0]
        Behave_ProbMat[i, 1, 0] = corr_indx
        Behave_ProbMat[i, 1, 1] = err_indx

    Behave_ProbMat = Behave_ProbMat / len(DDMTr_sim)

    with open('./ParamsData/' + str(pi) + '.pkl', 'wb') as f:
        pickle.dump([param_Choice, param_RT, DDMTr_sim, ParamVars, Behave_ProbMat], f)

    B0 = np.random.uniform(5.0, 70.0)
    K0 = np.random.uniform(.05, .9)
    ndt0 = int(np.random.uniform(50, 900))
    SP0 = np.random.uniform(-20, 20)
    Vars0_fit = [B0, K0, ndt0, SP0]

    res = gp_minimize(DDMModel,  # the function to minimize
                      [(5., 70.), (.05, .9), (50, 900), (-20.0, 20.0)],  # the bounds on each dimension of x
                      x0=Vars0_fit,  # the starting point
                      n_calls=n_calls,  # the number of evaluations of f including at x0
                      n_random_starts=n_start,  # the number of random initial points
                      random_state=777,
                      n_initial_points=200,
                      verbose=0)

    Params = res.x
    func_vals = res.func_vals

    with open('./ParamsData/' + str(pi) + '_rec.pkl', 'wb') as f:
        pickle.dump([Params, func_vals, Vars0_fit], f)

##

