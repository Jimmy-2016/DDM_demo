
import numpy as np
import matplotlib.pyplot as plt
import pickle
import matplotlib
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 15}

matplotlib.rc('font', **font)



NumIter = 30
ParamsMat = np.zeros((NumIter, 4, 2))
for pi in range(NumIter):
    with open('./ParamsData/' + str(pi) + '.pkl', 'rb') as f:
        param_Choice, param_RT, DDMTr, ParamVars, Behave_ProbMat = pickle.load(f)

    with open('./ParamsData/' + str(pi) + '_rec.pkl', 'rb') as f:
        Params, func_vals, Vars0 = pickle.load(f)


    ParamsMat[pi, :, 0] = ParamVars
    ParamsMat[pi, :, 1] = Params


for i in range(4):
    plt.figure()
    plt.scatter(ParamsMat[:, i, 0], ParamsMat[:, i, 1], s=40)
    plt.title(f'Correlation  = {np.corrcoef(ParamsMat[:, i, 0], ParamsMat[:, i, 1])[1][0]:.3f}')
    plt.ylabel('Recovered Param')
    plt.xlabel('True Param')



plt.show()