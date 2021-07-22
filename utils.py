import numpy as np
import torch
# import pandas as pd

def combined_lable(y, z):
    target = torch.empty_like(y)
    for i, (yi, zi) in enumerate(zip(y, z)):
        if yi==0 and zi==0:
            target[i] = 0
        elif yi==0 and zi==1:
            target[i] = 1
        elif yi==1 and zi==0:
            target[i] = 2
        elif yi==1 and zi==1:
            target[i] = 3
    return target

def measures_from_Yhat(Y, Z, Yhat=None, threshold=0.5):
    assert isinstance(Y, np.ndarray)
    assert isinstance(Z, np.ndarray)
    assert Yhat is not None
    assert isinstance(Yhat, np.ndarray)
    
    if Yhat is not None:
        Ytilde = (Yhat >= threshold).astype(np.float32)
    assert Ytilde.shape == Y.shape and Y.shape == Z.shape
    
    # Accuracy
    acc = (Ytilde == Y).astype(np.float32).mean()
    # DP  
    DDP = abs(np.mean(Ytilde[Z==0])-np.mean(Ytilde[Z==1]))
    # EO
    Y_Z0, Y_Z1 = Y[Z==0], Y[Z==1]
    Y1_Z0 = Y_Z0[Y_Z0==1]
    Y0_Z0 = Y_Z0[Y_Z0==0]
    Y1_Z1 = Y_Z1[Y_Z1==1]
    Y0_Z1 = Y_Z1[Y_Z1==0]
    
    FPR, FNR = {}, {}
    FPR[0] = np.sum(Ytilde[np.logical_and(Z==0, Y==0)])/len(Y0_Z0)
    FPR[1] = np.sum(Ytilde[np.logical_and(Z==1, Y==0)])/len(Y0_Z1)

    FNR[0] = np.sum(1 - Ytilde[np.logical_and(Z==0, Y==1)])/len(Y1_Z0)
    FNR[1] = np.sum(1 - Ytilde[np.logical_and(Z==1, Y==1)])/len(Y1_Z1)
    
    TPR_diff = abs((1-FNR[0]) - (1-FNR[1]))
    FPR_diff = abs(FPR[0] - FPR[1])
    DEO = TPR_diff + FPR_diff
    DEOpp = abs((1-FNR[0]) - (1-FNR[1]))
    # eacc = (FPR[0] + FPR[1] + FNR[0] + FNR[1])/4
    
    return acc, DDP, DEO/2, DEOpp
    # data = [acc, DDP, DEO]
    # columns = ['acc', 'DDP', 'DEO']
    # return pd.DataFrame([data], columns=columns)
