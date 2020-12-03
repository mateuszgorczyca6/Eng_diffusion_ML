N = 10000
T_long = 100
expo_CTRW = [0.1*i for i in range(1,11)]
expo_FBM = [0.1+0.15*i for i in range(1,11)]
MODELS = ['attm', 'ctrw', 'fbm', 'lw', 'sbm']
SNRs = [-1,1,3,10,100]