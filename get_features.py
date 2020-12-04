import numba
import multiprocessing as mp
from functools import partial
from global_params import SNRs, MODELS, T_long
import pandas as pd
from numpy import linalg as LA
from numpy import log, exp, mean, var
from numpy import arange, array, zeros
from time import time

def TAMSD(s, T):
  tamsds = [0] * T
  for n in range(1, T): # gaps
    suma = 0
    for i in range(T - n):
      suma += (s[i+n] - s[i]) ** 2
    tamsds[n] = suma / (T - n)
  return tamsds

def read_traj(model, SNR, n = 0):
  models = ['attm', 'ctrw', 'fbm', 'lw', 'sbm']
  fname = f'data/{models[model]}_noisy_{SNR}.txt'
  traj = []
  print(f'ODCZYT PLIKU {fname}', end='')
  with open(fname) as f:
    lines = f.readlines()
  print(' --- ZAKOŃCZONO')
  print('Zapisywanie wartości:')
  if n == 0:
    num_lines = len(lines)
  else:
    num_lines = n
  for l_n in range(num_lines):
    line = lines[l_n]
    traj.append([eval(line.strip())])
    if l_n%5000==0:
      print(f'{fname} - {l_n}/{num_lines}')
  return traj

def movement_to_steps(x,y,T):
    ''' Zwraca krok zamiast pozycji.'''
    s_x = [0] * (T - 1)
    s_y = [0] * (T - 1)
    for i in range(1, T):
        s_x[i-1] = x[i] - x[i-1]
        s_y[i-1] = y[i] - y[i-1]
    return s_x, s_y
    
def norm(x,y, T):
    l = [0] * (T)
    for i in range(T):
        l[i] = LA.norm([x[i], y[i]])
    return l

def diffusivity(rho1):
    return rho1 / 4

def efficiency(x, y, s, T):
    return (T - 1) * LA.norm([x[-1],y[-1]]) ** 2 / sum([s[i] ** 2 for i in range(T)])

def slowdown(n, s):
    return sum(s[-n:]) / sum(s[:n])

def msd_ratio(tamsds, T):
    kappas = [0] * (T - 1)
    for n in range(T - 1):
        kappas[n] = tamsds[n] / tamsds[n+1] - n / (n+1)
    return kappas

def TAMSD4(s, T):
    tamsds = [0] * T
    for n in range(1, T): # gaps
        suma = 0
        for i in range(T - n):
            suma += (s[i+n] - s[i]) ** 4
        tamsds[n] = suma / (T - n)
    return tamsds

def antigaussinity(tamsds, tamsds4, T):
    Gs = [0] * (T - 1)
    for delt in range(1, T):
        Gs[delt - 1] = tamsds4[delt] / (2 * tamsds[delt])
    return Gs

def straightness(s, length):
    return s[-1] / length

def autocorr(s, T):
    E = mean(s)
    Var = var(s)
    ds=[s[i] - E for i in range(T)]
    autocor = [0] * T
    autocor[0] = 1
    for lag in range(1, T):
        acov = sum([ds[i] * ds[i + lag] for i in range(T-lag)])
        autocor[lag] = acov / ((T - lag) * Var)
    return autocor

def max_distance(x, y, T):
    max_d = 0
    for i in range(T):
        for j in range(i, T):
            d = LA.norm([ x[i] - x[j], y[i] - y[j] ])
        if d > max_d:
            max_d = d
    return max_d

def trappedness(D, T, max_dist):
    return 1-exp(0.2045 - 0.25117 * D * T / (max_dist / 2) ** 2)

def fractal_dimension(T, max_dist, length):
    return log(T) / log(T * max_dist / length)

def get_info(T, s_n, model, SNR, traj_num, traj):
    global l_t
    traj = traj[0]
    ex = traj[0]
    x = traj[1]                                       # położenie x
    y = traj[2]                                       # położenie y
    s_x, s_y = movement_to_steps(x, y, T+1)           # przesunięcia w x i y
    s = norm(s_x, s_y, T)                             # długości kroków
    length = sum(s)                                   # długość trajektorii (dystans)
    tamsds = TAMSD(s, T)                              # współczynniki TAMSD
    D = diffusivity(tamsds[1])                        # współczynnik dyfuzyjności
    E = efficiency(x, y, s, T)                        # wydajność
    ss = slowdown(s_n, s)                             # współczynnik spowolnienia
    kappas = msd_ratio(tamsds, T)                     # współczynniki MSD dla różnych n
    tamsds4 = TAMSD4(s, T)                            # czasowe średnie odchylenie ^4
    G = antigaussinity(tamsds, tamsds4, T)            # anty-gaussyjność
    S = straightness(s, length)                       # liniowość
    R = autocorr(s, T)                                # autokorelacja
    max_dist = max_distance(x, y, T + 1)              # maksymalny dystans między punktami
    Trap = trappedness(D, T + 1, max_dist)            # współczynnik uwięzienia
    frac_dim = fractal_dimension(T + 1, max_dist, length) # wymiar fraktalny
    l_t += 1
    if l_t%500==0:
        print(f'odczyt - {MODELS[model]} - {SNR} - {l_t}/{traj_num / 3}')
    return ex, D, E, ss, kappas, G, S, R, max_dist, Trap, frac_dim

def get_features(model, SNR):
    global l_t
    print(f'Wyciąganie danych z {MODELS[model]} dla SNR = {SNR}')
    T = T_long
    trajs = read_traj(model, SNR)
    traj_num = len(trajs)
    print()
    # odczyt danych z trajektorii
    print('Odczyt danych z trajektorii')
    l_t = 0
    with mp.Pool(3) as pool:
        temp = partial(get_info, T, 5, model, SNR, traj_num)
        result = pool.map(temp, trajs)
        pool.close()
        pool.join()
    # zapis do pandas
    print('Zmiana rozszerzenia')
    traj_info = pd.DataFrame(columns=['SNR',
                                    'alpha',
                                    'diffusivity',
                                    'efficiency',
                                    'slowdown',
                                    'MSD_ratio',
                                    'antigaussinity',
                                    'straigthness',
                                    'autocorrelation',
                                    'max_distance',
                                    'trappedness',
                                    'fractal_dim'],
                            index = range(traj_num))
    l_t = 0
    for traj in result:
        traj_info.loc[l_t] = [SNR, *traj]
        l_t += 1
        if l_t%500==0:
            print(f'translacja - {MODELS[model]} - {SNR} - {l_t}/{traj_num}')
    # zapis do pliku
    print(f'Zapisywanie danych do pliku data/{MODELS[model]}_noisy_{SNR}_features.csv ', end = '')
    traj_info.to_csv(f'data/{MODELS[model]}_noisy_{SNR}_features.csv')
    print(' --- ZAKOŃCZONO')
    return traj_info
    
def repair_fractal_dim():
    T = T_long
    #for model in [1,2]:
    for model in [1]:
        print(MODELS[model])
        for SNR in SNRs:
            trajs = read_traj(model, SNR)
            table = pd.read_csv(f'data/{MODELS[model]}_noisy_{SNR}_features.csv')
            for i in range(len(trajs)):
                if i%500 == 0:
                    print(f'{i}/{len(trajs)}')
                x = trajs[i][0][1]
                y = trajs[i][0][2]
                s_x, s_y = movement_to_steps(x, y, T+1)
                s = norm(s_x, s_y, T)
                length = sum(s)
                max_dist = table['max_distance'][i]
                frac_dim = fractal_dimension(T + 1, max_dist, length)
                table.at[i, 'fractal_dim'] = frac_dim
            table.to_csv(f'data/{MODELS[model]}_noisy_{SNR}_features.csv')

if __name__ == '__main__':
    T = T_long
    #for model in [1,2]:
    for model in [2]:
        print(MODELS[model])
        for SNR in SNRs:
            get_features(model, SNR)
    # repair_fractal_dim()