from get_features import read_traj, norm, diffusivity, TAMSD
from generating_data import read_real_expo, dirmake
from numpy import log
import pandas as pd
from matplotlib import pyplot as plt
import multiprocessing as mp
from functools import partial

def estimate_expo(t, tamsds, D, T):
  log_t_2 = [log(i) ** 2 for i in t]
  log_t = [log(i) for i in t]
  s_log_t_2 = sum(log_t_2)
  s_log_t = sum(log_t)
  log_rho = [log(tamsds[i]) for i in range(1, T)]
  s_log_t_x_log_rho = 0
  for i in range(T - 1):
    s_log_t_x_log_rho += log_t[i] * log_rho[i]
  
  return (s_log_t_x_log_rho - log(4 * D) * s_log_t) / s_log_t_2

def TAMSD_estimation_traj(part, traj_num, give):
  global liczydlo
  if part == 1:
    exps, traj = give
    x, y = traj
    T = len(x) - 1
    real_exp = exps
    r = norm(x, y, T)
    tamsds = TAMSD(r, T)
    D = diffusivity(tamsds[1])
    t = range(1, T + 1)
    est_exp = estimate_expo(t, tamsds, D, T)
    result = D, real_exp, est_exp, tamsds
    liczydlo += 1
    if liczydlo%500 == 0:
      print(f'TAMSDS - estymacja - {liczydlo}/{traj_num/3} --- estimate -> {est_exp}/{real_exp} <- real')
    return result

def TAMSD_estimation(trajectories, exps, part):
  global liczydlo
  if part == 1:
    print('Obliczanie estymacji TAMSDS...')
    liczydlo = 0
    traj_num = len(trajectories)
    traj_info = pd.DataFrame(columns = ['D', 'expo', 'expo_est', 'tamsds'],
                            index = range(traj_num))
    # 2 argumenty iterwane do poola
    give = []
    for i in range(traj_num):
      give.append([exps[i], trajectories[i]])
    with mp.Pool(3) as pool:
      temp = partial(TAMSD_estimation_traj, part, traj_num)
      result = pool.map(temp, give)
      pool.close()
      pool.join()
    print(' --- ZAKOŃCZONO')
    print('Translacja wyników TAMSD...')
    liczydlo = 0
    for i in result:
      traj_info.loc[liczydlo] = i
      liczydlo += 1
      if liczydlo%500 == 0:
        print(f'TAMSD - translacja - {liczydlo}/{traj_num}')
    print(' --- ZAKOŃCZONO')
    path = 'data/part1/TAMSD/'
    dirmake(path)
    fname = path + str('estimated.csv')
    print(f'Zapisywanie wyników do pliku {fname}')
    traj_info.to_csv(fname)
    print(' --- ZAKOŃCZONO')
