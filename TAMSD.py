from get_features import TAMSD, read_traj, movement_to_steps, norm, diffusivity
from global_params import T_long, MODELS, SNRs
from numpy import log
import pandas as pd
from matplotlib import pyplot as plt

def estimate_expo(t, tamsds, D, T, printing = False):
  log_t_2 = [log(i) ** 2 for i in t]
  s_log_t_2 = sum(log_t_2)
  log_rho_2 = [log( tamsds[i]/(4 * D) ) ** 2 for i in range(1, T)]
  s_log_t_2_x_log_rho_2 = 0
  for i in range(T - 1):
    s_log_t_2_x_log_rho_2 += log_t_2[i] * log_rho_2[i]
  if printing:
    print(log_t_2)
    print(log_rho_2)
  return s_log_t_2 / s_log_t_2_x_log_rho_2
  

if __name__ == '__main__':
  T = T_long
  for model in [1,2]:
    print(MODELS[model])
    for SNR in SNRs:
      trajs = read_traj(model, SNR, 500)
      traj_num = len(trajs)
      print(f'Estymowanie wartości dla modelu {MODELS[model]} i SNR = {SNR}')
      l_t = 0
      traj_info = pd.DataFrame(columns=['D',
                                        'expo',
                                        'expo_est'],
                            index = range(traj_num))
      for traj in trajs:
        traj = traj[0]
        expo = traj[0]
        x = traj[1]
        y = traj[2]
        s_x, s_y = movement_to_steps(x, y, T + 1)
        s = norm(s_x, s_y, T)
        tamsds = TAMSD(s, T)
        D = diffusivity(tamsds[1])
        t = range(1, T + 1)
        printing = False
        if l_t == 100:
          printing = True
        expo_est = estimate_expo(t, tamsds, D, T, printing)
        traj_info.loc[l_t] = [D, expo, expo_est]
        if l_t == 100:
          print(D)
          print(expo)
          print(expo_est)
          plt.plot(x, y)
          plt.show()
          plt.plot([0, *t], [4 * D * i ** expo_est for i in [0, *t]], 'b')
          plt.plot([0, *t], [4 * D * i ** expo for i in [0, *t]], 'r')
          plt.show()
        l_t += 1
        if l_t%5000 == 0:
          print(f'estymacja - {MODELS[model]} - {SNR} - {l_t}/{traj_num}')
      print(' --- ZAKOŃCZONO')
      fname = f'data/TAMSD/{MODELS[model]}_{SNR}_TAMSD_estimated.csv'
      print(f'Zapis do pliku {fname}')
      traj_info.to_csv(fname)
