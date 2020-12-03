import andi
from global_params import *
AD = andi.andi_datasets()

def generate_trajectories(N, models, T, expo):
  '''Models:
  'attm', 'ctrw', 'fbm', 'lw', 'sbm'''
  trajectories = AD.create_dataset(T = T,
                                   N = N,
                                   exponents = expo,
                                   models = models,
                                   dimension=2,
                                   save_trajectories = True,
                                   path = 'data/')

def noise_trajectories(N, models, T, expo, SNR):
  '''Dodaje szum gaussa do trajektorii.
  Zwrac: trajektoria bez szumu, trajektoria z szumem'''
  trajectories = AD.create_dataset(T = T,
                                   N = N,
                                   exponents = expo,
                                   models = models,
                                   dimension=2,
                                   load_trajectories = True,
                                   path = 'data/')
  noisy_trajectories = AD.create_noisy_localization_dataset(dataset = trajectories.copy(),
                                                            T = T,
                                                            N = N,
                                                            exponents = expo,
                                                            models = models,
                                                            dimension=2,
                                                            mu=0,
                                                            sigma=1/SNR)
  return trajectories, noisy_trajectories

def noisy_save(N, model, T, expo, SNR = -1):
  models = ['attm', 'ctrw', 'fbm', 'lw', 'sbm']
  data = ""
  l_e = 0
  for ex in expo:
    if SNR == -1:
      trajs, _ = noise_trajectories(N, model, T, ex, 1)
    else:
      _, trajs = noise_trajectories(N, model, T, ex, SNR)
    l_t = 0
    for traj in trajs:
      x = [0, *traj[2:T+2]]
      y = [0, *traj[T+2:]]
      data += str(ex) + ", " + str(x) + ", " + str(y) + "\n"
      l_t += 1
    l_e += 1
    print(f"{models[model]} - {SNR} - {l_e}/{len(expo)}")
  with open(f"data/{models[model]}_noisy_{SNR}.txt", "w") as f:
    f.write(data)

if __name__ == "__main__":
  ## Generowanie trajektorii długich (T=100)
  T=T_long
  # CTRW a \in {0.1, 0,2, ..., 1}
  model = 1 # CTRW
  generate_trajectories(N, model, T, expo_CTRW)
  # FBM a \in {0.25, 0.4, ..., 1.6}
  model = 2 # FBM
  generate_trajectories(N, model, T, expo_FBM)
  print("Zaszumianie i zapisywanie trajektorii CTRW.")
  for SNR in [-1, 1, 3, 10, 100]:
    print(f"SNR = {SNR}")
    noisy_save(N, 1, T, expo_CTRW, SNR)
  print("Zaszumianie i zapisywanie trajektorii FBM.")
  for SNR in [-1, 1, 3, 10, 100]:
    print(f"SNR = {SNR}")
    noisy_save(N, 2, T, expo_FBM, SNR)