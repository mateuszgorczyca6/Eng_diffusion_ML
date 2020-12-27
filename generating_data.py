import andi
from plotting import plot_traj
from os import path,mkdir
import pandas as pd
from global_params import logg as log
from datetime import datetime
from math import floor
from stolen_from_andi import andi_dataset_2

AD = andi.andi_datasets()

def dirmake(ptch):
    a=ptch.split('/')
    pach=a[0]
    if not path.exists(pach):mkdir(pach)
    for i in a[1:]:
        pach+='/'+i
        if not path.exists(pach):mkdir(pach)

def generate_trajectories(N, part, Model):
  print('Generowanie trajektorii')
  if part == 1:
    path = f'data/part1/model{Model}/generated/'
    dirmake(path)
    start = datetime.now()
    log(f'GEN - generowanie trajektorii - start [{path}]')
    if Model == 'A':
      AD.andi_dataset(N = N, tasks = 1, dimensions = 2, save_dataset = True, min_T=99, max_T=100, path_datasets=path)
    if Model == 'B':
      AD.andi_dataset(N = N, tasks = 1, dimensions = 2, save_dataset = True, min_T=20, max_T=21, path_datasets=path)
    if Model == 'C':
      AD.andi_dataset(N = N, tasks = 1, dimensions = 2, save_dataset = True, min_T=20, max_T=100, path_datasets=path)
    stop = datetime.now()
    log(f'GEN - generowanie trajektorii - koniec [{stop - start}]')
  
  if part == 2:
    path = f'data/part2/model{Model}/generated/'
    dirmake(path)
    
    models = [2]  # FBM
    if Model == 'A':
      T = 100
    if Model == 'B':
      T = 20
    
    log(f'GEN - generowanie trajektorii - start [{path}]')
    start = datetime.now()
    if Model in ['A', 'B']:
      andi_dataset_2(N = N, tasks = [1], dimensions = [2], save_dataset = True, min_T = T - 1, max_T = T, path_datasets = path, models = models)
    else:
      andi_dataset_2(N = N, tasks = [1], dimensions = [2], save_dataset = True, min_T = 20, max_T = 100, path_datasets = path, models = models)
    stop = datetime.now()
    log(f'GEN - generowanie trajektorii - koniec [{stop - start}]')
  print(' --- ZAKOŃCZONO')
    
def read_trajectories(part, Model, mode = 'all', N = 0):
  print('Ładowanie trajektorii')
  if part == 1 or part == 2:
    path = f'data/part{part}/model{Model}/generated/'
    trajectories = []
    log(f'GEN - odczyt trajektorii - start [{path+"task1.txt"}]')
    start = datetime.now()
    file_path = path+'task1.txt'
    with open(file_path) as f:
      n = 0
      trajs = f.readlines()
      
    if mode == 'start':
      trajs = trajs[:N]
    if mode == 'end':
      trajs = trajs[N:]
      
    l_t = len(trajs)
    for traj in trajs:
      traj = traj.strip()
      traj = traj.split(';')[1:]
      T = int(len(traj) / 2)
      sx, sy = traj[:T], traj[T:] # strings
      x, y = [], []               # floats
      for i in range(len(sx)):
        x.append(float(sx[i]))
        y.append(float(sy[i]))
      trajectories.append([x, y])
      n += 1
      if n%500 == 0:
        print(f'czytanie trajektorii - {n}/{l_t}')
    stop = datetime.now()
    log(f'GEN - odczyt trajektorii - koniec {stop - start}')
  print(' --- ZAKOŃCZONO')
  return trajectories
  # return trajectories[:1100]

def read_real_expo(part, Model):
  print('Ładowanie prawdziwych wartości exp...')
  if part == 1 or part == 2:
    path = f'data/part{part}/model{Model}/generated/'
    exps = []
    with open(path + str('ref1.txt')) as f:
      for traj in f.readlines():
        traj = traj.strip()
        traj = float(traj.split(';')[1])
        exps.append(traj)
  print(' --- ZAKOŃCZONO')
  return exps

def read_TAMSD(part, Model):
  print('Ładowanie wartości wyestymowanych w TAMSD')
  if part == 1 or part == 2:
    traj_info = pd.read_csv(f'data/part{part}/model{Model}/TAMSD/estimated.csv')
  print(' --- ZAKOŃCZONO')
  return traj_info

def read_ML_features(part, Model):
  print('Ładowanie właściwości ruchu dla ML')
  if part == 1 or part == 2:
    traj_info = pd.read_csv(f'data/part{part}/model{Model}/ML/features.csv', index_col='Unnamed: 0')
    # overflow delete
    traj_info = traj_info[traj_info < 10**10 ]
    traj_info = traj_info.dropna()
  print(' --- ZAKOŃCZONO')
  return traj_info