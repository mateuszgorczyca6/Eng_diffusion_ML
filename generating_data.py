import andi
from plotting import plot_traj
from os import path,mkdir
import pandas as pd

AD = andi.andi_datasets()

def dirmake(ptch):
    a=ptch.split('/')
    pach=a[0]
    if not path.exists(pach):mkdir(pach)
    for i in a[1:]:
        pach+='/'+i
        if not path.exists(pach):mkdir(pach)

def generate_trajectories(N, T, part):
  print('Generowanie trajektorii')
  if part == 1:
    path = 'data/part1/generated/'
    dirmake(path)
    AD.andi_dataset(N = N, tasks = 1, dimensions = 2, save_dataset = True, min_T=T-1, max_T=T, path_datasets=path)
  print(' --- ZAKOŃCZONO')

def read_trajectories(part):
  print('Ładowanie trajektorii')
  if part == 1:
    path = 'data/part1/generated/'
    dirmake(path)
    trajectories = []
    with open(path+'task1.txt') as f:
      n = 0
      trajs = f.readlines()
    l_t = len(trajs)
    for traj in trajs:
      traj = traj.strip()
      traj = traj.split(';')[1:]
      T = int(len(traj) / 2)
      sx, sy = traj[:T], traj[T:]
      x, y = [], []
      for i in range(len(sx)):
        x.append(float(sx[i]))
        y.append(float(sy[i]))
      trajectories.append([x, y])
      n += 1
      if n%500 == 0:
        print(f'czytanie trajektorii - {n}/{l_t}')
  print(' --- ZAKOŃCZONO')
  return trajectories

def read_real_expo(part):
  print('Ładowanie prawdziwych wartości exp...')
  if part == 1:
    path = 'data/part1/generated/'
    exps = []
    with open(path + str('ref1.txt')) as f:
      for traj in f.readlines():
        traj = traj.strip()
        traj = float(traj.split(';')[1])
        exps.append(traj)
  print(' --- ZAKOŃCZONO')
  return exps

def read_TAMSD(part):
  print('Ładowanie wartości wyestymowanych w TAMSD')
  if part == 1:
    traj_info = pd.read_csv('data/part1/TAMSD/estimated.csv')
  print(' --- ZAKOŃCZONO')
  return traj_info

def read_ML_features(part):
  print('Ładowanie właściwości ruchu dla ML')
  if part == 1:
    traj_info = pd.read_csv('data/part1/ML/features.csv', index_col='Unnamed: 0')
  print(' --- ZAKOŃCZONO')
  return traj_info