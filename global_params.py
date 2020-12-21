
from datetime import datetime

def logg(text):
    with open('log.txt', 'a') as f:
        f.write(f'[{datetime.now()}] {text}\n')
        
        
        
# part1
N_long = 110000
number_to_learn = 100000
# number_to_learn = 1000
MODELS = ['attm', 'ctrw', 'fbm', 'lw', 'sbm']
colors = ['darkorange', 'darkgreen', 'royalblue', 'pink', 'red', 'magenta']
color_maps = ['Oranges', 'Greens', 'Blues', 'RdPu', 'Reds', 'BuPu']
