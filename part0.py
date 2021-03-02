import andi
import matplotlib.pyplot as plt
from generating_data import dirmake
from global_params import colors, logg, color_maps
from TAMSD import TAMSD_estimation, estimate_expo

def example_trajs():
    print('Generowanie i zapisywanie przykładowych trajektorii...')
    path = 'data/part0/example_traj'
    dirmake(path)
    logg('Generowanie przykładowych trajektorii - start')
    AD = andi.andi_datasets()
    for model in range(5):
        try:
            dataset = AD.create_dataset(100, 1,[0.7],[model], 2)
        except:
            dataset = AD.create_dataset(100, 1,[1.7],[model], 2)

        x = dataset[0][2:102]
        y = dataset[0][102:]
        plt.figure(figsize= (2,2))
        plt.cla()
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(AD.avail_models_name[model], loc = 'left')
        plt.plot(x,y, color = colors[model], linewidth=2, alpha=0.5)
        plt.scatter(x,y, c=range(len(x)), cmap=color_maps[model], marker = '.', s=100)
        plt.savefig(path+'/'+str(AD.avail_models_name[model])+'.pdf', transparent = True, bbox_inches = 'tight', dpi = 300)
    logg('Generowanie przykładowych trajektorii - stop')
    print(' --- ZAKOŃCZONO')

def example_TAMSD():
    print('Generowanie i zapisywanie przykładowych TAMSD...')
    path = 'data/part0/example_TAMSD'
    dirmake(path)
    logg('Generowanie przykładowych TAMSD - start')
    
    AD = andi.andi_datasets()
    dataset = AD.create_dataset(200, 1,[0.7],[2], 2)
    x = dataset[0][2:202]
    y = dataset[0][202:]
    trajectory = [x, y]
    D, expo, expo_est, tamsds = TAMSD_estimation(trajectory, 0.7, 0, 'A')
    tamsds = tamsds[:100]
    t = range(1, len(tamsds)+1)
    expo_est = estimate_expo(t, tamsds, D, 100)
    
    plt.cla()
    plt.figure(figsize=(3,3))
    plt.plot(t, tamsds, '.', label = 'punkty TAMSD')
    plt.plot(t, [4 * D * i ** expo_est for i in t], 'b', label = r'Wyestymowana krzywa wzorcowa')
    plt.plot(t, [4 * D * i ** expo for i in t], 'r', label = r'Prawdziwa krzywa wzorcowa')
    plt.xlabel('t')
    plt.ylabel(r'$\rho(t)$')
    plt.title('c', loc = 'left')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.savefig(path+'/TAMSD.pdf', transparent = True, bbox_inches = 'tight', dpi = 300)
    
    
    plt.cla()
    plt.loglog(t, tamsds, '.', label = 'punkty TAMSD')
    plt.loglog(t, [4 * D * i ** expo_est for i in t], 'b', label = r'Wyestymowana krzywa TAMSD')
    plt.loglog(t, [4 * D * i ** expo for i in t], 'r', label = r'Prawdziwa krzywa wzorcowa')
    plt.xlabel('t')
    plt.ylabel(r'$\rho(t)$')
    plt.legend(loc='lower left', bbox_to_anchor=(1.05, 1))
    plt.title('d', loc = 'left')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.savefig(path+'/TAMSD_loglog.pdf', transparent = True, bbox_inches = 'tight', dpi = 300)
        
    # perfekcyjne tamsd
    plt.cla()
    D = 0.3
    t = [0.1 * i for i in range(101)]
    exps = [0.7,1,1.3]
    label = ['superdyfuzja', 'dyfuzja normalna', 'subdyfuzja']
    for expo in exps:
        plt.plot(t, [4 * D * i ** expo for i in t], color = colors[exps.index(expo)], label = r'$\alpha=\ $'+str(expo)+' - '+label[-exps.index(expo)-1])
    plt.xlabel('t')
    plt.ylabel(r'$\rho(t)$')
    plt.legend(loc='lower left', bbox_to_anchor=(1.05, 1), ncol = 3)
    plt.title('a', loc = 'left')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.savefig(path+'/perfect_TAMSD.pdf', transparent = True, bbox_inches = 'tight', dpi = 300)
    
    # perfekcyjne tamsd - loglog
    plt.cla()
    D = 1
    t = [0.1 * i for i in range(101)]
    exps = [0.7,1,1.3]
    for expo in exps:
        plt.loglog(t, [4 * D * i ** expo for i in t], color = colors[exps.index(expo)], label = r'$\alpha=\ $'+str(expo))
    plt.xlabel('t')
    plt.ylabel(r'$\rho(t)$')
    plt.title('b', loc = 'left')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.savefig(path+'/perfect_TAMSD_loglog.pdf', transparent = True, bbox_inches = 'tight', dpi = 300)
    
    logg('Generowanie przykładowych TAMSD - stop')
    print(' --- ZAKOŃCZONO')
