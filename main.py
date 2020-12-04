from global_params import T_long, N_long
from generating_data import generate_trajectories, read_trajectories, dirmake, read_TAMSD
from plotting import plot_traj, plot_TAMSD
from TAMSD import TAMSD_estimation

colors = ['darkorange', 'darkgreen', 'royalblue'] # kolory do rysowania plotów
part = int(input("Jaką część projektu chcesz uruchomić: "))
if part == 1:
    Q_generate = input("Czy chcesz wygenerować dane? (Y/n): ")
    Q_generate_plot = int(input("Ile trajektorii zapisać w postaci graficznej: "))
    Q_TAMSD = input("Czy chcesz przeliczyć TAMSD? (Y/n): ")
    Q_TAMSD_plot = int(input("Ile obliczonych TAMSD chcesz zapisać w postaci graficznej: "))
    print(64 * '-')
    T = T_long
    N = N_long
    path = 'data/part1/'
    traject_loaded = False
    if Q_generate == 'Y':
        print('Nastąpi generowanie danych')
        generate_trajectories(N, T, part)
        print(64 * '-')
    if Q_generate_plot > 0:
        if not traject_loaded:
            trajectories = read_trajectories(part)
            traject_loaded = True
        print(f'Tworzenie {Q_generate_plot} wykresów...')
        dirmake(path+'generated/traj_plot/')
        n = 0
        for traj in trajectories[:Q_generate_plot]:
            plot_traj(*traj, path+'generated/traj_plot/plot_1_'+str(n)+'.png', {'color': colors[n]}, {'figsize': (2,2)})
            n += 1
        print(64 * '-')
    if Q_TAMSD == 'Y':
        if not traject_loaded:
            trajectories = read_trajectories(part)
            traject_loaded = True
        TAMSD_estimation(trajectories, part)
        print(64 * '-')
    if Q_TAMSD_plot > 0:
        print('Ładowanie wartości z TAMSD')
        tamsd_data = read_TAMSD(part)
        print(f' --- ZAKOŃCZONO\nTworzenie {Q_TAMSD_plot} wykresów...')
        dirmake(path+'TAMSD/plot/')
        n = 0
        for i in range(Q_TAMSD_plot):
            plot_TAMSD(list(tamsd_data.loc[i]), path+'TAMSD/plot/tamsd'+str(i)+'.png', {'figsize': (4,3)}, ['a','b','c','d','e','f'][i])
            n += 1