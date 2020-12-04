import matplotlib.pyplot as plt

def plot_traj(x, y, path, opts_plot = {}, opts_fig = {}):
    plt.figure(**opts_fig)
    plt.cla()
    plt.plot(x,y, **opts_plot)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(path, transparent = True, bbox_inches = 'tight', dpi = 300)

def plot_TAMSD(data, path, opts_fig = {}, title = ''):
    _, D, expo, expo_est, tamsds = data
    D = float(D)
    expo = float(expo)
    expo_est = float(expo_est)
    tamsds = eval(tamsds)
    t = range(len(tamsds))
    plt.cla()
    plt.figure(**opts_fig)
    plt.loglog(t, tamsds, '.', label = 'punkty TAMSD')
    plt.loglog(t, [4 * D * i ** expo_est for i in t], 'b', label = r'Wyestymowana $\alpha$')
    plt.loglog(t, [4 * D * i ** expo for i in t], 'r', label = r'Prawdziwa $\alpha$')
    plt.xlabel('t')
    plt.ylabel(r'$\rho(t)$')
    plt.legend()
    plt.title(title, loc = 'left')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.savefig(path, transparent = True, bbox_inches = 'tight', dpi = 300)