import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Helvetica Neue"

BINS = 1000

muct = False

if not muct:
    low = 'errors/helen_120'
    medium = 'errors/helen_100'
    high = 'errors/helen_80'
else:
    low = 'errors/muct_65'
    medium = 'errors/muct_60'
    high = 'errors/muct_55'


with open('{}_.npx'.format(low), 'rb') as f:
    data_low = np.load(f)

with open('{}_.npx'.format(high), 'rb') as f:
    data_high = np.load(f)

with open('{}_.npx'.format(medium), 'rb') as f:
    data_medium = np.load(f)


def plot_hist(data):
    _, bins = np.histogram(data, bins=BINS)
    plt.hist(data, bins=bins)


def plot_cdf(data, color='black'):
    counts, bins = np.histogram(data, bins=BINS)
    cdf = np.cumsum(counts) / len(data)
    bins = np.insert(bins, 0, 0)
    cdf = np.insert(cdf, 0, 0)
    bins = np.insert(bins, -1, 0)
    cdf = np.insert(cdf, 0, 0)
    if muct:
        cdf = np.append(cdf[:990], 1)
        bins = np.append(bins[:990], 0.7)
        plt.plot(bins, cdf, color=color)
    else:
        plt.plot(bins[:200], cdf[:200], color=color)


plot_cdf(data_low, '#7570b3ff')
plot_cdf(data_medium, '#e7298aff')
plot_cdf(data_high, '#d95f02ff')
if muct:
    plt.legend(['65 parâmetros', '60 parâmetros', '55 parâmetros'])
else:
    plt.legend(['120 parâmetros', '100 parâmetros', '80 parâmetros'])

plt.grid(dashes=(2, 2))
plt.xlim(0, 0.6)
plt.xlabel('Erro médio')
plt.ylabel('Proporção')
if muct:
    plt.savefig('muct.pdf', bbox_inches='tight')
else:
    plt.savefig('helen.pdf', bbox_inches='tight')

plt.show()
