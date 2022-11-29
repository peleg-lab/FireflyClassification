import matplotlib.cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq


class DataExplorer:
    def __init__(self):
        self.data_root = "data"
        self.data_dir = "data/real_data"
        self.binary_data = pd.read_csv(self.data_root + '/real_data/binary_sequences_4.csv', header=None, sep='\n')
        self.metadata = pd.read_csv(self.data_root + '/real_data/flash_data_4.csv')
        self.params = pd.read_csv(self.data_root + '/params_4species.csv',
                                  names=['n_flashes', 'flash_length', 'ifi', 'species_ID'])
        self.colormap = {0: 'orange', 1: 'olivedrab', 2: 'mediumorchid', 3: 'dodgerblue'}

    def plot_representative_seqs(self, plot_ten=False, do_fourier=False):
        params = self.params.loc[self.params['n_flashes'] > 1]
        bd = self.binary_data.loc[self.params['n_flashes'] > 1]
        md = self.metadata.loc[self.params['n_flashes'] > 1]
        seq_0s = []
        seq_1s = []
        seq_2s = []
        seq_3s = []
        for i, seq in enumerate(bd.values):
            if md['species_label'].values[i] == 0:
                seq_0s.append(np.array(eval(seq[0])))
            elif md['species_label'].values[i] == 1:
                seq_1s.append(np.array(eval(seq[0])))
            elif md['species_label'].values[i] == 2:
                seq_2s.append(np.array(eval(seq[0])))
            else:
                seq_3s.append(np.array(eval(seq[0])))
        carolinus_js = [22, 38, 50, 53, 57, 58, 70, 72, 73, 100]
        knulli_js = [-17, 4, -9, 107, -3, -2, -1, 5, 330, 869]
        frontalis_js = [45, 145, 88, 107, 410, 661, 347, 306, 330, 869]
        bw_js = [17, 21, 22, 51, 32, 25, 26, 40, 28, 41]
        all_seqs = {'knulli': [seq_0s[knulli_js[0]], seq_0s[knulli_js[1]], seq_0s[knulli_js[2]], seq_0s[knulli_js[3]],
                               seq_0s[knulli_js[4]], seq_0s[knulli_js[5]], seq_0s[knulli_js[6]], seq_0s[knulli_js[7]],
                               seq_0s[knulli_js[8]], seq_0s[knulli_js[9]]],
                    'frontalis': [seq_1s[frontalis_js[0]], seq_1s[frontalis_js[1]], seq_1s[frontalis_js[2]],
                                  seq_1s[frontalis_js[3]],
                                  seq_1s[frontalis_js[4]], seq_1s[frontalis_js[5]], seq_1s[frontalis_js[6]],
                                  seq_1s[frontalis_js[7]],
                                  seq_1s[frontalis_js[8]], seq_1s[frontalis_js[9]]
                                  ],
                    'carolinus': [seq_2s[carolinus_js[0]], seq_2s[carolinus_js[1]], seq_2s[carolinus_js[2]],
                                  seq_2s[carolinus_js[3]],
                                  seq_2s[carolinus_js[4]], seq_2s[carolinus_js[5]], seq_2s[carolinus_js[6]],
                                  seq_2s[carolinus_js[7]],
                                  seq_2s[carolinus_js[8]], seq_2s[carolinus_js[9]]],
                    'bw': [seq_3s[bw_js[0]], seq_3s[bw_js[1]], seq_3s[bw_js[2]], seq_3s[bw_js[3]],
                           seq_3s[bw_js[4]], seq_3s[bw_js[5]], seq_3s[bw_js[6]], seq_3s[bw_js[7]],
                           seq_3s[bw_js[8]], seq_3s[bw_js[9]]]
                    }
        if plot_ten:
            self.plot_sp(seq_0s, knulli_js, color='orange', name='knulli')
            self.plot_sp(seq_1s, frontalis_js, color='olivedrab', name='frontalis')
            self.plot_sp(seq_2s, carolinus_js, color='mediumorchid', name='carolinus')
            self.plot_sp(seq_3s, bw_js, color='dodgerblue', name='bw')
        seq_0_s = []
        seq_1_s = []
        seq_2_s = []
        seq_3_s = []
        for i in seq_0s:
            seq_0_s.append(np.pad(i, (0, max(map(len, seq_0s)) - len(i))))
        for i in seq_1s:
            seq_1_s.append(np.pad(i, (0, max(map(len, seq_1s)) - len(i))))
        for i in seq_2s:
            seq_2_s.append(np.pad(i, (0, max(map(len, seq_2s)) - len(i))))
        for i in seq_3s:
            seq_3_s.append(np.pad(i, (0, max(map(len, seq_3s)) - len(i))))
        ref_seq_0 = np.sum(seq_0_s, axis=0) / len(seq_0_s)
        ref_seq_1 = np.sum(seq_1_s, axis=0) / len(seq_1_s)
        ref_seq_2 = np.sum(seq_2_s, axis=0) / len(seq_2_s)
        ref_seq_3 = np.sum(seq_3_s, axis=0) / len(seq_3_s)
        ref_seqs = (ref_seq_0, ref_seq_1, ref_seq_2, ref_seq_3)
        if do_fourier:
            fts = (fft(ref_seq_0), fft(ref_seq_1), fft(ref_seq_2), fft(ref_seq_3))
        colormap = self.colormap
        fig, ax = plt.subplots(4)

        for i, j in enumerate([3, 2, 0, 1]):
            ax[i].set_yscale('log')
            ax[i].spines['top'].set_visible(False)
            ax[i].spines['right'].set_visible(False)
            if not do_fourier:
                try:
                    xs = np.arange(0.0, len(ref_seqs[j]) / 30, (1 / 30))
                    ax[i].plot(xs, ref_seqs[j], color=colormap[j])
                    ax[i].set_xlim(0.0, 6.0)
                except ValueError:
                    xs = np.arange(0.0, (len(ref_seqs[j])) / 30, (1 / 30))[:-1]
                    ax[i].plot(xs, ref_seqs[j], color=colormap[j])
                    ax[i].set_xlim(0.0, 6.0)
            else:
                N = len(ref_seqs[j])
                # sample spacing
                T = 1.0 / 30.0
                ft = np.abs(fts[j][0:N // 2])
                xf = fftfreq(N, T)[:N // 2]
                ax[i].plot(xf, 2.0 / N * ft, color=colormap[j])
        if do_fourier:
            ax[3].set_xlabel('Frequency (Hz)')
        else:
            ax[3].set_xlabel('T[s]')
        plt.tight_layout()
        if do_fourier:
            stri = 'fourier'
        else:
            stri = 'log'
        plt.savefig('figs/representative_figs_{}.png'.format(stri))

    def plot_sp(self, seqs, js, color, name):
        fig, axes = plt.subplots(10, figsize=(6, 4))
        fig.subplots_adjust(hspace=-0.90)
        for i, ax in enumerate(axes):
            j = js[i]
            ax.cla()
            ax.bar(x=list([y[0] / 30 for y in enumerate(seqs[j])]), height=seqs[j] / 100, width=(1 / 30),
                   color=color)
            ax.set_ylim(0.0, 0.5)
            ax.set_xlim(0.0, 3.0)
            ax.axis('off')
        plt.savefig('figs/10_seqs_{}'.format(name))

    def plot_downsampled(self):
        params = self.params.loc[self.params['n_flashes'] > 1]
        bd = self.binary_data.loc[self.params['n_flashes'] > 1]
        md = self.metadata.loc[self.params['n_flashes'] > 1]

        value_counts = params['species_ID'].value_counts()
        value_counts_train = value_counts * 0.8
        value_counts_train = value_counts_train.astype(int)
        value_counts_valid = value_counts * 0.1
        value_counts_valid = value_counts_valid.astype(int)
        value_counts_test = value_counts * 0.1
        value_counts_test = value_counts_test.astype(int)
        value_counts_train_dict = {'B. wickershamorum': value_counts_train[3], 'P. carolinus': value_counts_train[2],
                                   'P. knulli': value_counts_train[0],
                                   'P. frontalis': value_counts_train[1]}
        value_counts_valid_dict = {'B. wickershamorum': value_counts_valid[3], 'P. carolinus': value_counts_valid[2],
                                   'P. knulli': value_counts_valid[0],
                                   'P. frontalis': value_counts_valid[1]}
        value_counts_test_dict = {'B. wickershamorum': value_counts_test[3], 'P. carolinus': value_counts_test[2],
                                  'P. knulli': value_counts_test[0], 'P. frontalis': value_counts_test[1]}

        downsample_value_counts_train = {'B. wickershamorum': 89, 'P. carolinus': 89, 'P. knulli': 89,
                                         'P. frontalis': 89}
        downsample_value_counts_valid = {'B. wickershamorum': value_counts_valid[3],
                                         'P. carolinus': value_counts_valid[3], 'P. knulli': value_counts_valid[3],
                                         'P. frontalis': value_counts_valid[3]}
        downsample_value_counts_test = {'B. wickershamorum': value_counts_test[3], 'P. carolinus': value_counts_test[3],
                                        'P. knulli': value_counts_test[3], 'P. frontalis': value_counts_test[3]}
        all_data = {
            'B. wickershamorum': {'pre': {'train': 89, 'valid': value_counts_valid[3], 'test': value_counts_test[3]},
                                  'post': {'train': 89, 'valid': value_counts_valid[3], 'test': value_counts_test[3]}
                                  },
            'P. carolinus': {'pre': {'train': value_counts_train[2],
                                     'valid': value_counts_valid[2],
                                     'test': value_counts_test[2]},
                             'post': {'train': 89, 'valid': value_counts_valid[3], 'test': value_counts_test[3]}
                             },
            'P. knulli': {'pre': {'train': value_counts_train[0],
                                  'valid': value_counts_valid[0],
                                  'test': value_counts_test[0]},
                          'post': {'train': 89, 'valid': value_counts_valid[3], 'test': value_counts_test[3]}
                          },
            'P. frontalis': {'pre': {'train': value_counts_train[1],
                                     'valid': value_counts_valid[1],
                                     'test': value_counts_test[1]},
                             'post': {'train': 89, 'valid': value_counts_valid[3], 'test': value_counts_test[3]}
                             }
            }
        all_df = pd.DataFrame(all_data)
        all_df.to_csv('data/pre_post_data.csv')
        fig, ax = plt.subplots(1, 2)
        font = {'family': 'Calibri',
                'size': 12}

        matplotlib.rc('font', **font)
        ax[1].set_yscale('log')
        ax[0].set_yscale('log')
        width = 0.8
        ax[0].bar(list(value_counts_train_dict.keys()), height=list(value_counts_train_dict.values()), width=width,
                  color=['dodgerblue', 'mediumorchid', 'orange', 'olivedrab'])
        ax[0].bar(list(value_counts_valid_dict.keys()), height=list(value_counts_valid_dict.values()), width=width,
                  bottom=list(value_counts_train_dict.values()),
                  color=['lightskyblue', 'plum', 'gold', 'yellowgreen'])
        ax[0].bar(list(value_counts_test_dict.keys()), height=list(value_counts_test_dict.values()), width=width,
                  bottom=np.array(list(value_counts_valid_dict.values())) + np.array(
                      list(value_counts_train_dict.values())),
                  color=['powderblue', 'palevioletred', 'khaki', 'palegreen'])
        ax[1].bar(list(downsample_value_counts_train.keys()), height=list(downsample_value_counts_train.values()),
                  width=width,
                  color=['dodgerblue', 'mediumorchid', 'orange', 'olivedrab'])
        ax[1].bar(list(downsample_value_counts_valid.keys()), height=list(downsample_value_counts_valid.values()),
                  width=width,
                  bottom=list(downsample_value_counts_train.values()),
                  color=['lightskyblue', 'plum', 'gold', 'yellowgreen'])
        ax[1].bar(list(downsample_value_counts_test.keys()), height=list(downsample_value_counts_test.values()),
                  width=width,
                  bottom=np.array(list(downsample_value_counts_valid.values())) + np.array(
                      list(downsample_value_counts_train.values())),
                  color=['powderblue', 'palevioletred', 'khaki', 'palegreen'])
        ax[0].set_xlabel('Sp. label')
        ax[1].set_xlabel('Sp. label')
        ax[0].set_ylabel('Count')
        ax[0].set_ylim(1, )
        ax[1].set_ylim(1, 10000)

        plt.tight_layout()
        plt.savefig('figs/downsampling.png')

    @staticmethod
    def get_bcs(x):
        bin_centers = 0.5 * (x[1:] + x[:-1])
        return bin_centers

    def lineplot(self, a, b, c, d):
        ns = []
        n1s = []
        n2s = []
        xs = []
        x1s = []
        x2s = []
        for arr in (c, d, a, b):
            n, x = np.histogram(arr['n_flashes'], density=True, bins=np.arange(0.0, 30.0, 1))
            n1, x1 = np.histogram(arr['ifi'], density=True, bins=np.arange(0.1, 2.0, 0.03))
            n2, x2 = np.histogram(arr['flash_length'], density=True, bins=np.arange(0.0, 0.4, 0.025))
            ns.append(n)
            n1s.append(n1)
            n2s.append(n2)
            xs.append(x)
            x1s.append(x1)
            x2s.append(x2)
        fig, axes = plt.subplots(1, 3)

        for i, (_n, _x) in enumerate(list(zip(ns, xs))):
            bin_centers = self.get_bcs(_x)
            axes[0].plot(bin_centers, _n, color=self.colormap[i])
        for i, (_n1, _x1) in enumerate(list(zip(n1s, x1s))):
            bin_centers = self.get_bcs(_x1)
            axes[1].plot(bin_centers, _n1, color=self.colormap[i])
        for i, (_n2, _x2) in enumerate(list(zip(n2s, x2s))):
            bin_centers = self.get_bcs(_x2)
            axes[2].plot(bin_centers, _n2, color=self.colormap[i])
        axes[0].spines['top'].set_visible(False)
        axes[0].spines['right'].set_visible(False)
        axes[1].spines['top'].set_visible(False)
        axes[1].spines['right'].set_visible(False)
        axes[2].spines['top'].set_visible(False)
        axes[2].spines['right'].set_visible(False)
        axes[0].set_xlabel('n_flashes')
        axes[1].set_xlabel('ifi')
        axes[2].set_xlabel('flash_length')
        axes[0].set_ylabel('pdf')
        axes[0][0].set_yscale('log')
        axes[0][1].set_yscale('log')
        axes[0][2].set_yscale('log')
        axes[0][3].set_yscale('log')

        return axes

    def plot_distributions(self):
        font = {'family': 'Calibri',
                'size': 12}

        matplotlib.rc('font', **font)
        params = self.params.loc[self.params['n_flashes'] > 1]
        bd = self.binary_data.loc[self.params['n_flashes'] > 1]
        md = self.metadata.loc[self.params['n_flashes'] > 1]

        zeros = params.loc[params['species_ID'] == 0]
        ones = params.loc[params['species_ID'] == 1]
        twos = params.loc[params['species_ID'] == 2]
        threes = params.loc[params['species_ID'] == 3]

        axes = self.lineplot(threes, twos, zeros, ones)
        plt.savefig('figs/distributions.png')


do_fourier = False
plot_ten = True
de = DataExplorer()
de.plot_downsampled()
de.plot_representative_seqs(plot_ten, do_fourier)
de.plot_distributions()
print('done')
