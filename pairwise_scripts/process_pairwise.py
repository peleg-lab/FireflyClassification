import json
import random
import os
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd
import seaborn as sns


def plot_each(rdict):
    colormap = {'bw': 'dodgerblue', 'ic': 'mediumorchid', 'uf': 'olivedrab', 'ik': 'orange', 'io': 'midnightblue'}
    linestylemap = {'bw': {'ic': 'solid',
                           'uf': 'dotted',
                           'ik': 'dashed',
                           'io': 'dashdot'},
                    'ic': {'bw': 'solid',
                           'uf': 'dotted',
                           'ik': 'dashed',
                           'io': 'dashdot'},
                    'uf': {'bw': 'solid',
                           'ic': 'dotted',
                           'ik': 'dashed',
                           'io': 'dashdot'},
                    'ik': {'bw': 'solid',
                           'ic': 'dotted',
                           'uf': 'dashed',
                           'io': 'dashdot'},
                    'io': {'bw': 'solid',
                           'ic': 'dotted',
                           'uf': 'dashed',
                           'ik': 'dashdot'}
                    }

    species_pairs_seen = []
    species_vals = {}

    for k in rdict.keys():
        substrings = k[int(len(k) / 2):], k[:int(len(k) / 2)]
        print(substrings)

        for i, substring in enumerate(substrings):
            x_val = float(substring[3:])
            species_val = substring[:2]
            if i == 0:
                other_species_val = substrings[1][:2]
            else:
                other_species_val = substrings[0][:2]
            y_val = min(1, (rdict[k][species_val] / rdict[k]['iterations']))
            if species_vals.get(species_val):
                if species_vals[species_val].get(other_species_val):
                    species_vals[species_val][other_species_val].append((x_val, y_val))
                else:
                    species_vals[species_val][other_species_val] = [(x_val, y_val)]

            else:
                species_vals[species_val] = {
                    other_species_val: [(x_val, y_val)]
                }

    for sp in species_vals.keys():
        fig, ax = plt.subplots()
        for osp in species_vals[sp].keys():
            xs = [x[0] for x in species_vals[sp][osp]]
            ys = [x[1] for x in species_vals[sp][osp]]
            if (sp, osp) in species_pairs_seen:
                ax.plot(xs, ys, c=colormap[sp], linestyle=linestylemap[sp][osp])
            else:
                ax.plot(xs, ys, c=colormap[sp], linestyle=linestylemap[sp][osp], label='{} with {}'.format(sp, osp))
            species_pairs_seen.append((sp, osp))

        ax.set_xlabel('% species')
        ax.set_ylabel('Determination rate')
        handles, labels = ax.get_legend_handles_labels()
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
        ax.legend(handles, labels, ncol=2, loc='lower right')
        ax.set_ylim(0.0, 1.0)
        plt.savefig('figs/all_sweeps_scores_{}_line.svg'.format(sp), dpi=800)


def plot_pairwise(rdict, sp):
    data = pd.read_csv('data/real_data/flash_sequence_data.csv')
    for species_y in ['io', 'uf', 'ic', 'bw', 'ik']:
        if sp != species_y:
            fig, ax = plt.subplots()
            indices = []
            for k in rdict.keys():
                if '{}0.05'.format(sp) in k and species_y in k:
                    indices.extend(rdict[k]['indices'][0])
            subset = data.iloc[indices]
            subset = subset[~subset.index.duplicated()]
            sns.histplot(data=subset, x='flash_duration', hue='species_label', ax=ax, common_norm=False, common_bins=True,
                         stat='probability')
            plt.savefig('{}0.05_w_{}_duration.png'.format(sp, species_y))
            ax.clear()
            sns.histplot(data=subset, x='ifi', hue='species_label', ax=ax, common_norm=False,
                         common_bins=True,
                         stat='probability')
            plt.savefig('{}0.05_w_{}_ifi.png'.format(sp, species_y))
            ax.clear()
            sns.histplot(data=subset, x='num_flashes', hue='species_label', ax=ax, common_norm=False,
                         common_bins=True,
                         stat='probability')
            plt.savefig('{}0.05_w_{}_num_flashes.svg'.format(sp, species_y), dpi=800)


def plot_sweeps(rdict):
    colormap = {'bw': 'dodgerblue', 'ic': 'mediumorchid', 'uf': 'olivedrab', 'ik': 'orange', 'io': 'midnightblue'}
    fig, ax = plt.subplots()
    markermap = {'bw': '.', 'ic': '<', 'uf': '>', 'ik': 's', 'io': 'x'}
    species_pairs_seen = []
    for k in rdict.keys():
        substrings = k[int(len(k) / 2):], k[:int(len(k) / 2)]
        print(substrings)
        for i, substring in enumerate(substrings):
            x_val = float(substring[3:])
            species_val = substring[:2]
            if i == 0:
                other_species_val = substrings[1][:2]
            else:
                other_species_val = substrings[0][:2]
            y_val = min(1, (rdict[k][species_val] / rdict[k]['iterations']))
            if (species_val, other_species_val) in species_pairs_seen:
                ax.scatter(x_val, y_val, c=colormap[species_val], marker=markermap[other_species_val])
            else:
                ax.scatter(x_val, y_val, c=colormap[species_val], marker=markermap[other_species_val],
                           label='{} and {}'.format(species_val, other_species_val))
            species_pairs_seen.append((species_val, other_species_val))
    handles, labels = ax.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    ax.legend(handles, labels, ncol=2, loc='lower right')
    plt.tight_layout()
    ax.set_xlabel('% species')
    ax.set_ylabel('Determination rate')
    plt.savefig('figs/all_sweeps_scores.png')


def plot_all(rdict):
    colormap = {'bw': 'dodgerblue', 'ic': 'mediumorchid', 'uf': 'olivedrab', 'ik': 'orange', 'io': 'midnightblue'}
    fig, ax = plt.subplots()
    linestylemap = {'bw': {'ic': 'solid',
                           'uf': 'dotted',
                           'ik': 'dashed',
                           'io': 'dashdot'},
                    'ic': {'bw': 'solid',
                           'uf': 'dotted',
                           'ik': 'dashed',
                           'io': 'dashdot'},
                    'uf': {'bw': 'solid',
                           'ic': 'dotted',
                           'ik': 'dashed',
                           'io': 'dashdot'},
                    'ik': {'bw': 'solid',
                           'ic': 'dotted',
                           'uf': 'dashed',
                           'io': 'dashdot'},
                    'io': {'bw': 'solid',
                           'ic': 'dotted',
                           'uf': 'dashed',
                           'ik': 'dashdot'}
                    }

    species_pairs_seen = []
    species_vals = {}

    for k in rdict.keys():
        substrings = k[int(len(k) / 2):], k[:int(len(k) / 2)]
        print(substrings)

        for i, substring in enumerate(substrings):
            x_val = float(substring[3:])
            species_val = substring[:2]
            if i == 0:
                other_species_val = substrings[1][:2]
            else:
                other_species_val = substrings[0][:2]
            y_val = min(1, (rdict[k][species_val] / rdict[k]['iterations']))
            if species_vals.get(species_val):
                if species_vals[species_val].get(other_species_val):
                    species_vals[species_val][other_species_val].append((x_val, y_val))
                else:
                    species_vals[species_val][other_species_val] = [(x_val, y_val)]

            else:
                species_vals[species_val] = {other_species_val: [(x_val, y_val)]
                                             }

    for sp in species_vals.keys():
        for osp in species_vals[sp].keys():
            xs = [x[0] for x in species_vals[sp][osp]]
            ys = [x[1] for x in species_vals[sp][osp]]
            if (sp, osp) in species_pairs_seen:

                ax.plot(xs, ys, c=colormap[sp], linestyle=linestylemap[sp][osp])
            else:
                ax.plot(xs, ys, c=colormap[sp], linestyle=linestylemap[sp][osp], label='{} with {}'.format(sp, osp))
            species_pairs_seen.append((sp, osp))

    ax.set_xlabel('% species')
    ax.set_ylabel('Determination rate')
    ax.set_ylim(0.0, 1.0)
    handles, labels = ax.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    ax.legend(handles, labels, ncol=2)
    plt.savefig('figs/all_sweeps_scores_plus_line.svg', dpi=800)


def plot_different(rdict):
    fig,ax = plt.subplots()
    total_scores = {}
    errors = {}
    total_at_each = {}
    for k in rdict.keys():
        substrings = k[int(len(k) / 2):], k[:int(len(k) / 2)]
        for i, substring in enumerate(substrings):
            x_val = float(substring[3:])
            species_val = substring[:2]
            if total_at_each.get(x_val):
                total_at_each[x_val].append(rdict[k][species_val] / rdict[k]['iterations'])
            else:
                total_at_each[x_val] = [(rdict[k][species_val] / rdict[k]['iterations'])]
    for k in rdict.keys():
        substrings = k[int(len(k) / 2):], k[:int(len(k) / 2)]
        print(substrings)
        for i, substring in enumerate(substrings):
            x_val = float(substring[3:])
            species_val = substring[:2]
            if total_scores.get(x_val):
                total_scores[x_val] += rdict[k][species_val] / rdict[k]['iterations']
            else:
                total_scores[x_val] = rdict[k][species_val] / rdict[k]['iterations']
            if not errors.get(x_val):
                errors[x_val] = stats.sem(total_at_each[x_val])
    error_s = [(a, b) for a, b in errors.items()]
    sorted_error_s = sorted(error_s, key=lambda x: x[0])
    eys = [y for x, y in sorted_error_s]

    pairs = [(a, (5*b) / 100) for a,b in total_scores.items()]
    sorted_pairs = sorted(pairs, key=lambda x: x[0])
    xs = [x for x,y in sorted_pairs]
    ys = [y for x, y in sorted_pairs]
    ax.errorbar(xs, ys, fmt='o', yerr=eys, capsize=1, mfc='white', zorder=1,
                color='darkblue')
    ax.plot(xs, ys, color='darkblue', alpha=0.5)
    plt.tight_layout()
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel('% species')
    ax.set_ylabel('Determination rate')
    plt.savefig('figs/aggregate_500.svg', dpi=800)


with open('results_dict_500.json', 'r') as json_file:
    results_dict = json.load(json_file)

plot_sweeps(results_dict)
plot_different(results_dict)
plot_each(results_dict)
plot_all(results_dict)
for sp in ['ik', 'bw', 'io', 'ic', 'uf']:
    plot_pairwise(results_dict, sp)
