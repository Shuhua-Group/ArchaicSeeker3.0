import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
matplotlib.use("Agg")

import seaborn as sns
sns.set_style("darkgrid")

from sklearn import metrics


sstar_1src_accuracy = pd.read_csv(snakemake.input.sstar_1src_accuracy, sep="\t").dropna()
sprime_1src_accuracy = pd.read_csv(snakemake.input.sprime_1src_accuracy, sep="\t").dropna()
skovhmm_1src_accuracy = pd.read_csv(snakemake.input.skovhmm_1src_accuracy, sep="\t").dropna()

sstar_1src_accuracy_grouped = sstar_1src_accuracy.groupby(['demography', 'scenario', 'sample', 'cutoff'], as_index=False)
sprime_1src_accuracy_grouped = sprime_1src_accuracy.groupby(['demography', 'sample', 'cutoff'], as_index=False)
skovhmm_1src_accuracy_grouped = skovhmm_1src_accuracy.groupby(['demography', 'sample', 'cutoff'], as_index=False)

sstar_1src_accuracy_mean = sstar_1src_accuracy_grouped.mean()
sprime_1src_accuracy_mean = sprime_1src_accuracy_grouped.mean()
skovhmm_1src_accuracy_mean = skovhmm_1src_accuracy_grouped.mean()

sstar_1src_accuracy_mean.to_csv(snakemake.output.sstar_1src_accuracy_mean, sep="\t", index=False)
sprime_1src_accuracy_mean.to_csv(snakemake.output.sprime_1src_accuracy_mean, sep="\t", index=False)
skovhmm_1src_accuracy_mean.to_csv(snakemake.output.skovhmm_1src_accuracy_mean, sep="\t", index=False)

sprime_1src_accuracy_mean['scenario'] = ['true'] * len(sprime_1src_accuracy_mean)
skovhmm_1src_accuracy_mean['scenario'] = ['true'] * len(skovhmm_1src_accuracy_mean)

sstar_2src_accuracy = pd.read_csv(snakemake.input.sstar_2src_accuracy, sep="\t").dropna()
sprime_2src_accuracy = pd.read_csv(snakemake.input.sprime_2src_accuracy, sep="\t").dropna()
archaicseeker2_2src_accuracy = pd.read_csv(snakemake.input.archaicseeker2_2src_accuracy, sep="\t").dropna()

sstar_2src_accuracy_grouped = sstar_2src_accuracy.groupby(['demography', 'sample', 'cutoff', 'src'], as_index=False)
sprime_2src_accuracy_grouped = sprime_2src_accuracy.groupby(['demography', 'sample', 'cutoff', 'src'], as_index=False)
archaicseeker2_2src_accuracy_grouped = archaicseeker2_2src_accuracy.groupby(['demography', 'sample', 'cutoff', 'src'], as_index=False)

sstar_2src_accuracy_mean = sstar_2src_accuracy_grouped.mean()
sprime_2src_accuracy_mean = sprime_2src_accuracy_grouped.mean()
archaicseeker2_2src_accuracy_mean = archaicseeker2_2src_accuracy_grouped.mean()

sstar_2src_accuracy_mean.to_csv(snakemake.output.sstar_2src_accuracy_mean, sep="\t", index=False)
sprime_2src_accuracy_mean.to_csv(snakemake.output.sprime_2src_accuracy_mean, sep="\t", index=False)
archaicseeker2_2src_accuracy_mean.to_csv(snakemake.output.archaicseeker2_2src_accuracy_mean, sep="\t", index=False)

methods1 = ['sstar', 'sprime', 'skovhmm']
demography1 = ['HumanNeanderthal', 'BonoboGhost']
samples = ['nref_10_ntgt_1', 'nref_50_ntgt_1']
scenarios = ['true', 'const', 'ref_tgt_only']
accuracy1 = {
    'sstar': sstar_1src_accuracy_mean,
    'sprime': sprime_1src_accuracy_mean,
    'skovhmm': skovhmm_1src_accuracy_mean,
}
methods2 = [
    'sstar', 
    'sprime', 
    'archaicseeker2'
]
demography2 = ['HumanNeanderthalDenisovan', 'ChimpBonoboGhost']
accuracy2 = {
    'sstar': sstar_2src_accuracy_mean,
    'sprime': sprime_2src_accuracy_mean,
    'archaicseeker2': archaicseeker2_2src_accuracy_mean,
}

fig, axs = plt.subplots(nrows=2, ncols=3, constrained_layout=True, figsize=(7.5,4), dpi=350)
gridspec = axs[0, 0].get_subplotspec().get_gridspec()
for a in axs[:,2]:
    a.remove()

markers = {
    'nref_10_ntgt_1': {'symbol':'.', 'size': 6},
    'nref_50_ntgt_1': {'symbol':'*', 'size': 6},
}

colors = {
    'sstar': {'true': 'blue', 'const': 'cyan', 'ref_tgt_only': 'purple'},
    'skovhmm': 'green',
    'sprime': 'orange',
    'archaicseeker2': 'magenta',
}

linestyles = {
    'const': 'dotted',
    'true': 'solid',
    'ref_tgt_only': (0, (3, 1, 1, 1, 1, 1)),
}

titles = {
    'HumanNeanderthal': 'Human-Neanderthal model',
    'BonoboGhost': 'Bonobo-Ghost model',
    'HumanNeanderthalDenisovan': 'Human-Neanderthal-Denisovan model',
    'ChimpBonoboGhost': 'Chimpanzee-Ghost-Bonobo model',
}

zorders = {
    'sstar': 2,
    'skovhmm': 5,
    'sprime': 10,
}

j = 0
for d in demography1:
    for s in samples:
        for sc in scenarios:
            for m in methods1:
                if m == 'sstar': color = colors[m][sc]
                else: color = colors[m]
                df = accuracy1[m][
                        (accuracy1[m]['demography'] == d) &
                        (accuracy1[m]['sample'] == s) &
                        (accuracy1[m]['scenario'] == sc)
                    ].sort_values(by='recall', ascending=False)
                recall = df['recall']
                precision = df['precision']
                if (m == 'sprime') or (m == 'skovhmm'):
                    if sc != 'true': continue

                if d == 'BonoboGhost':
                    axs[0,j].plot(recall, precision,
                        marker=markers[s]['symbol'], ms=markers[s]['size'],
                        c=color, zorder=zorders[m])
                else:
                    axs[0,j].plot(recall, precision,
                        marker=markers[s]['symbol'], ms=markers[s]['size'],
                        c=color)

    axs[0,j].set_xlabel('Recall (%)', fontsize=10)
    axs[0,j].set_ylabel('Precision (%)', fontsize=10)
    axs[0,j].set_xlim([-5, 105])
    axs[0,j].set_ylim([-5, 105])
    axs[0,j].set_title(titles[d], fontsize=8, weight='bold')
    if j == 0: 
        axs[0,j].text(-35, 110, 'B', fontsize=10, weight='bold')
        axs[0,j].plot([0,100],[2.25,2.25], c='red', alpha=0.5)
    if j == 1: 
        axs[0,j].text(-35, 110, 'C', fontsize=10, weight='bold')
        axs[0,j].plot([0,100],[2,2], c='red', alpha=0.5)

    f_scores = np.linspace(20, 80, num=4)
    lines, labels = [], []
    for f_score in f_scores:
        x = np.linspace(1, 100)
        y = f_score * x / (2 * x - f_score)
        (l,) = axs[0,j].plot(x[y >= 0], y[y >= 0], color="black", alpha=0.4, linestyle='dotted', zorder=1)
        axs[0,j].annotate("F1={0:0.0f}%".format(f_score), xy=(101, y[45] + 2), fontsize=8)

    j += 1

j = 0
for d in demography2:
    for s in samples:
        for m in methods2:
            if m == 'sstar': color = colors[m]['true']
            else: color = colors[m]
            src1_df = accuracy2[m][
                         (accuracy2[m]['demography'] == d) &
                         (accuracy2[m]['sample'] == s) &
                         (accuracy2[m]['src'] == 'src1')
                     ].sort_values(by='recall', ascending=False)
            src2_df = accuracy2[m][
                         (accuracy2[m]['demography'] == d) &
                         (accuracy2[m]['sample'] == s) &
                         (accuracy2[m]['src'] == 'src2')
                     ].sort_values(by='recall', ascending=False)
            src1_recall = src1_df['recall']
            src1_precision = src1_df['precision']
            src2_recall = src2_df['recall']
            src2_precision = src2_df['precision']
            axs[1,j].plot(src1_recall, src1_precision,
                     marker=markers[s]['symbol'], ms=markers[s]['size'],
                     c=color, markerfacecolor='white')
            axs[1,j].plot(src2_recall, src2_precision,
                     marker=markers[s]['symbol'], ms=markers[s]['size'],
                     c=color, linestyle='dashdot')

    axs[1,j].set_xlabel('Recall (%)', fontsize=10)
    axs[1,j].set_ylabel('Precision (%)', fontsize=10)
    axs[1,j].set_xlim([-5, 105])
    axs[1,j].set_ylim([-5, 105])
    axs[1,j].set_title(titles[d], fontsize=8, weight='bold')
    if j == 0: 
        axs[1,j].text(-35, 110, 'D', fontsize=10, weight='bold')
        axs[1,j].plot([0,100],[0.2,0.2], c='red', alpha=0.5)
        axs[1,j].plot([0,100],[4,4], c='red', linestyle='dotted')
    if j == 1: 
        axs[1,j].text(-35, 110, 'E', fontsize=10, weight='bold')
        axs[1,j].plot([0,100],[2,2], c='red', alpha=0.5)
        axs[1,j].plot([0,100],[2,2], c='red', linestyle='dotted')

    f_scores = np.linspace(20, 80, num=4)
    lines, labels = [], []
    for f_score in f_scores:
        x = np.linspace(1, 100)
        y = f_score * x / (2 * x - f_score)
        (l,) = axs[1,j].plot(x[y >= 0], y[y >= 0], color="black", alpha=0.4, linestyle='dotted', zorder=1)
        axs[1,j].annotate("F1={0:0.0f}%".format(f_score), xy=(101, y[45] + 2), fontsize=8)

    j += 1

# legend
subfig = fig.add_subfigure(gridspec[:,2])
handles, labels = subfig.gca().get_legend_handles_labels()
sstar_line = plt.Line2D([0], [0], label='sstar (full)', color=colors['sstar']['true'])
sstar_line2 = plt.Line2D([0], [0], label='sstar (constant)', color=colors['sstar']['const'])
sstar_line3 = plt.Line2D([0], [0], label='sstar (only ref & tgt)', color=colors['sstar']['ref_tgt_only'])
skovhmm_line = plt.Line2D([0], [0], label='SkovHMM', color=colors['skovhmm'])
sprime_line = plt.Line2D([0], [0], label='SPrime', color=colors['sprime'])
archaicseeker2_line = plt.Line2D([0], [0], label='ArchaicSeeker2.0', color=colors['archaicseeker2'])
baseline1 = plt.Line2D([0], [0], label='baseline/src1 baseline', color='red', alpha=0.5)
baseline2 = plt.Line2D([0], [0], label='src2 baseline', color='red', linestyle='dotted')
f1_curves = plt.Line2D([0], [0], label='iso-F1 curves', color='black', alpha=0.4, linestyle='dotted')
nref_10_ntgt_1 = plt.Line2D([0], [0], marker=markers['nref_10_ntgt_1']['symbol'],
                            ms=5, label='Nref = 10', color='black', linewidth=0)
nref_50_ntgt_1 = plt.Line2D([0], [0], marker=markers['nref_50_ntgt_1']['symbol'],
                            ms=5, label='Nref = 50', color='black', linewidth=0)
src1 = plt.Line2D([0], [0], label='src1', color='black', marker='o', ms=4, markerfacecolor='white')
src2 = plt.Line2D([0], [0], label='src2', color='black', marker='o', ms=4, linestyle='dotted')

handles.extend([sstar_line, sstar_line2, sstar_line3, skovhmm_line, sprime_line, archaicseeker2_line,
                baseline1, baseline2, f1_curves, nref_10_ntgt_1, nref_50_ntgt_1, src1, src2])
subfig.legend(handles=handles, fontsize=8, handlelength=1.5)

fig.set_constrained_layout_pads(w_pad=4 / 72, h_pad=4 / 72, hspace=0, wspace=0.1)
plt.savefig(snakemake.output.accuracy, bbox_inches='tight')
