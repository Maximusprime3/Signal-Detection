import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import math
import seaborn as sns
import os

from scipy.stats import ttest_1samp, wilcoxon, ttest_ind, mannwhitneyu, levene, shapiro, bartlett, ks_2samp


import scipy.stats as st

#from statsmodels.stats.multicomp import (pairwise_tukeyhsd,MultiComparison)


def compare_three_absErrordist(data_1, data_2, data_3, name_1, name_2, name_3, bins=40):
    ones = np.ones(len(data_1))
    twos = ones * 2
    ot = np.append(ones, twos)

    fig, axs = plt.subplots(1, 2)
    fig.suptitle('Comparison of ' + name_1 + ' and ' + name_2)

    axs[1].axvline(np.mean(data_1), linestyle='--', color='royalblue', label='_no_legend_')
    axs[1].axvline(np.mean(data_2), linestyle='--', color='chocolate', label='_no_legend_')
    axs[1].axvline(np.mean(data_3), linestyle='--', color='green', label='_no_legend_')

    bins = np.linspace(0, max(max(data_1), max(data_2), max(data_2)), bins)
    sns.set_style(None)
    sns.distplot(data_1, bins=bins, label=name_1)
    sns.distplot(data_2, bins=bins, label=name_2)
    sns.distplot(data_3, bins=bins, label=name_3, color='green')

    axs[1].set_xlim([0, 11])
    # axs[1].set_ylabel('Probability')
    axs[1].set_xlabel('Absolute Error')
    axs[1].legend()

    axs[0].set_ylim([0, 11])

    bp = axs[0].boxplot([data_1, data_2, data_3], 0, '', positions=[1, 2, 3], widths=0.25, patch_artist=True)
    axs[0].set_xticklabels(['0.7', '1.25', '2.4'])
    bp['boxes'][0].set(color='royalblue', alpha=0.25)
    bp['boxes'][1].set(color='orange', alpha=0.5)
    bp['boxes'][2].set(color='green', alpha=0.5)
    axs[0].set_ylabel('Absolute Error')
    axs[0].set_xlabel('Phase')
    print(name_1 + ' mean: ' + str(round(np.mean(data_1), 3)) + ', ste: ' + str(
        round(np.std(data_1) / ((len(data_1)) ** 0.5), 3)))
    print(name_2 + ' mean: ' + str(round(np.mean(data_2), 3)) + ', ste: ' + str(
        round(np.std(data_2) / ((len(data_2)) ** 0.5), 3)))
    print(name_2 + ' mean: ' + str(round(np.mean(data_3), 3)) + ', ste: ' + str(
        round(np.std(data_3) / ((len(data_3)) ** 0.5), 3)))
    print('------------------------------------------')
    print(name_1, name_2)
    anova_f, anova_p = st.f_oneway(data_1, data_2)
    print('Anova p-value: ', anova_p)
    kruskal_stat, kruskal_p = st.kruskal(data_1, data_2)
    print('Kruskal p-value: ', kruskal_p)
    ks_2samp_D, ks_2samp_p = ks_2samp(data_1, data_2)
    print('Kolmogorov-Smirnov -value: ', ks_2samp_p)
    print('------------------------------------------')
    print(name_1, name_3)
    anova_f, anova_p = st.f_oneway(data_1, data_3)
    print('Anova p-value: ', anova_p)
    kruskal_stat, kruskal_p = st.kruskal(data_1, data_3)
    print('Kruskal p-value: ', kruskal_p)
    ks_2samp_D, ks_2samp_p = ks_2samp(data_1, data_3)
    print('Kolmogorov-Smirnov -value: ', ks_2samp_p)
    print('------------------------------------------')
    print(name_2, name_3)
    anova_f, anova_p = st.f_oneway(data_2, data_3)
    print('Anova p-value: ', anova_p)
    kruskal_stat, kruskal_p = st.kruskal(data_2, data_3)
    print('Kruskal p-value: ', kruskal_p)
    ks_2samp_D, ks_2samp_p = ks_2samp(data_2, data_3)
    print('Kolmogorov-Smirnov -value: ', ks_2samp_p)

    # plt.savefig(r'C:\Users\Max\Documents\BA\Git\BA\Statistics\Results\\BigQ\heavyLight/KNS_m123.pdf',bbox_inches='tight')
    plt.show()


def compare_two_absErrordist(data_1, data_2, name_1, name_2, bins=40, onesided=False):
    ones = np.ones(len(data_1))
    twos = ones * 2
    ot = np.append(ones, twos)

    fig, axs = plt.subplots(1, 2)
    fig.suptitle('Comparison of ' + name_1 + ' and ' + name_2)

    axs[1].axvline(np.mean(data_1), linestyle='--', color='royalblue', label='_no_legend_')
    axs[1].axvline(np.mean(data_2), linestyle='--', color='chocolate', label='_no_legend_')

    bins = np.linspace(0, max(max(data_1), max(data_2)), bins)
    sns.set_style(None)
    sns.distplot(data_1, bins=bins, label=name_1)
    sns.distplot(data_2, bins=bins, label=name_2)
    axs[1].set_xlim([0, 15])
    # axs[1].set_ylabel('Probability')
    axs[1].set_xlabel('Absolute Error')
    axs[1].legend()

    axs[0].set_ylim([0, 15])

    bp = axs[0].boxplot([data_1, data_2], 0, '', positions=[1, 2], widths=0.25, patch_artist=True)

    bp['boxes'][0].set(color='royalblue', alpha=0.25)
    bp['boxes'][1].set(color='orange', alpha=0.5)
    axs[0].set_ylabel('Absolute Error')
    axs[0].set_xlabel('Phase')
    print(name_1 + ' mean: ' + str(round(np.mean(data_1), 3)) + ', ste: ' + str(
        round(np.std(data_1) / ((len(data_1)) ** 0.5), 3)))
    print(name_2 + ' mean: ' + str(round(np.mean(data_2), 3)) + ', ste: ' + str(
        round(np.std(data_2) / ((len(data_2)) ** 0.5), 3)))
    anova_f, anova_p = st.f_oneway(data_1, data_2)
    print('Anova p-value: ', anova_p)
    kruskal_stat, kruskal_p = st.kruskal(data_1, data_2)
    print('Kruskal p-value: ', kruskal_p)

    ks_2samp_D, ks_2samp_p = ks_2samp(data_1, data_2)
    print('Kolmogorov-Smirnov -value: ', ks_2samp_p)
    # plt.savefig(r'C:\Users\Max\Documents\BA\Git\BA\Statistics\Results\\BigQ/'+name_1+'_'+name_2+'.pdf',bbox_inches='tight')
    plt.show()



def is_a_less_b(data_1, data_2, name_1, name_2, bins=40):
    ones = np.ones(len(data_1))
    twos = ones * 2
    ot = np.append(ones, twos)

    fig, axs = plt.subplots(1, 2)
    fig.suptitle('Comparison of ' + name_1 + ' and ' + name_2)

    axs[1].axvline(np.mean(data_1), linestyle='--', color='royalblue', label='_no_legend_')
    axs[1].axvline(np.mean(data_2), linestyle='--', color='chocolate', label='_no_legend_')

    bins = np.linspace(0, max(max(data_1), max(data_2)), bins)
    sns.set_style(None)
    sns.distplot(data_1, bins=bins, label=name_1)
    sns.distplot(data_2, bins=bins, label=name_2)
    axs[1].set_xlim([0, 15])
    # axs[1].set_ylabel('Probability')
    axs[1].set_xlabel('Absolute Error')
    axs[1].legend()

    axs[0].set_ylim([0, 15])

    bp = axs[0].boxplot([data_1, data_2], 0, '', positions=[1, 2], widths=0.25, patch_artist=True)

    bp['boxes'][0].set(color='royalblue', alpha=0.25)
    bp['boxes'][1].set(color='orange', alpha=0.5)
    axs[0].set_ylabel('Absolute Error')
    axs[0].set_xlabel('Phase')
    print(name_1 + ' mean: ' + str(round(np.mean(data_1), 3)) + ', ste: ' + str(
        round(np.std(data_1) / ((len(data_1)) ** 0.5), 3)))
    print(name_2 + ' mean: ' + str(round(np.mean(data_2), 3)) + ', ste: ' + str(
        round(np.std(data_2) / ((len(data_2)) ** 0.5), 3)))
    wilcoxon_a, wilcoxon_p = st.wilcoxon(data_1, data_2, alternative = 'greater')
    print('Wilcoxon data 2 < data 1 p: ', wilcoxon_p)
    ks_2samp_D, ks_2samp_p = ks_2samp(data_1, data_2)

    # plt.savefig(r'C:\Users\Max\Documents\BA\Git\BA\Statistics\Results\\BigQ/'+name_1+'_'+name_2+'.pdf',bbox_inches='tight')
    plt.show()


#    print('Pairwise Tukey HSD: ')
#
#    data = np.append(data_1, data_2)
#    print(data)
#    print(data.shape)
#    print(ot.shape)
#    print(type(data))
#
#    res2 = pairwise_tukeyhsd(data, ot)
#    print(res2)

### Lab PC
# Group1 'FRON03','WOTE08','GELA01','MIYA09'--mya,GELA missconcepted
# nice Group2 'CANA01','FRTE05','ANNE04','VOJA08','KLTE12'
# nice Group3 'FRID06','JOTE10','ERNA04','STRA06','KAEA05'
# Group4 'DIKE12','GELA06','JAKE10','HEIN10' ,'DIKE11' #dicke11 weg
vpCodes = ['G1', 'G2', 'G3', 'G4']
blocks = [1, 2, 5, 6]
g = 3
# group = 'All'
# names = [group, #0
#         group+'_phase1',group+'_phase2', #1,2
#         group+'FB_phase1', group+'FB_phase2',#3,4
#         group+'FB_phase1_m1', group+'FB_phase1_m2',#5,6
#         group+'FB_phase2_m1', group+'FB_phase2_m2',#7,8
#         group+'NOFB_phase1', group+'NOFB_phase2',#9,10
#         group+'NOFB_pre_phase1',group+'NOFB_post_phase1',#11,12
#         group+'NOFB_pre_phase2',group+'NOFB_post_phase2']#13,14
name = 'Gas_FB_m2'
# 1,2,3,4
# 5,6,7,8

i = 0
alleG = pd.DataFrame()
while i < len(vpCodes):

    fb1 = []
    fb2 = []
    alleblocks = pd.DataFrame()
    # subjects singls
    path = r"C:\Users\Max\Documents\BA\Git\BA\Statistics\SortedData\Grouped/" + vpCodes[i] + "/*.csv"
    current = [pd.read_csv(f) for f in sorted(glob.glob(path))]

    j = 0
    for c in current:
        c['Abs_Error'] = abs(c['Fehler'])
        c['Block'] = j
        if c['PuckMasse'].iloc[0] == 0.7:
            c['PuckName'] = 'light'
        elif c['PuckMasse'].iloc[0] == 1.25:
            c['PuckName'] = 'medium'
        elif c['PuckMasse'].iloc[0] == 2.4:
            c['PuckName'] = 'heavy'
        elif c['PuckMasse'].iloc[0] == 1:
            c['PuckName'] = 'neutral'
        c['Group'] = i + 1
        if i == 0:
            c['GroupR'] = 'Group1'
        if i == 1:
            c['GroupR'] = 'Group2'
        if i == 2:
            c['GroupR'] = 'Group3'
        if i == 3:
            c['GroupR'] = 'Group4'

        if j == 0:
            c['Phase'] = 'prior'
        if j == 1 or j == 2:
            c['Phase'] = 'FB1'
        if j == 3:
            c['Phase'] = 'NoFBpre1'
        if j == 4:
            c['Phase'] = 'NoFBpost1'
        if j == 5 or j == 6:
            c['Phase'] = 'FB2'
        if j == 7:
            c['Phase'] = 'NoFBpre2'
        if j == 8:
            c['Phase'] = 'NoFBpost2'

        alleblocks = pd.concat([alleblocks, c], sort=False)
        j = j + 1

    alleG = pd.concat([alleG, alleblocks], sort=False)

    # print(alleG[(alleG.Phase == 'prior' )])
    i = i + 1

#alleG.to_csv(r'C:\Users\Max\Documents\BA\Git\BA\Statistics\SortedData/Grouped/BiqQuestions/AllData.csv')

colors = ['black', 'royalblue', 'orange', 'green', 'darkorchid']

for sub, data in alleG[(alleG.Phase == 'FB1') | (alleG.Phase == 'FB2')].groupby('VPcode'):
    coil = data[data.Feder == 'konstant']
    coil = coil['Abs_Error']
    gas = data[data.Feder == 'exponentiell']
    gas = gas['Abs_Error']

    color = colors[data['Group'].iloc[0]]
    plt.figure(1)
    plt.title('Absolute Error Difference between Springs')
    plt.ylim([0.5, 3.5])
    plt.xlim([0.5, 3.5])
    plt.plot(np.linspace(0, 10, 1000), np.linspace(0, 10, 1000), color='grey', alpha=0.2)
    plt.xlabel('Coil Spring Absolute Error')
    plt.ylabel('Gas Spring Absolute Error')
    plt.errorbar(np.mean(coil), gas.mean(), coil.std() / (len(coil) ** 0.5), gas.std() / (len(gas) ** 0.5), fmt='.',
                 markersize=8, capsize=5, color=color)

    # plt.scatter(np.mean(coil), gas.mean(), color = color)
#    for cap in caps:
#        cap.set_markeredgewidth(5)
# crossPlot(coil, gas, data['Group'].iloc[0], color)

for sub, data in alleG[(alleG.Phase == 'NoFBpost1') | (alleG.Phase == 'NoFBpost2')].groupby('VPcode'):
    # print(sub)

    coil = data[data.Feder == 'konstant']
    coil = coil['Abs_Error']
    gas = data[data.Feder == 'exponentiell']
    gas = gas['Abs_Error']
    # print(coil.mean())
    # print(gas.mean())
    color = colors[data['Group'].iloc[0]]
    plt.figure(2)
    plt.title('Absolute Error Difference between Springs')
    #    plt.ylim([0.5,3.5])
    #    plt.xlim([0.5,3.5])
    plt.plot(np.linspace(0, 10, 1000), np.linspace(0, 10, 1000), color='grey', alpha=0.2)
    plt.xlabel('Coil Spring Absolute Error')
    plt.ylabel('Gas Spring Absolute Error')
    plt.errorbar(np.mean(coil), gas.mean(), coil.std() / (len(coil) ** 0.5), gas.std() / (len(gas) ** 0.5), fmt='.',
                 markersize=8, capsize=5, color=color)

# Spring Phases
phs1 = []
j = 0
# data.Group == 1-4 &
for sub, data in alleG[(alleG.Block == 1) | (alleG.Block == 2) | (alleG.Block == 3) | (alleG.Block == 4)].groupby(
        'VPcode'):
    if j == 0:
        phs1 = data
        j = j + 1
    else:
        phs1 = pd.concat([phs1, data])

phs2 = []
j = 0
for sub, data in alleG[(alleG.Block == 5) | (alleG.Block == 6) | (alleG.Block == 7) | (alleG.Block == 8)].groupby(
        'VPcode'):
    if j == 0:
        phs2 = data
        j = j + 1
    else:
        phs2 = pd.concat([phs2, data])

compare_two_absErrordist(phs1['Abs_Error'], phs2['Abs_Error'], 'phase1', 'phase2', bins=40)

# Masses
path = r"C:\Users\Max\Documents\BA\Git\BA\Statistics\SortedData\Grouped/G1/*.csv"
g1 = [pd.read_csv(f) for f in sorted(glob.glob(path))]
path = r"C:\Users\Max\Documents\BA\Git\BA\Statistics\SortedData\Grouped/G2/*.csv"
g2 = [pd.read_csv(f) for f in sorted(glob.glob(path))]
for t in g2:
    t['Abs_Error'] = np.abs(t['Fehler'])
path = r"C:\Users\Max\Documents\BA\Git\BA\Statistics\SortedData\Grouped/G3/*.csv"
g3 = [pd.read_csv(f) for f in sorted(glob.glob(path))]
for t in g3:
    t['Abs_Error'] = np.abs(t['Fehler'])
path = r"C:\Users\Max\Documents\BA\Git\BA\Statistics\SortedData\Grouped/G4/*.csv"
g4 = [pd.read_csv(f) for f in sorted(glob.glob(path))]

coilm1 = pd.concat([g1[1], g4[5]], sort=False)
coilm21 = pd.concat([g1[2], g4[6]], sort=False)
coilm22 = pd.concat([g2[1], g3[5]], sort=False)
coilm3 = pd.concat([g2[2], g3[6]], sort=False)

gasm1 = pd.concat([g2[5], g3[1]], sort=False)
gasm21 = pd.concat([g2[6], g3[2]], sort=False)
gasm22 = pd.concat([g1[5], g4[1]], sort=False)
gasm3 = pd.concat([g1[6], g4[2]], sort=False)

coilm1['Abs_Error'] = np.abs(coilm1['Fehler'])
coilm22['Abs_Error'] = np.abs(coilm22['Fehler'])
coilm3['Abs_Error'] = np.abs(coilm3['Fehler'])
gasm1['Abs_Error'] = np.abs(gasm1['Fehler'])
gasm21['Abs_Error'] = np.abs(gasm21['Fehler'])
gasm3['Abs_Error'] = np.abs(gasm3['Fehler'])

coilm2 = pd.concat([coilm21, coilm22], sort=False)
gasm2 = pd.concat([gasm21, gasm22], sort=False)

compare_two_absErrordist(coilm21['Abs_Error'], coilm22['Abs_Error'], 'Coilm2 heavy', 'coilm2 light', bins=40)
compare_two_absErrordist(gasm21['Abs_Error'], gasm22['Abs_Error'], 'Gasm2 heavy', 'gasm2 light', bins=40)

compare_three_absErrordist(gasm1['Abs_Error'], gasm2['Abs_Error'], gasm3['Abs_Error'], 'Gasm1', 'gasm2', 'gasm3',
                           bins=40)
compare_three_absErrordist(coilm1['Abs_Error'], coilm2['Abs_Error'], coilm3['Abs_Error'], 'coilm1', 'coilm2', 'coilm3',
                           bins=40)

# NO FB
path = r"C:\Users\Max\Documents\BA\Git\BA\Statistics\SortedData\Grouped/G/*.csv"
g = [pd.read_csv(f) for f in sorted(glob.glob(path))]
pre = pd.concat([g[3], g[7]], sort=False)
post = pd.concat([g[4], g[8]], sort=False)
compare_two_absErrordist(pre['Abs_Error'], post['Abs_Error'], 'All pre', 'All post', bins=40)
is_a_less_b(pre['Abs_Error'], post['Abs_Error'], 'All pre', 'All post', bins=40)

# no fb coil heavy
coil_pre_h = pd.concat([g1[3], g4[7]], sort=False)
coil_post_h = pd.concat([g1[4], g4[8]], sort=False)
compare_two_absErrordist(coil_pre_h['Abs_Error'], coil_post_h['Abs_Error'], 'coil_pre_heavy', 'coil_post_heavy',
                         bins=40)
is_a_less_b(coil_pre_h['Abs_Error'], coil_post_h['Abs_Error'], 'coil_pre_heavy', 'coil_post_heavy',bins=40)

# no fb coil light
coil_pre_light = pd.concat([g2[3], g3[7]], sort=False)
coil_post_light = pd.concat([g2[4], g3[8]], sort=False)
compare_two_absErrordist(coil_pre_light['Abs_Error'], coil_post_light['Abs_Error'], 'coil_pre_light', 'coil_post_light',
                         bins=40)
is_a_less_b(coil_pre_light['Abs_Error'], coil_post_light['Abs_Error'], 'coil_pre_light', 'coil_post_light',bins=40)

# no fb gas heavy
gas_pre_h = pd.concat([g2[7], g3[3]], sort=False)
gas_post_h = pd.concat([g2[8], g3[4]], sort=False)
compare_two_absErrordist(gas_pre_h['Abs_Error'], gas_post_h['Abs_Error'], 'gas_pre_heavy', 'gas_post_heavy', bins=40)
is_a_less_b(gas_pre_h['Abs_Error'], gas_post_h['Abs_Error'], 'gas_pre_heavy', 'gas_post_heavy',bins=40)

# no fb gas light
gas_pre_light = pd.concat([g1[7], g4[3]], sort=False)
gas_post_light = pd.concat([g1[8], g4[4]], sort=False)
compare_two_absErrordist(gas_pre_light['Abs_Error'], gas_post_light['Abs_Error'], 'gas_pre_light', 'gas_post_light',
                         bins=40)
is_a_less_b(gas_pre_light['Abs_Error'], gas_post_light['Abs_Error'], 'gas_pre_light', 'gas_post_light',bins=40)

# NO FB post vs FB
compare_two_absErrordist(coil_post_h['Abs_Error'], coilm3['Abs_Error'], 'coil m3 no FB post', 'coil m3 FB', bins=40)
compare_two_absErrordist(coil_post_light['Abs_Error'], coilm1['Abs_Error'], 'coil m1 no FB post', 'coil m1 FB', bins=40)
compare_two_absErrordist(gas_post_h['Abs_Error'], gasm3['Abs_Error'], 'gas m3 no FB post', 'gas m3 FB', bins=40)
compare_two_absErrordist(gas_post_light['Abs_Error'], gasm1['Abs_Error'], 'gas m1 no FB post', 'gas m1 FB', bins=40)

#
# together=[]
# together=pd.DataFrame(together)
# current=[]
# i=0
# while i< len(vpCodes):
#    print(vpCodes[i])
#    fb1 = []
#    fb2 = []
#    #subjects singls
#    path=r"C:\Users\Max\Documents\BA\Git\BA\Statistics\SortedData\Grouped/"+vpCodes[i]+"/*.csv"
#    current = [pd.read_csv(f) for f in sorted(glob.glob(path))]
#
#    for c in current:
#        c['Abs_Error']=abs(c['Fehler'])
#
#    fb1 = pd.concat([current[1],current[2]])
#
#    fb2 = pd.concat([current[5],current[6]])
#
#    subjects1 = fb1.groupby(fb1['VPcode'])
#    subjects2 = fb2.groupby(fb2['VPcode'])
#
#    for sub in subjects1:
#        for sub2 in subjects2:
#            if sub[0] == sub2[0]:
#                print(sub[0])
#                coil =  []
#                gas = []
#                if sub[1]['Feder'].iloc[0] == 'konstant':
#                    coil = sub[1]['Abs_Error']
#                    gas = sub2[1]['Abs_Error']
#                elif sub[1]['Feder'].iloc[0] == 'exponentiell':
#                    gas = sub[1]['Abs_Error']
#                    coil = sub2[1]['Abs_Error']
#                crossPlot(coil, gas, 1, 'green')
#
#    i=i+1


# name='Gas_FB_m1'
# data1 = pd.read_csv(r'C:\Users\Max\Documents\BA\Git\BA\Statistics\SortedData\Grouped/BiqQuestions/'+name+'.csv')
#
# name='Gas_FB_m2'
# data2 = pd.read_csv(r'C:\Users\Max\Documents\BA\Git\BA\Statistics\SortedData\Grouped/BiqQuestions/'+name+'.csv')
#
# name='Gas_FB_m3'
# data3 = pd.read_csv(r'C:\Users\Max\Documents\BA\Git\BA\Statistics\SortedData\Grouped/BiqQuestions/'+name+'.csv')
#
#
#
# compare_three_absErrordist(data1['Abs_Error'], data2['Abs_Error'] , data3['Abs_Error'], 'Light puck', 'Medium Puck', 'Heavy Puck')

# plt.figure(1)
# plt.ylim([0,5])
# devided=[data['Abs_Error'],data2['Abs_Error'],data3['Abs_Error']]
# plt.boxplot(devided,0,'',positions=[1,2,3], widths=0.5)
# plt.boxplot(data['Abs_Error'][35:],0,'', positions=[2,1], widths=50)
