import pandas as pd
import numpy as np
import glob
import bayesfit as bf
import matplotlib.pyplot as plt
import psignifit as ps


path=r"C:\Users\Max\PycharmProjects\SignalDetection\data\Max2/*.csv"
raw_data = [pd.read_csv(f) for f in sorted(glob.glob(path))]

i=0
for sig_int, data in raw_data.groupby(raw_data.signal_intensity):
    i=i+1

    test_trials = data[:5]
    data2 = data[5:]
    hit = [x for _, x in data2.groupby(data2.response == 'Hit')]
    miss = [x for _, x in data2.groupby(data2.response == 'Miss')]


print('Durchgänge: ',i)

data = np.ndarray(shape=(len(raw_data), 3), dtype=float)

i = 0
for d in raw_data:
    d = d[1:]
    signal_intensity = d['signal_intensity'].iloc[0]

    miss = [x for _, x in d.groupby(d.response == 'w')]
    hit = [x for _, x in d.groupby(d.response == 'c')]
    for h in hit:
        print(len(h))
    print(hit[0]['response'])


    total_trials = len(d)
    correct_trials = len(hit[1])

    data[i] = [signal_intensity, correct_trials, total_trials]

    i = i+1

print(data)
options             = dict();   # initialize as an empty dictionary
options['sigmoidName'] = 'norm';   # choose a cumulative Gauss as the sigmoid
options['expType']     = '2AFC';   # choose 2-AFC as the experiment type
                                   # this sets the guessing rate to .5 (fixed) and
                                   # fits the rest of the parameters
result = ps.psignifit(data,options);
ps.psigniplot.plotPsych(result)


metrics, options = bf.fitmodel(data, nafc = 2)
bf.plot_psyfcn(data, options, metrics)

#plt.savefig(r'C:\Users\Max\PycharmProjects\SignalDetection\DataAnalysis/BayesFitPlot.pdf')
plt.show()








path=r"C:\Users\Max\PycharmProjects\SignalDetection\data\Max2/*.csv"
raw_data = [pd.read_csv(f) for f in sorted(glob.glob(path))]

print('Durchgänge: ',len(raw_data))

data = np.ndarray(shape=(len(raw_data), 3), dtype=float)

i = 0
for d in raw_data:
    d = d[1:]
    signal_intensity = d['signal_intensity'].iloc[0]

    miss = [x for _, x in d.groupby(d.response == 'w')]
    hit = [x for _, x in d.groupby(d.response == 'c')]
    #for h in hit:
       # print(len(h))
    #print(hit[0]['response'])


    total_trials = len(d)
    correct_trials = len(hit[1])

    data[i] = [signal_intensity, correct_trials, total_trials]

    i = i+1




metrics, options = bf.fitmodel(data, nafc = 2)
bf.plot_psyfcn(data, options, metrics)

#plt.savefig(r'C:\Users\Max\PycharmProjects\SignalDetection\DataAnalysis/BayesFitPlot.pdf')
plt.show()


options             =dict()# initialize as an empty struct
options['sigmoidName'] = 'norm'
options['expType'] = '2AFC'

#options.sigmoidName = 'norm'
#options.expType     = 'YesNo'
result = pn.psignifit(data,options)
pn.psigniplot.plotPsych(result)
plt.show()
print('ferrtig')
#Daten Speicherung vom 2IFC mit response = first, second, nothing   Und stimulus = first, second
#Vergleich von Daten mit vergleichbaren Bedingungen Threshold, Parameter von Signifit
