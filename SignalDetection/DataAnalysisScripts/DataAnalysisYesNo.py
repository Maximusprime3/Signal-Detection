import pandas as pd
import numpy as np
import glob
import bayesfit as bf
import matplotlib.pyplot as plt
import psignifit as pn

#YesNo
path=r"C:\Users\Max\PycharmProjects\SignalDetection\data\YesNo3/*.csv"
raw_data = [pd.read_csv(f) for f in sorted(glob.glob(path))]
raw_data = raw_data[0]
preprocessed_data = []
for sig, d in raw_data.groupby('signal_intensity'):
    preprocessed_data.append(d)



print('Durchgänge: ',len( preprocessed_data))




data = np.ndarray(shape=(len( preprocessed_data), 3), dtype=float)

i = 0
for  d in  preprocessed_data:
    d = d[5:]
    signal_intensity = d['signal_intensity'].iloc[0]
    subject_true = d[(d.Subject_Action == 1)]
    hit = subject_true[(subject_true.Stimulus_Present == 1)]
    false_alarm =   subject_true[(subject_true.Stimulus_Present == 0)]
    subject_false = d[(d.Subject_Action == 0)]
    correctrejection = subject_false[(subject_false.Stimulus_Present == 0)]
    miss = subject_false[(subject_false.Stimulus_Present == 1)]

    #for h in hit:
        #print(len(h))
   # print(hit[0]['response'])


    total_trials = len(d)
    correct_trials = len(hit['Subject_Action'])+len(correctrejection['Subject_Action'])
    #correct_trials = len(hit[1]) + len(correctrejection[1])
    data[i] = [signal_intensity, correct_trials, total_trials]

    i = i+1


#print(data)

#metrics, options = bf.fitmodel(data, nafc = 2)
#bf.plot_psyfcn(data, options, metrics)

options             =dict()# initialize as an empty struct
options['sigmoidName'] = 'norm'
options['expType'] = '2AFC'

#options.sigmoidName = 'norm'
#options.expType     = 'YesNo'
result = pn.psignifit(data,options)
pn.psigniplot.plotPsych(result)
plt.show()
##plt.show()

#plt.savefig(r'C:\Users\Max\PycharmProjects\SignalDetection\DataAnalysis/BayesFitPlot.pdf')

print('Threshold:', result['Fit'][0])
print('Width:    ', result['Fit'][1])
print('Lambda:   ', result['Fit'][2])
print('Gamma:    ', result['Fit'][3])
print('Eta:      ', result['Fit'][4])

