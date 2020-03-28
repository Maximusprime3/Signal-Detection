import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import psignifit as pn

#YesNo
#Get the data
path=r"C:\Users\Max\PycharmProjects\SignalDetection\data\TestYesNo/*.csv"
raw_data = [pd.read_csv(f) for f in sorted(glob.glob(path))]

print('Durchgänge: ',len(raw_data))

#create empty data frame the right format for psignifit
data = np.ndarray(shape=(len(raw_data), 3), dtype=float)

i = 0
#go through every trial block (signal intensities)
for d in raw_data:
    d = d[5:]
    signal_intensity = d['signal_intensity'].iloc[0]
    #sort for subject positve anserws (signal seen)
    subject_true = d[(d.Subject_Action == 1)]
    hit = subject_true[(subject_true.Stimulus_Present == 1)]
    false_alarm =   subject_true[(subject_true.Stimulus_Present == 0)]
    #sort for subject negative asnerw (no signal seen)
    subject_false = d[(d.Subject_Action == 0)]
    correctrejection = subject_false[(subject_false.Stimulus_Present == 0)]
    miss = subject_false[(subject_false.Stimulus_Present == 1)]

    total_trials = len(d)
    correct_trials = len(hit['Subject_Action'])+len(correctrejection['Subject_Action'])

    #store in right format for psignifit
    data[i] = [signal_intensity, correct_trials, total_trials]

    i = i+1


#psignifit options
options =dict()# initialize as an empty struct
options['sigmoidName'] = 'norm'
options['expType'] = '2AFC'
#compute psignifit result
result = pn.psignifit(data,options)
#plot the psychometric function
pn.psigniplot.plotPsych(result)
plt.show()

#save the plot
#plt.savefig(r'C:\Users\Max\PycharmProjects\SignalDetection\DataAnalysis/BayesFitPlot.pdf')

#print the parameters of the psychometric function
print('Threshold:', result['Fit'][0])
print('Width:    ', result['Fit'][1])
print('Lambda:   ', result['Fit'][2])
print('Gamma:    ', result['Fit'][3])
print('Eta:      ', result['Fit'][4])
