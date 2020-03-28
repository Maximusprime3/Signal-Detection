import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import psignifit as pn


#collect data
path=r"C:\Users\Max\PycharmProjects\SignalDetection\data\2AFC2/*.csv"
raw_data = [pd.read_csv(f) for f in sorted(glob.glob(path))]
raw_data = raw_data[0]

#recreate subject answer from where the stimilus was and if the answer was correct
# (Couldn't find/alter data saving in experiment and add subject answer)
preprocessed_data = []
i=0
#recreate subject answer for every trial block (signal intensity)
for sig, d in raw_data.groupby('signal_intensity'):
    print(sig)
    d=d[5:]
    #check where the stimulus was and save it as a number
    subject_1 = d[d.signal_on_stimuluspos1 == True]
    subject_1.loc[:,'Stimulus_Present'] = 1
    subject_2= d[d.signal_on_stimuluspos2 == True]
    subject_2.loc[:,'Stimulus_Present'] = 2
    d = pd.concat([subject_1, subject_2])

    #split data into correct and wrong answerws
    correct = d[d.response == 'c']
    #print(correct[:3])
    wrong = d[d.response == 'w']
    #if the response was correct the subject answered with the location of the stimulus
    if len(correct['Subject_Action']) > 0:
        correct.loc[:,'Subject_Action'] = correct['Stimulus_Present']
    #if the response was wrong the subject answered with the other location where the stimulus was not
    if len(wrong['Subject_Action'])>0:
        wrong.loc[:,'Subject_Action'] = (3 - wrong['Stimulus_Present'])

    #append data and safe in preprocessed_data
    d= pd.concat([correct, wrong])
    preprocessed_data.append(d)
    i=i+1

print('Durchgänge: ',len(preprocessed_data))

#make empty data frame for psignifit
data = np.ndarray(shape=(len(preprocessed_data), 3), dtype=float)

i = 0
#go through every trial block in the data
for d in preprocessed_data:

    signal_intensity = d['signal_intensity'].iloc[0]
    #from the preprocessing we can do this
    response = d['Stimulus_Present'] - d['Subject_Action'] # 0=correct -1 or 1 = wrong
    #but we already have wrong and correct responses
    miss = d[(d.response == 'w')]
    hit = d[(d.response == 'c')]

    total_trials = len(d)
    correct_trials = len(hit['response'])

    #append the data in the right format for psignifit
    data[i] = [signal_intensity, correct_trials, total_trials]

    i = i+1



#set options for psignifit
options =dict()# initialize as an empty struct
options['sigmoidName'] = 'norm'
options['expType'] = '2AFC'
#calculate result and plot the psychometric function
result = pn.psignifit(data,options)
pn.psigniplot.plotPsych(result)
plt.show()
#print the parameters of the psychometric function
print('Threshold:', result['Fit'][0])
print('Width:    ', result['Fit'][1])
print('Lambda:   ', result['Fit'][2])
print('Gamma:    ', result['Fit'][3])
print('Eta:      ', result['Fit'][4])