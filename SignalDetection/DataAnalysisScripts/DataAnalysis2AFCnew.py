import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import psignifit as pn

#2AFCnew
#Get the data
path=r"C:\Users\Max\PycharmProjects\SignalDetection\data\2AFCnew/*.csv"
raw_data = [pd.read_csv(f) for f in sorted(glob.glob(path))]
raw_data = raw_data[2]
preprocessed_data = []
for sig, d in raw_data.groupby('signal_intensity'):
    preprocessed_data.append(d)

print('Durchgänge: ',len( preprocessed_data))

#create empty data frame the right format for psignifit
data_first_answer = np.ndarray(shape=(len( preprocessed_data), 3), dtype=float)
data_second_answer = np.ndarray(shape=(len( preprocessed_data), 3), dtype=float)
i = 0
#go through every trial block (signal intensities)
#fist answer
for d in  preprocessed_data:
    #excluede test trials
    d = d[5:]

    signal_intensity = d['signal_intensity'].iloc[0]

    # first answer == 1 -> signal was seen in first stimulus
    # first answer == 2 -> signal was not seen in first stimulus
    # first answer == 0 -> no answer was given
    #sort for correct and wrong first answers
    correct = d[(d.first_answer == d.stimulus_pos)]
    wrong = d[(np.logical_and(d.first_answer != 0, d.first_answer != d.stimulus_pos))]
    total_trials = len(correct) + len(wrong)
    correct_trials = len(correct)

    #store in right format for psignifit
    data_first_answer[i] = [signal_intensity, correct_trials, total_trials]


    #store final choice where the signal was seen either in 1st or 2nd stimulus
    second_answer =[]

    #second answer == 1 -> signal was seen in second stimulus
    #second answer == 2 -> signal was not seen in second stimulus -> seen in first
    #second answer == 0 -> no answer was given --> first answer = final answer
    h = 0
    first_answer = d['first_answer'].reset_index()

    for entry in d['second_answer']:
        if entry == 1:
            second_answer.append(2)
        if entry == 2:
            second_answer.append(1)
        if entry == 0:
            second_answer.append(first_answer['first_answer'][h])
        h = h+1
    d['final_choice'] = second_answer

    correct2 = d[(d.final_choice == d.stimulus_pos)]
    wrong2 = d[(np.logical_and(d.final_choice != 0, d.final_choice != d.stimulus_pos))]
    correct_trials2 = len(correct2)
    total_trials2 = len(correct2) + len(wrong2)
    data_second_answer[i] = [signal_intensity, correct_trials2, total_trials2]

    i = i+1


#psignifit options
options =dict()# initialize as an empty struct
options['sigmoidName'] = 'norm'
options['expType'] = '2AFC'
#compute psignifit result
result = pn.psignifit(data_first_answer,options)
#plot the psychometric function
pn.psigniplot.plotPsych(result)
plt.show()
#print the parameters of the psychometric function
print('Threshold:', result['Fit'][0])
print('Width:    ', result['Fit'][1])
print('Lambda:   ', result['Fit'][2])
print('Gamma:    ', result['Fit'][3])
print('Eta:      ', result['Fit'][4])



#compute psignifit result
result = pn.psignifit(data_second_answer, options)
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
