#!/usr/bin/python
import sys
import os
from os import listdir
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import corner
import pandas as pd
from shutil import copy
from getvars import get_vars

#getting data from tempfile
tempfile = sys.argv[1]

datepath, timedatepath, filedatepath, N, quasar, intrindex, mockdex, theta_true, truths, labels = get_vars(tempfile)

if intrindex == 'Intr':
    A_m_true, omeg_m_true, deltaT_true = theta_true

else:
    A_m_true, omeg_m_true, A_l_true, omeg_l_true, deltaT_true = theta_true

def get_data():
    #extracts emcee data from files and compiles it into one array

    onlyfiles = [f for f in listdir(filedatepath) if os.path.isfile(os.path.join(filedatepath, f))]
    print(onlyfiles)

    global nfiles
    nfiles = len(onlyfiles)

    flat_samples = []
    for i in range(nfiles):
        lst = pd.read_csv('{}/{}'.format(filedatepath,onlyfiles[i]), encoding= 'unicode_escape')
        flat_samples.append(lst)
    
    global ndim
    _, ndim = flat_samples[0].shape

    flat_array = np.concatenate(flat_samples, axis=0 )

    return flat_array

def make_output():
    #makes an output file: basically copies the tempfile into the run's directory, where we will add the final calculated values
    wd = os.getcwd()
    finalfile = '{}/output.txt'.format(timedatepath)
    copy('{}/{}.txt'.format(wd,tempfile), '{}/output.txt'.format(timedatepath))
    os.remove('{}/{}.txt'.format(wd,tempfile)) #deletes tempfile after data has been transferred

    return finalfile

def cornerplot(flat_array):
    #plots emcee data in a corner (staircase) plot

    plt.figure()
    fig2 = corner.corner(
        flat_array, labels=labels, truths=truths, levels=(0.68,0.95)
    );
    #plt.plot(flat_array)
    resultsfile = '{}/{}_{}{}_results_{}samples.png'.format(timedatepath, quasar, intrindex, mockdex, (N*nfiles))
    plt.savefig(resultsfile)

def append_new_line(file_name, text_to_append):
    #same copiedd function for adding lines to files

    """Append given text as a new line at the end of file"""
    # Open the file in append & read mode ('a+')
    with open(file_name, "a+") as file_object:
        # Move read cursor to the start of file.
        file_object.seek(0)
        # If file is not empty then append '\n'
        data = file_object.read(100)
        if len(data) > 0:
            file_object.write("\n")
        # Append text at the end of file
        file_object.write(text_to_append)

def final_estimates(flat_array):
    #calculates final values

    finalfile = make_output()

    mcmc_vals=[]
    mcmc_err=[]

    for i in range(ndim):

        mcmc = np.percentile(flat_array[:, i], [16, 50, 84])
        mcmc_vals.append(mcmc[1])

        if i != (ndim-1):

            q = []
            for j in range(len(mcmc)-1):

                qval = np.exp(mcmc[j+1]) - np.exp(mcmc[j])
                q.append(qval)

            avg = sum(q)/len(q)
            mcmc_err.append(avg)

        else:

            q1 = np.diff(mcmc)
            avg1 = sum(q1)/len(q1)
            mcmc_err.append(avg1)

    append_new_line(finalfile,'A_m = {} +/- {}'.format(np.exp(mcmc_vals[0]),mcmc_err[0]))
    append_new_line(finalfile,'omega_0_m = {} +/- {}'.format(np.exp(mcmc_vals[1]),mcmc_err[1]))

    if intrindex == 'Intr':
        append_new_line(finalfile,'deltaT = {} +/- {}'.format(mcmc_vals[2],mcmc_err[2]))

    else:
        append_new_line(finalfile,'A_l = {} +/- {}'.format(np.exp(mcmc_vals[2]),mcmc_err[2]))
        append_new_line(finalfile,'omega_0_l = {} +/- {}'.format(np.exp(mcmc_vals[3]),mcmc_err[3]))
        append_new_line(finalfile,'deltaT = {} +/- {}'.format(mcmc_vals[4],mcmc_err[4]))

#main code
samples = get_data()
cornerplot(samples)
final_estimates(samples)
