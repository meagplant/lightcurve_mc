#!/usr/bin/python
import numpy as np
import matplotlib
matplotlib.use('Agg')
import os
from func import getdata, covariance, chi_sqrd_like
import matplotlib.pyplot as plt
import emcee
from schwimmbad import MultiPool
from contextlib import closing
import csv
from datetime import datetime
from getvars import get_vars
import sys

tempfile = sys.argv[1] #name of tempfile from command line

#getting data from tempfile
datepath, timedatepath, filedatepath, N, quasar, intrindex, mockdex, theta_true, truths, labels = get_vars(tempfile)

if intrindex == 'Intr':
    A_m_true, omeg_m_true, deltaT_true = theta_true

else:
    A_m_true, omeg_m_true, A_l_true, omeg_l_true, deltaT_true = theta_true

time, mag, magerr, imgno = getdata(tempfile,quasar,mockdex)

if mockdex == 'Mock':
    mock = np.loadtxt("{}/mockdata.csv".format(timedatepath), delimiter=",")

def set_prior_range(set_range=False):
    #sets the range the walkers are allowed to search in

    if set_range == True:

        Am_upper, Am_lower = 0, -20
        omegam_upper, omegam_lower = 0, -20
        Al_upper, Al_lower = 0, -20
        omegal_upper, omegal_lower = 0, -20
        dT_upper, dT_lower = deltaT_true+75, deltaT_true-75

    else:

        Am_upper, Am_lower = A_m_true+5, A_m_true-5
        omegam_upper, omegam_lower = omeg_m_true+10, omeg_m_true-10
        Al_upper, Al_lower = A_l_true+10, A_l_true-10
        omegal_upper, omegal_lower = omeg_l_true+10, omeg_l_true-10
        dT_upper, dT_lower = deltaT_true+75, deltaT_true-75

    return Am_upper, Am_lower, omegam_upper, omegam_lower, Al_upper, Al_lower, omegal_upper, omegal_lower, dT_upper, dT_lower

def log_likelihood(theta):
    #likelihood function using chi-squared calculated from covariance matrix

    covar = covariance(theta, time, magerr, imgno, intrindex)

    if mockdex == 'Mock':
        chisqrd = chi_sqrd_like(covar, mock)
    else:
        chisqrd = chi_sqrd_like(covar, mag)

    like = -chisqrd/2

    print('likelihood ran')

    return like

def log_prior(theta):
    #makes the probability zero everywhere outside the prior range

    print(len([theta]))
    
    Am_upper, Am_lower, omegam_upper, omegam_lower, Al_upper, Al_lower, omegal_upper, omegal_lower, dT_upper, dT_lower = set_prior_range(set_range=True)
    
    if intrindex == 'Intr':
        A_m, omega_m, deltaT = theta
        if Am_lower < A_m < Am_upper and omegam_lower < omega_m < omegam_upper and dT_lower < deltaT < dT_upper:
            return 0.0
        else:
            return -np.inf        

    else:
        A_m, omega_m, A_l, omega_l, deltaT = theta
        if Am_lower < A_m < Am_upper and omegam_lower < omega_m < omegam_upper and Al_lower < A_l < Al_upper and omegal_lower < omega_l < omegal_upper and dT_lower < deltaT < dT_upper:
            return 0.0
        else:
            return -np.inf

def log_probability(theta):
    #using likelihood and prior functions, calculates the probability

    lp = log_prior(theta)

    if not np.isfinite(lp):
        return -np.inf

    print('probablility ran')
    return lp + log_likelihood(theta)

def progress_fig(ndim, dat, run):
    #creates a figure for each run on each node that shows the progress of the walkers as a function of the step number
    
    plt.figure(1)
    fig, axes = plt.subplots(len(truths), figsize=(10, 7), sharex=True)
    for i in range(ndim):
        ax = axes[i]
        ax.plot(dat[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(dat))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)
    axes[-1].set_xlabel("step number");
    #plt.plot(dat[:, :, 0], "k", alpha=0.3)
    positionsfile = '{}/positionsofwalkers_{}samples_{}.png'.format(timedatepath, N, int(run.microsecond))
    plt.savefig(positionsfile)

def reject3sig(array):
    #rejects values not within 3 sigma

    meanval = np.mean(array)
    std = np.std(array)
    for i in range(len(array)):
        var = array[i]
        if var > (meanval + 3*std) or var < (meanval - 3*std):
            np.where(array==var, 0, array)

def flatten_samples(dat):
    #flattens the sample range to speed up calculations

    # tau = sampler.get_autocorr_time()
    # print(tau)
       
    print("Mean acceptance fraction: {0:.3f}".format(np.mean(dat.acceptance_fraction)))

    tau = 100

    flat_samples = dat.get_chain(discard=int(tau), thin=int(tau/2), flat=True)
    print(flat_samples.shape)

    for j in range(len(flat_samples[0])):
        reject3sig(flat_samples[:,j])

    return flat_samples

def write_datafiles(dat):
    #writes emcee data to files which will be plotted

    now1 = datetime.now()
    with open('{}/flatsamples_{}.csv'.format(filedatepath,int(now1.microsecond)), 'w') as f:
        write = csv.writer(f)
        write.writerows(dat)
        print('file written')

#################
# running emcee #
#################

soln = truths[:]

init = datetime.now()

with closing(MultiPool()) as pool:

    pos = soln + np.random.randn(35, len(truths))
    nwalkers, ndim = pos.shape

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool=pool, moves=emcee.moves.DEMove())
    sampler.run_mcmc(pos, N, progress=True);
    pool.terminate()

samples = sampler.get_chain()

now2 = datetime.now()
timediff = now2-init
print('time taken:', timediff.seconds)

################################
# make figures and write files #
################################

progress_fig(ndim, samples, now2)
flat_samples = flatten_samples(sampler)

write_datafiles(flat_samples)
