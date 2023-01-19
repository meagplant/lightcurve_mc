#!/usr/bin/python
import numpy as np
import sys
import os
import matplotlib
matplotlib.use('Agg')
from func import getdata, reg_vals, intrinsic_vals, covariance, chi_sqrd_like
from datetime import datetime
from scipy.signal import argrelextrema
from scipy.linalg import sqrtm
from tqdm import trange
import matplotlib.pyplot as plt

####################
# global variables #
####################

quasar = sys.argv[1] #this should be the name of the file you want to evaluate inputted in the command line of shell script

#truth values:
A_m_true = np.log(0.003)
omeg_m_true = np.log(0.003)
A_l_true = np.log(0.0001)
omeg_l_true = np.log(0.0001)
dT_true = float(sys.argv[2]) #also in command line

N = float(sys.argv[3]) #total number of runs divided by the amount of nodes used (i.e. for 2000 total runs on 4 nodes, this would be 500
day = datetime.now()
intrindex = sys.argv[4] #tells script whether only intrinsic variables are used, should be set to 'Intr' on command line if so
mockdex = sys.argv[5] #tells script whether data will be user generated, should be set to 'Mock' if so
tempfile = sys.argv[6] #for TDC runs, this should be set to which rung is being used (i.e. rung0) so that it can access that folder. otherwise, I normally just set it to the name of the name of the lightcurve (i.e. j1001) so that i can run other lightcurves at the same time and the data won't overwrite itself

print(omeg_m_true)

#############
# functions #
#############

def filenames(now):
    #creates the names of the folders data will be stored in

    wd = os.getcwd()
    datepath = "{}/{}".format(wd,now.strftime("%d-%m-%Y"))
    timedatepath = "{}/{}_{}{}2_{}".format(datepath,quasar,intrindex,mockdex,now.strftime("%H%M"))
    filedatepath = "{}/files".format(timedatepath)

    return wd, datepath, timedatepath, filedatepath

def makefiles():
    #generates folders

    wd, datepath, timedatepath, filedatepath = filenames(day)

    if os.path.exists(datepath)==False:
        os.mkdir(datepath)
        print("Date Directory Created")

    if os.path.exists(timedatepath)==False:
        os.mkdir(timedatepath)
        print("Time Directory Created")

    if os.path.exists(filedatepath)==False:
        os.mkdir("{}".format(filedatepath))
        print("File Directory Created")

def append_new_line(file_name, text_to_append):
    #funtion i copieddd off the internet to append a new line to a text file

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

def write_file(intr):
    #creates tempfile to store directory pathways and variables to be accessed by future files in the run

    wd, datepath, timedatepath, filedatepath = filenames(day)

    if intr == 'Intr':
        theta_truth, truths, labels = intrinsic_vals(A_m_true, omeg_m_true, dT_true)

    else:
        theta_truth, truths, labels = reg_vals(A_m_true, omeg_m_true, A_l_true, omeg_l_true, dT_true)

    outputfile = '{}/{}.txt'.format(wd,tempfile)
    if os.path.exists(outputfile)==True:
        os.remove(outputfile)

    append_new_line(outputfile, datepath)
    append_new_line(outputfile, timedatepath)
    append_new_line(outputfile, filedatepath)

    append_new_line(outputfile, str(N))
    append_new_line(outputfile, quasar)

    append_new_line(outputfile, intrindex)
    append_new_line(outputfile, mockdex)

    for i in range(len(truths)):
        append_new_line(outputfile, '{}: {}'.format(labels[i],truths[i]))

    append_new_line(outputfile, 'move: DE')

###############
# unused code #
###############

def plotdata(file):
    #plots the lightcurves

    timelist, maglist, magerrlist, imgnolist = getdata(tempfile,file,mockdex)

    timedex = len(timelist)/2
    time1 = timelist[:timedex]

    wd, datepath, timedatepath, filedatepath = filenames(day)

    plt.figure(1)
    plt.plot(time1,maglist[:timedex],label='Image 1')
    plt.plot(time1,maglist[timedex:],label='Image 2')
    plt.legend()
    plt.xlabel('Days')
    plt.ylabel('Magnitude')
    plt.savefig('{}/{}_data.png'.format(timedatepath,quasar))

def genmockdata(file):
    #generates mock light curves using specified truth values to test code's accuracy

    if intrindex == 'Intr':
        theta_truth, truths, labels = intrinsic_vals(A_m_true, omeg_m_true, dT_true)

    else:
        theta_truth, truths, labels = reg_vals(A_m_true, omeg_m_true, A_l_true, omeg_l_true, dT_true)

    timelist, maglist, magerrlist, imgnolist = getdata(file)
    wd, datepath, timedatepath, filedatepath = filenames(day)

    print(len(timelist))

    covarmatrix = covariance(theta_truth, timelist, magerrlist, imgnolist, intrindex)
    randvector = np.random.normal(0,1,size=len(timelist))

    mockdata = np.dot(sqrtm(covarmatrix),randvector).astype(float)

    # chisqrd = chi_sqrd_like(covarmatrix,mockdata)

    # cov0 = covariance([A_m_true, omeg_m_true, 0], timelist, magerrlist, imgnolist, intrindex)
    # cov37 = covariance([A_m_true, omeg_m_true, -37], timelist, magerrlist, imgnolist, intrindex)

    # chi0 = chi_sqrd_like(cov0,mockdata)
    # chi37 = chi_sqrd_like(cov37,mockdata)

    # print(chi0,chi37)

    timedex = int(len(timelist)/2)
    time1 = timelist[:timedex]
    
    plt.figure()
    plt.plot(time1,mockdata[:len(time1)])
    plt.plot(time1,mockdata[len(time1):])
    plt.savefig('{}/mockdata.png'.format(timedatepath))

    output_array = np.array(mockdata)
    np.savetxt("{}/mockdata.csv".format(timedatepath), output_array, delimiter=",")

    return mockdata

def chisquared_plot(file, intr):
    #plots the chi-squared as a function of one of the parameters

    timelist, maglist, magerrlist, imgnolist = getdata(file)
    wd, datepath, timedatepath, filedatepath = filenames(day)

    if mockdex == 'Mock':
        mock = genmockdata(file)

    dAs = np.linspace(-50,50,1000)
    def chiloop(omeg):

        chisqrd_array = np.zeros(len(omeg))
        for i in trange(len(omeg)):
            if intr == 'Intr':
                cov = covariance([A_m_true, omeg_m_true, omeg[i]], timelist, magerrlist, imgnolist, intrindex)
            else:
                cov = covariance([A_m_true, omeg[i], A_l_true, omeg_l_true, dT_true], timelist, magerrlist, imgnolist, intrindex)

            if mockdex == 'Mock':
                chisqrd = chi_sqrd_like(cov,mock)
            else:
                chisqrd = chi_sqrd_like(cov,maglist)

            chisqrd_array[i] = chisqrd

        return chisqrd_array

    chisqrd_list = chiloop(dAs)

    minAs = argrelextrema(np.array(chisqrd_list), np.less)
    mindex = minAs[0]
    min_omeg = [float("{:.3f}".format(dAs[i])) for i in mindex]
    mins = [float("{:.3f}".format(chisqrd_list[j])) for j in mindex]

    print(min_omeg,mins)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(dAs,chisqrd_list)
    # plt.plot(min_omeg,mins,'go')
    # for i,j in zip(min_omeg,mins):
    #     ax.annotate('%s)' %j, xy=(i,j), xytext=(5,-10), textcoords='offset points')
    #     ax.annotate('(%s,' %i, xy=(i,j))#, xytext=(0,0), textcoords='offset points')
    plt.xlabel('omega')
    plt.ylabel('Chi-Squared')
    plt.savefig('{}/{}_{}{}_chisqrd.png'.format(timedatepath, quasar, intrindex, mockdex))


#############
# main code #
#############

makefiles()
write_file(intrindex)
if mockdex == 'Mock':
    mock = genmockdata(quasar)
plotdata(quasar)

