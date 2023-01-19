import sys
import os
from func import reg_vals, intrinsic_vals

def get_vars(filename):
    #this function extracts the data from the temporary file to be used in main.py and plotting.py

    wd = os.getcwd()
    
    #reading the file:
    outputfile = open('{}/{}.txt'.format(wd,filename),'r')
    lines = outputfile.readlines()
    lines = [item.replace("\n", "") for item in lines]
    
    #extracting pathways:
    datepath, timedatepath, filedatepath = [lines[i] for i in range(3)]
    
    #getting variables used regardless of intrisic/lensing run
    N = float(lines[3])
    quasar = lines[4]
    intrindex = lines[5]
    mockdex = lines[6]

    #getting intrinsic variables
    if intrindex == 'Intr':

        stringvals = [lines[i] for i in range(7,10)]

        vals = []
        for i in range(len(stringvals)):
            val = stringvals[i]
            vlist = val.split(':')
            vals.append(float(vlist[1]))

        A_m_true, omeg_m_true, dT_true = vals[:]

        theta_true, truths, labels = intrinsic_vals(A_m_true, omeg_m_true, dT_true)
    
    #getting lensing variables
    else:

        stringvals = [lines[i] for i in range(7,12)]

        vals = []
        for i in range(len(stringvals)):
            val = stringvals[i]
            vlist = val.split(':')
            vals.append(float(vlist[1]))

        A_m_true, omeg_m_true, A_l_true, omeg_l_true, dT_true = vals[:]

        theta_true, truths, labels = reg_vals(A_m_true, omeg_m_true, A_l_true, omeg_l_true, dT_true)
    
    return datepath, timedatepath, filedatepath, N, quasar, intrindex, mockdex, theta_true, truths, labels
