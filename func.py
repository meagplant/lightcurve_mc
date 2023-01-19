import numpy as np
import os

def mean(valset):
    return sum(valset)/len(valset)

def getdata(rung,file,mockdex):
    #this function retrieves the data from the file speicified, and sorts it into time, magnitude, and magnitude error
    #it also creates the large datasets with both image's data

    wd = wd = os.getcwd()
    
    if mockdex == 'TDC':
        rawdat = np.loadtxt('{}/TimeDelayChallenge/{}/{}.txt'.format(wd,rung,file))
    else:
        rawdat = np.loadtxt('{}/rdb/python/{}_python.csv'.format(wd,file),delimiter=',')

    time1 = rawdat[:,0]
    magA = rawdat[:,1]
    magAerr = rawdat[:,2]
    magB = rawdat[:,3]
    magBerr = rawdat[:,4]

    imgnolist = [1] * len(magAerr) + [2] * len(magBerr)

    timelist = []
    init = time1[0]
    for i in range(len(time1)):
        timelist.append(time1[i]-init)
    for k in time1:
        timelist.append(k-init)

    maglist = []
    for i in range(len(magA)):
        maglist.append(magA[i]-mean(magA))
    for j in magB:
        maglist.append(j-mean(magB))

    magerrlist = []

    for i in range(len(magAerr)):
        magerrlist.append(magAerr[i])
    for j in magBerr:
        magerrlist.append(j)

    return timelist, maglist, magerrlist, imgnolist

def reg_vals(A_m_truth, omeg_m_truth, A_l_truth, omega_l_truth, dT_truth):
    #this creates the true theta (list of inputted parameter starting values) with lensing included

    theta_truth = A_m_truth, omeg_m_truth, A_l_truth, omega_l_truth, dT_truth
    truthlist = [A_m_truth, omeg_m_truth, A_l_truth, omega_l_truth, dT_truth]
    labellist = ["ln(A_m)", "ln(omega_m)", "ln(A_l)", "ln(omega_l)", "deltaT"]

    return theta_truth, truthlist, labellist

def intrinsic_vals(A_m_truth, omeg_m_truth, dT_truth):
    #same but for only intrinsic

    theta_truth = A_m_truth, omeg_m_truth, dT_truth
    truthlist = [A_m_truth, omeg_m_truth, dT_truth]
    labellist = ["A_m", "ln(omega_m)", "deltaT"]

    return theta_truth, truthlist, labellist

def covariance(theta, timescale, magnitudeerror, imgno, intrinsic):
    #function called in emcee to create the covariance matrix, includeds both intrinsic only and full lensing setups

    n = len(magnitudeerror)
    C = np.empty((n,n))

    if intrinsic == 'Intr':

        A_m, omega_m, deltaT = theta
        expOmg = np.exp(omega_m)

        for mu in range(n):
            for nu in range(n):
                if imgno[mu] == imgno[nu]: #same image
                    if mu == nu: #same datapoint

                        valmu_l = (magnitudeerror[mu])**2 + (A_m*np.pi)/expOmg #B_m=(A_m*np.pi)/omega_m,  exp(B_l)
                        C[mu,nu] = valmu_l

                    else:

                        valmunu_l = ((A_m*np.pi)/expOmg)*np.exp(-expOmg*abs(timescale[mu]-timescale[nu]))
                        C[mu,nu] = valmunu_l

                else:

                    valmunu_m = ((A_m*np.pi)/expOmg)*np.exp(-expOmg*abs((imgno[mu]-imgno[nu])*deltaT+(timescale[mu]-timescale[nu])))
                    C[mu,nu] = valmunu_m

    else:

        A_m, omega_m, A_l, omega_l, deltaT = theta
        expAm, expOm = np.exp(A_m), np.exp(omega_m)
        expAl, expOl = np.exp(A_l), np.exp(omega_l)

        for mu in range(n):
            for nu in range(n):
                if imgno[mu] == imgno[nu]: #same image
                    if mu == nu: #same datapoint

                        valmu_l = (magnitudeerror[mu])**2 + ((expAm*np.pi)/expOm) + ((expAl*np.pi)/expOl) #B_m=(A_m*np.pi)/omega_m,  exp(B_l)
                        C[mu,nu] = valmu_l

                    else:

                        valmunu_l = ((expAm*np.pi)/expOm)*np.exp(-expOm*abs(timescale[mu]-timescale[nu]))+((expAl*np.pi)/expOl)*np.exp(-expOl*abs(timescale[mu]-timescale[nu]))
                        C[mu,nu] = valmunu_l

                else:

                    valmunu_m = ((expAm*np.pi)/expOm)*np.exp(-expOm*abs((imgno[mu]-imgno[nu])*deltaT+timescale[mu]-timescale[nu]))
                    C[mu,nu] = valmunu_m
    
    return C

def chi_sqrd_like(covariance, magnitude):
    #calculates chi-squared from covariance matrix
    n = len(magnitude)

    I = np.array(magnitude).reshape((n,1))
    It = np.transpose(I).reshape((1,n))
    sign, logdet = np.linalg.slogdet(covariance)

    #chi_sqrd = float(np.dot(It,np.linalg.solve(covariance,I)) + logdet)
    matrix_val = float(np.dot(It,np.linalg.solve(covariance,I)))

    return matrix_val+float(logdet)
