# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 19:47:46 2023

@author: Valdemar
"""

from python_speech_features import mfcc
from matplotlib import pyplot as plt
import numpy as np
import scipy as sp
import scipy.stats
import os
'''
Full Hidden Markov Model Class Implementation
'''
s={}
class HMM:
    
    '''
    Constructor
    '''
    def __init__(self, q, A, B, finite = False):
        self.A = A
        self.B = B
        self.q = q
        self.finite = finite
        
    '''
    Generate Observations from a HMM
    '''
    def rand(self, num):
        X = (np.zeros((num, self.B[0].random(1).shape[1])))
        S = np.zeros((num), dtype = int)
        temp = np.random.choice(range(self.q.shape[0]), 1, p=self.q)[0]
        S[0] = temp
        X[0] = self.B[temp].rand()
        num -= 1
        if num == 0:
            return X,S
        for i in range(num):
            temp = np.random.choice(range(self.A.shape[1]), 1, p=self.A[S[i]])[0]
            if temp == self.A.shape[0]: #End state
                return X,S
            S[i+1] = temp
            dist = self.B[temp]
            X[i+1] = self.B[temp].rand()
        return X,S
    
    '''
    Calculate alpha_hat values for the forward algorithm
    '''
    def alphahat(self, obs, scale = True):
        if self.finite:
            A = self.A[:,:-1]
        else:
            A = self.A
        p, scaled = prob(obs, self.B)
        if not scale:
            scaled = p
        c = np.zeros(obs.shape[0])
        alpha = np.zeros((obs.shape[0], A.shape[0]))
        temp = np.zeros(A.shape[0])
        c[0] = np.sum(self.q*scaled[0])
        alpha[0,:] = (self.q*scaled[0])/c[0]
    
        for t in range(1, obs.shape[0]):
            for j in range(A.shape[0]):
                temp[j] = alpha[t-1].dot(A[:,j]) * scaled[t, j]
            c[t] = np.sum(temp)
            alpha[t, :] = temp/c[t]
    
        if self.finite:
            variable = alpha[-1].dot(self.A[:, -1])
            c = np.append(c, np.array([variable]))
        return alpha, c
    
    '''
    Calculate beta_hat values for the backward algorithm
    '''
    def betahat(self, obs, scale = True):
        if self.finite:
            A = self.A[:,:-1]
        else:
            A = self.A
        p, scaled = prob(obs, self.B)
        if not scale:
            scaled = p
        alphas, cs = self.alphahat(obs)
        beta = np.zeros((obs.shape[0], self.A.shape[0]))
        temp = np.zeros(self.A.shape[0])
        if self.finite:
            temp = self.A[:,-1]
            temp = temp/(cs[-1]*cs[-2])
        else:
            temp = np.ones((self.A.shape[0]))
            temp = temp/cs[-1]
        beta[-1] = temp
        for t in range(obs.shape[0]-2, -1, -1):
            temp = np.zeros(A.shape[0])
            for i in range(A.shape[0]):                
                for j in range(A.shape[0]):
                    temp[i] += A[i,j]*scaled[t+1,j]*beta[t+1, j]
            beta[t] = temp/cs[t]
        return beta
    
    '''
    Viterbi algorithm for maximum likelihood sequence detection
    '''
    def viterbi(self, obs, scale = True):
        if self.finite:
            A = self.A[:,:-1]
        else:
            A = self.A
        chi = np.zeros((obs.shape[0], A.shape[0]))
        prev = np.zeros((obs.shape[0]-1, A.shape[0]), dtype=int)
        p, scaled = logprob(obs, self.B)
        if not scale:
            scaled = p
        chi[0,:] = np.log(self.q) + scaled[0]
    
        for t in range(1, obs.shape[0]):
            for j in range(A.shape[0]):
                proba = chi[t-1] + np.log(A[:, j])             
                prev[t-1, j] = np.argmax(proba)
                chi[t,j] = np.amax(proba + scaled[t,j])
        
        if self.finite:
            chi[-1] += np.log(self.A[:, -1])
        ihat = np.zeros(obs.shape[0], dtype = int)
        last = np.argmax(chi[-1, :])
        ihat[0] = last
        
        index = 1
        for i in range(obs.shape[0]-2, -1, -1):
            temp = prev[i, int(last)]
            ihat[index] = temp
            last = temp
            index += 1
        ihat = np.flip(ihat, axis = 0)
        return ihat
    
    #page 113
    def calcgammas(self, alphahats, betahats, cs, obs, uselog=False):
            gammas = []
            for i in range(obs.shape[0]):
                temp = []
                for t in range(obs[i].shape[0]):
                    if uselog:
                        temp += [np.log(alphahats[i][t])+np.log(betahats[i][t])+np.log(cs[i][t])]
                    else:
                        temp += [alphahats[i][t]*betahats[i][t]*cs[i][t]]
                gammas += [np.array(temp)]
            gammas = np.array(gammas)
            return gammas
    
    #page 130
    def calcinit(self, gammas, uselog=False):
        f = []
        for gamma in gammas:
            f.append(gamma[0])
        f=np.array(f)
        if uselog:
            return np.sum(np.exp(f), axis = 0)/np.sum(np.exp(f))
        else: 
            return np.sum(f, axis = 0)/np.sum(f)
    
    def calcabc(self, obs):
        alphahats = []
        betahats = []
        cs = []
        for i in range(obs.shape[0]):
            alph, c = self.alphahat(obs[i])
            beth = self.betahat(obs[i])
            alphahats += [alph]
            betahats += [beth]
            cs += [c]
        return alphahats, betahats, cs
    
    #page 132
    def calcxi(self, alphahats, betahats, cs, obs, uselog=False):
        xirbars = []
        xirs = []
        for i in range(obs.shape[0]): 
            if self.finite:
                xi = np.zeros((obs[i].shape[0], self.A.shape[0], self.A.shape[1]))
            else:
                xi = np.zeros((obs[i].shape[0]-1, self.A.shape[0], self.A.shape[1]))
            p, scaled = prob(obs[i], self.B)
            if uselog: 
                xi = np.log(xi)
                p, scaled = logprob(obs[i], self.B)
            for t in range(obs[i].shape[0]-1):
                for j in range(self.A.shape[0]):
                    for k in range(self.A.shape[0]):
                        if uselog:
                            xi[t, j, k] = np.log(alphahats[i][t][j])+np.log(self.A[j,k])+scaled[t+1][k]+np.log(betahats[i][t+1][k])
                        else:
                            xi[t, j, k] = alphahats[i][t][j]*self.A[j,k]*scaled[t+1][k]*betahats[i][t+1][k]
            if self.finite:
                for j in range(self.A.shape[0]):
                    if uselog:
                        xi[-1][j][-1] = np.log(alphahats[i][-1][j])+np.log(betahats[i][-1][j])+np.log(cs[i][-1])
                    else:
                        xi[-1][j][-1] = alphahats[i][-1][j]*betahats[i][-1][j]*cs[i][-1]
                
            if uselog:
                xi = np.exp(xi)
            xirs += [xi]
            xirbars += [np.sum(xi, axis = 0)]
            
        xibar = np.sum(xirbars, axis = 0)
        return xibar
    
    def printoutput(self, newmean, newcov):
        print("Estimated a:")
        print(self.q)
        print()
        print("Estimated A:")
        print(self.A)
        print()
        print("Estimated means:")
        print(newmean)
        print()
        print("Estimated covariances:")
        print(newcov)        
        
    
    '''
    Baum-Welch Algorithm for learning HMM parameters from observations
    '''
    def baum_welch(self, obs, niter, uselog=False, scaled = True):     
        A = self.A
        B = self.B
        
        for it in range(niter):
            print("Iteration: " + str(it))
            alphahats, betahats, cs = self.calcabc(obs) #from Assignment 3 and 4
            gammas = self.calcgammas(alphahats, betahats, cs, obs, uselog) #alpha*beta*c
            newpi = self.calcinit(gammas, uselog) #average of gammas[:,0]
            print("Updated initial probabilities \n")
            print(newpi)
            xibar = self.calcxi(alphahats, betahats, cs, obs, uselog) #page 132
            if uselog: 
                xibar = np.exp(xibar)
                
            newA = np.array([i/np.sum(i) for i in xibar]) #xibar/sum_k(xibar); page 130
            print("Updated transition probability matrix \n")
            print(newA)
            
            if uselog: 
                gammas = np.exp(gammas)
    
            for i in range(B.shape[0]):
                self.B[i].em(obs,gammas, i)
            
            #update all variables
            self.q = newpi
            self.A = newA
            
        
class GMM:
    mean = np.array([0])
    cov = np.array([[0]])
    def __init__(self, mus, Cs, ws):
        self.mean = mus
        self.cov = Cs
        self.w = ws
        return
    def random(self, num):
        return np.random.multivariate_normal(self.mean[0], self.cov[0], num)
    def rand(self):
        return np.random.multivariate_normal(self.mean[0], self.cov[0], 1)[0]
    def likelihood(self, X):
        pd = 0
        for i in range(self.cov.shape[0]):
                pd += self.w[i]*scipy.stats.multivariate_normal.pdf(X, mean=self.mean[i], cov=self.cov[i], allow_singular=(True))
        if pd == 0:
            pd = 10**(-100)
        return pd
    def loghood(self, X):
        return np.log(self.likelihood(X))
    def getmean(self):
        return self.mean
    def getcov(self):
        return self.cov
    def conditional(self, j, x):
        num = self.w[j]*scipy.stats.multivariate_normal.pdf(x, mean=self.mean[j], cov=self.cov[j], allow_singular = (True))
        den = self.likelihood(x)
        return num/den
    def em(self, obs, gamma, s):
        M=len(obs)
        new_w = np.zeros(self.w.shape)
        new_cov = np.zeros(self.cov.shape)
        new_mean = np.zeros(self.mean.shape)
        tot_cond_sum = 0
        for m in range(self.cov.shape[0]):
            for i in range(obs.shape[0]):
                sum_cond = 0
                sum_cond_x = 0
                num_cov = 0
                for t in range(obs[i].shape[0]):
                    conditional_i_x = self.conditional(m,obs[i][t])*gamma[i][t][s]
                    sum_cond += conditional_i_x
                    sum_cond_x += conditional_i_x*obs[i][t]
                    num_cov += conditional_i_x*np.outer((obs[i][t]-self.mean[m]), (obs[i][t]-self.mean[m]).T)
                
                new_w[m] = sum_cond
                if sum_cond > 0:
                    new_mean[m] = sum_cond_x/sum_cond
                    new_cov[m] = num_cov/sum_cond
                else:
                    new_mean[m] = self.mean[m]
                    new_cov[m] = self.cov[m]
        
        self.w = new_w/np.sum(new_w)
        self.mean = new_mean
        self.cov = (new_cov)
        print(np.sum(self.w))
                
    
def prob(x, B):
    T = x.shape[0]
    N = B.shape[0]
    res = np.zeros((T, N))
    for i in range(T):
        for j in range(N):
            res[i,j] = B[j].likelihood(x[i])
    scaled = np.zeros(res.shape)
    for i in range(scaled.shape[0]):
        for j in range(scaled.shape[1]):
            if np.amax(res[i])>0:
                scaled[i, j] = res[i,j]/(np.amax(res[i]))
    return res, scaled


def logprob(x, B):
    res, scaled = prob(x,B)
    return np.log(res), np.log(scaled)

def read(file):
    with open(file) as f:
        lines = f.readlines()
    x=[]
    for l in lines:
        xt = []
        s=''
        for c in l[12:]:
            if (c == '\t' or c == '\n') and len(s) > 0:
                if s[0] == ' ': continue
                xt.append(float(s))
                s=''
                continue
            s+=c
        x.append(np.array(xt))
    return np.array(x)
            
#Fast moving average without Fourier transform
def moving_average(a, n=50):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
      
def predict_sequence(x,hmm):
    i_hat = hmm.viterbi(x)
    S={0:'Stand', 1: 'Walk', 2: 'Run'}
    seq = []
    for i in i_hat:
        if len(seq) == 0:
            seq.append([S[i],1])
        elif seq[-1][0] != S[i]:
            seq.append([S[i],1])
        else:
            seq[-1][1] += 1
    return seq

def get_features(x, fs=100):
    t,f,Z=sp.signal.stft(x,fs=fs,nperseg=128, noverlap=64, nfft=128)
    Ps = np.abs(Z)**2
    feat = np.log10(Ps)
    feat = np.swapaxes(feat,0,1)
    feat = (feat - np.mean(feat,axis=0))/np.sqrt(np.var(feat, axis=0))
    feat = np.swapaxes(feat,0,1)
    return feat

M =10 #Amount of mixtures
#S = 1 -> stå, S=2 -> gå, S=3 -> springa
A_init = np.array([[1/2, 3/8, 1/8],
                   [1/4, 1/2, 1/4],
                   [1/8, 3/8, 1/2]])

q_init = np.array([6/7,1/7,0])

mean_init1 = np.array([np.zeros(65) +i + 1 for i in range(M)])
cov_init1 = np.array([(i+1)*np.eye(65)+.01 for i in range(M)])
mean_init2 = np.array([np.zeros(65) +2*i + 1 for i in range(M)])
cov_init2 = np.array([(i+1)*np.eye(65)+.02 for i in range(M)])
mean_init3 = np.array([np.zeros(65) +i + 1 for i in range(M)])
cov_init3 = np.array([.5*(i+1)*np.eye(65) for i in range(M)])

w_init = np.ones(M)/M
gaussian_init1 = GMM(mean_init3, .5*cov_init3, w_init)
gaussian_init2 = GMM(mean_init3, cov_init3, w_init)
gaussian_init3 = GMM(mean_init3, 1.5*cov_init3, w_init)
B_init = np.array([gaussian_init1, gaussian_init2,gaussian_init3])
D_dir = 'Database/Train'
v=np.array([])
D=[]
for filename in os.listdir(D_dir):
    a = read(D_dir +'/'+ filename)
    #a = np.genfromtxt(D_dir + '/' +filename, delimiter=';')
    #a=a[1:]
    x, y, z = moving_average((a[:,0])),moving_average(a[:,1]),moving_average((a[:,2]))
    f = np.concatenate((np.array([x]),np.array([y]),np.array([z])))
    feat_x = get_features(x)
    feat_y = get_features(y)
    feat_z = get_features(z)
    f = np.concatenate((feat_x,feat_y,feat_z), axis=1)
    f=f.T
    print(f.shape)
    D.append(f)

D=np.array(D, dtype=object)
print(D.shape)
hmm = HMM(q_init,A_init,B_init)
hmm.baum_welch(D,15)