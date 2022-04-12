#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 13:27:29 2020

@author: dgc
"""
import numpy as np
from numpy import matlib
import sys
import scipy as spy
import time

from ..instrument.tas_spectr import TripleAxisSpectr
from ..instrument.tools import _scalar, _star,_modvec, _cleanargs


class FitConv(object):
    """
    """
    
    def __init__(self, exp=None, sqw=None, prefactor=None, hkle=None, Iobs=None, dIobs=None, param=None, paramfixed=None, 
                 method=1, maxiter=200, tol=1e-6, dtol=1e-6):
        
        if exp is None:
            print("exp is not provided")
            exp=TripleAxisSpectr(efixed=5)
            return

        if sqw is None:
            print("sqw is not provided")
            return

        if prefactor is None:
            print("prefactor is not provided")
            return

        self.exp       =    exp
        self.sqw       =    sqw
        self.prefactor =    prefactor
        self.hkle      =    np.array(hkle)
        self.Iobs      =    Iobs
        self.dIobs     =    dIobs
        self.param     =    param
        self.paramfixed=    paramfixed
        self.method    =    method
        self.accuracy  =    None
        self.maxiter   =    maxiter
        self.tol       =    tol
        self.dtol      =    dtol
        self.fitcount  =    0
        self.chisq     =    0
        self.alpha     =    0
        self.beta      =    0
        

    def __repr__(self):
        return " this is a FitConv class for fitting the exp by convoluting the TAS instrument resolution"
    
    def __str__(self):
        return "this is a FitConv class"
    
    @property
    def exp(self):
        return self._exp

    @exp.setter
    def exp(self, value):
        self._exp = value

    @property
    def sqw(self):
        return self._sqw

    @sqw.setter
    def sqw(self, value):
        self._sqw = value  

    @property
    def prefactor(self):
        return self._prefactor

    @prefactor.setter
    def prefactor(self, value):
        self._prefactor = value  
        
    @property
    def hkle(self):
        return self._hkle

    @hkle.setter
    def hkle(self, value):
        self._hkle = value        
        
    @property
    def Iobs(self):
        return self._Iobs

    @Iobs.setter
    def Iobs(self, value):
        self._Iobs = value      
        

    @property
    def dIobs(self):
        return self._dIobs

    @dIobs.setter
    def dIobs(self, value):
        self._dIobs = value               

    @property
    def param(self):
        return self._param

    @param.setter
    def param(self, value):
        self._param = value       

    @property
    def paramfixed(self):
        return self._paramfixed

    @paramfixed.setter
    def paramfixed(self, value):
        self._paramfixed = value
        
    @property
    def method(self):
        return self._method

    @method.setter
    def method(self, value):
        self._method = value            
        
    @property
    def maxiter(self):
        return self._maxiter

    @maxiter.setter
    def maxiter(self, value):
        self._maxiter = value    

    @property
    def tol(self):
        return self._tol

    @param.setter
    def tol(self, value):
        self._tol = value    
        
    @property
    def dtol(self):
        return self._dtol

    @dtol.setter
    def dtol(self, value):
        self._dtol = value            
    
    @property
    def fitcount(self):
        return self._fitcount

    @fitcount.setter
    def fitcount(self, value):
        self._fitcount = value

    @property
    def chisq(self):
        return self._chisq

    @chisq.setter
    def chisq(self, value):
        self._chisq = value    
        
    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        self._alpha = value    
        
    @property
    def beta(self):
        return self._beta

    @beta.setter
    def beta(self, value):
        self._beta = value    
    
    
    def fitwithconv(self, exp=None, sqw=None, prefactor=None, hkle=None, Iobs=None, dIobs=None, param=None, paramfixed=None, 
                 method='fix', maxiter=200, tol=1e-6, dtol=1e-6):

        
        t              =    time.time()
        self.exp       =    exp
        self.sqw       =    sqw
        self.prefactor =    prefactor
        
        if hkle is not None and not isinstance(hkle, np.ndarray):
            self.hkle  =    np.array(hkle)

        if  Iobs is not None and not isinstance(Iobs, np.ndarray):
            self.Iobs  =    np.array(Iobs)
            
        if  dIobs is not None and not isinstance(dIobs, np.ndarray):
            self.dIobs =    np.array(dIobs)
        
        if  param is not None:
            self.param =    np.array(param)  
           
        if paramfixed is not None:
            self.paramfixed = np.array(paramfixed)
        else:
            self.paramfixed = np.ones(self.param.shape)

        self.method     =    method
        self.accuracy   = None
        self.maxiter    =    maxiter
        self.tol        =    tol
        self.dtol       =    dtol
        #check defaults and initial conditions
        self.fitcount   =   0
        
        
        #print(self.hkle)
        [H, K, L, W] = self.hkle
        #[dataLength, H, K, L, W, EXP, Iobs] = tools._cleanargs(H, K, L, W, self.exp, self.Iobs)
        dataLength=len(H)

        if  dIobs  is  None:
            dIobs=np.ones(H.size)
        dIobs=dIobs.reshape(-1, 1)

        if not dataLength == dIobs.size:
            print('Fata error: size mismatch between H,K,L,W or EXP and dIobs')
            
 
        if self.method is None:
            self.method='fix'                                   #warning('Using default "fix2" integration method')
        
        if self.accuracy is None:
            self.accuracy=np.array([7, 0])                      #warning('Using default density of point sampling: ACCURACY=[7 0]')
        
        #if nitmax is None:
        nitmax = 200                                            #warning('Using default limit on number of iterations: nitmax=20') 
        
        if tol is None:
            tol=0.001
            self.tol=tol                                        #warning('Using default tolerance: tol=0.001') 
        
        if dtol is None:
            dtol=1e-5
            self.dtol=dtol                                      #warning('Using default tolerance: dtol=1e-5')


        dpa      =      np.zeros(len(self.paramfixed))          #default error is zero (applies to fixed parameters)
        ivar     =      np.where(self.paramfixed>0)[0]          #list of elements corresponding to varying parameters
        DF       =      len(self.Iobs)-len(ivar)                #degrees of freedom
        lamda    =      0.001                                   #initial Marquardt parameter
        nit      =      0                                                   #initial number of iterations
        chisq_old=sys.float_info.max                            #initial chisq_old huge to ensure iteration


        dispia   =      np.where(self.paramfixed > 0, 1, 0) 
        
        print("Fitting {} to {} data points with {} free parameters and {} fixed parameters\n".format(sqw, len(self.Iobs),  np.sum(dispia), len(self.param)-np.sum(dispia)))

        
        #----------------------------------------------------------------------------------------------------------------
        [chisq, alpha, beta]    =    self.marqit(param,ivar)
        
        print ('Iteration #   FunCount         chi^2        lambda          time')
        # begin iterations
        while ( np.abs(chisq_old-chisq)  >  tol*chisq ):
            if ( nit >= nitmax ): 
                break

            dpa[ivar]  =  np.linalg.solve( (alpha+lamda*np.diag(np.diag(alpha))), beta )
            pt   =  param + dpa                                               # new trial parameters
            
            [chisq_trial, alpha_trial, beta_trial] = self.marqit(pt,ivar)     #get trial chi_squared
            if ( chisq_trial > chisq ):                                       #chisq increases ?
                lamda  =  10*lamda                                            #increase lamda and try again
                if ( lamda > 1e13 ):
                    chisq_old = chisq                                         #punt if lamda is stuck beyond reason
            else:                                                             #chisq decreased
                chisq_old = chisq                                             #update old chi-squared
                lamda     = lamda/10                                          #decrease lamda
                param     = pt                                                #update parameters
                nit       = nit+1                                             #call it an iteration
                [chisq, alpha, beta] = self.marqit(param,ivar)                #recalculate alpha,beta,chi-squared
                print(' {0:8d} {1:13f} {2:13f} {3:13f} {4:13f}  \n'.format(nit,self.fitcount,chisq/DF,lamda,round((time.time()-t)*10)/10))   
        
        #----------------------------------------------------------------------------------------------------------------
        # check convergence:
        if (nit >= nitmax or lamda > 1e13):
            kvg=0
        elif (lamda > 0.001):
            kvg=2
        else:
            kvg=1
        
        nargout=9
        #----------------------------------------------------------------------------------------------------------------
        # now that converged or kicked out calculate error quantities with lamda=0
        C  =  np.linalg.inv(alpha)                                            #raw correlation matrix
        dpa[ivar]  =  np.sqrt(np.diag(C))                                     #error in a(j) is sqrt(Cjj)
        chisqN  =  chisq / DF                                                 #normalized chi-squared
        CN = C/np.sqrt(np.abs(np.diag(C)*np.diag(C).reshape(-1,1)))           #normalized correlation matrix C(ij)/sqrt[C(ii)*C(jj)]
        if (nargout >=5):
            PQ = 1-spy.special.gammainc(chisq/2,DF/2)
        #----------------------------------------------------------------------------------------------------------------
        if (nargout>=8):
            sim = self.exp.ResConv(sqw=self.sqw, pref=self.prefactor, nargout=2, hkle=self.hkle, METHOD=self.method, ACCURACY=self.accuracy, p=param)
            self.fitcount  = self.fitcount + 1
   
        if (nargout==9):
            details = {"chisq": chisq,"Ndata": len(Iobs), "Npar": len(param), "Nva": len(ivar), "DF": DF, "C": C, "final_lamda": lamda}


        print('\n')
        if kvg==0:
            print('Stop: max allowed number of iterations exceeded.')
        if kvg==1:
            print('Stop: Converged normally in {} iterations.'.format(nit))
        if kvg==2:
            print('Stop: Convergence questionable.')
        print('\n')
        print('Final parameters:\n')
        self.param  = param
        
        for index, par in enumerate(param):
            print('{0:6d} \t {1:10f} {2:12f}\n'.format(index, par, dpa[index]))
            #print(dpa)
        
        return [param, dpa, chisqN, sim, CN, PQ, nit, kvg, details]



    def dfdp(self, param, ivar):

        [npoint,m] = self.hkle.shape
        #print("npoint:{},m:{}".format(npoint, m))
        dfp=np.zeros((m,len(ivar))) 
        #preallocate space for dfdp as zeros
        for i, item in enumerate(ivar):                                 #loop through varying parameters
            h    =     np.zeros(len(param))                                #initialize h
            t    =     param[item] + self.dtol*param[item]
            h[item]  =    t - param[item] 
            if param[item] == 0:                               #protect against zero values
                h[item]=1+1e-8-1
            
            pa1=param+h
            pa2=param-h
            # dgc attention to the param list            
            f1=self.exp.ResConv(sqw=self.sqw,pref=self.prefactor,nargout=2, hkle=self.hkle, METHOD=self.method,ACCURACY=self.accuracy,p=pa1)  
            self.fitcount=self.fitcount+1
            
            f2=self.exp.ResConv(sqw=self.sqw,pref=self.prefactor,nargout=2, hkle=self.hkle, METHOD=self.method,ACCURACY=self.accuracy,p=pa2)
            self.fitcount=self.fitcount+1

            dfp[:,i]=(f1-f2)/(2*h[ivar[i]])
            
        return dfp

#================================================================================================================

    def marqit(self, param, ivar):

        f = self.exp.ResConv(sqw=self.sqw,pref=self.prefactor, nargout=2, hkle=self.hkle, METHOD=self.method, ACCURACY=self.accuracy,p=param)

        wdiff  =  (self.Iobs-f)/self.dIobs                                 # weighted difference elementwise
        chisq  =  np.sum(wdiff**2)                                         # chi_squared
                                        
        dfp    =  self.dfdp(param, ivar)
        NP     =  len(ivar)                                                # number of varying parameters
        beta   =  np.sum(np.matlib.repmat(wdiff/self.dIobs,NP,1)*dfp.T, 1)
        alpha  =  dfp.T/np.matlib.repmat(self.dIobs,NP,1)                  #normalize derivative by sigma    
                                              
        alpha  =  np.matmul(alpha, alpha.T)
        
        return [chisq, alpha, beta]
