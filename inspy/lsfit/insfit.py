#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tuesday Aug 17 13:27:29 2021

This has not yet finished. This attempts to finish the log.

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
    
    
    
    
    
    def InsPeakConv(self, exp, parameters, hkle, Iobs):
        
        #pSQW='SqwDemo'
        #pPref='PrefDemo2'
        #pMethod=parameters['METHOD']
        #pAcc=parameters['ACCURACY']
        pEn1=float(parameters['En1'])
        pEn2=float(parameters['En2'])
        pRatio=float(parameters['IntRatio'])
        pGamma1=float(parameters['Gamma1'])
        pGamma2=float(parameters['Gamma2'])
        
        pInt=float(parameters['Int'])
        pBg=float(parameters['Bg'])
        
        pTemp=float(parameters['Temp'])
        
        #param=[pEn1, pEn2, pRatio, pGamma1, pGamma1, pInt, pBg, pTemp]
        param=[pEn1, pEn2, pRatio, pGamma1, pGamma2, pInt, pBg, pTemp]
        
        #pSeed=[1.7,3.5, 3, 5, 1, 200, 10, 1.5]
        



        Exp=npy.TripleAxisSpectr(efixed=5)
        Exp.method = 1  # 1 for Popovici, 0 for Cooper-Nathans
        Exp.moncor=0

        Exp.efixed = 5
        Exp.infin=-1    #const-Ef
        
        
        Exp.mono.dir =-1
        Exp.ana.dir  =-1
        
        Exp.mono.tau     = 'PG(002)'
        Exp.mono.mosaic  = 30
        Exp.mono.vmosaic = 30
        Exp.mono.height  = 10            #no need for /sqrt(12)
        Exp.mono.width   = 10
        Exp.mono.depth   = 0.2
        Exp.mono.rh      = 100
        Exp.mono.rv      = 100
        
        Exp.ana.tau = 'PG(002)'
        Exp.ana.mosaic = 30
        Exp.ana.vmosaic = 30
        Exp.ana.height = 10
        Exp.ana.width = 10
        Exp.ana.depth = 0.2
        Exp.ana.rh=100
        Exp.ana.rv=100
        
        #Put the sample information below
        Exp.sample.a = 4
        Exp.sample.b = 5
        Exp.sample.c = 6
        Exp.sample.alpha = 90
        Exp.sample.beta = 90
        Exp.sample.gamma = 90
        Exp.sample.mosaic=60
        Exp.sample.vmosaic=60
        Exp.sample.u=np.array([1, 0, 0])
        Exp.sample.u=np.array([0, 1, 0])
        Exp.sample.shape_type='rectangular'
        Exp.sample.shape = np.diag([0.6, 0.6, 10])**2
        
        
        Exp.hcol = [60, 60, 60, 60]
        Exp.vcol = [120, 120, 120, 120]
        Exp.arms = [200, 200, 200, 40,160]


        Exp.orient1 = np.array([1, 0, 0])
        Exp.orient2 = np.array([0, 1, 0])
        
        Exp.guide.height=15
        Exp.guide.width=5
        
        Exp.detector.height=15
        Exp.detector.width=2.5
        
        
        #[H,K,L,W]=hkle
        #print(len(W))
        conv=Exp.ResConv(SqwDemo,PrefDemo,nargout=2,hkle=hkle,METHOD='fix', ACCURACY=[5,5], p=param)
        
        
        return conv-Iobs
    
    
    
    
    
    def lmfitwithconv(self, exp=None, sqw=None, prefactor=None, hkle=None, Iobs=None, dIobs=None, param=None, paramfixed=None, 
                 method='fix', maxiter=200, tol=1e-6, dtol=1e-6):

        fitparams=Parameters()

        fitparams.add('En1',         value=1.616,     min=0,       max=2.5,    vary=True)
        fitparams.add('En2',         value=3.307,     min=2.5,     max=4,      vary=True)
        fitparams.add('IntRatio',    value=3.2907,    min=0.001,   max=1000,   vary=True)
        fitparams.add('Gamma1',      value=0.0180,    min=0.001,   max=1,      vary=True)
        fitparams.add('Gamma2',      value=0.002,     min=0.001,   max=1,      vary=True)
        fitparams.add('Int',         value=5000000,  min=0,       max=1000000000,  vary=True)
        fitparams.add('Bg',          value=0,         min=0,       max=0.2,     vary=True)
        fitparams.add('Temp',        value=1.5,       min=1.5,     max=1.6)

        initial=InsPeakConv(exp, fitparams, hkle, Iobs)+Iobs

        fit_kws={'maxfev':500} #maximum func call

        minner=Minimizer(InsPeakConv, fitparams, (hkle, Iobs),**fit_kws)

        result=minner.minimize()

        final=Iobs+result.residual

        print(fit_report(result))
        print(result.residual)
        
        return [param, dpa, chisqN, sim, CN, PQ, nit, kvg, details]


