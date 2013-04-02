#!/usr/bin/python
# -*- coding: utf-8 -*-

# Escrevendo em python3 e usando python2.6:
from __future__ import print_function, unicode_literals, absolute_import, division

from numpy import exp, arange, degrees, array, sum,std,  sqrt, median, polyfit, polyval, abs, log2
import scipy.interpolate as intp
import scipy.optimize as opt
import matplotlib.pyplot as plt
from collections import deque

# phase functions:
func_kaa = lambda p, x : p[0]*exp(-x/(float(p[1]) + 1e-20)) + p[2] + p[3]*x
func_sch = lambda p, x:  p[1] - p[0]/(1e0 + x + 1e-20) + p[2]*x


# Hapke Model
def Hapke(self,phase_range,**args):

    import hapke_model_v2 as hapke
   
    h  = hapke(phase_range,**args)

    # Make a phase curve with Hapke Model
    r = iRefl(h.g,h.args)
    ph = degrees(h.g)
       
    return r, ph


# Sum of square deviations
def ssd(guess, obs, ph, actual):
    
    res = lambda p, obs, ph: abs(obs - func_kaa(p, ph))
    sol,cov,infodict,mesg,ier = opt.leastsq(res, guess, args=(obs, ph), full_output=True)
       
    return sol, (sol - actual)**2

#######################################################################################
####################### Class: Best Number of Point Finder ############################
#######################################################################################


class BestNPoints:

   def __init__(self):
       pass

   def MakeCurve(self,r,ph,**arg):

       self.I = r
       self.ph = ph

       # Fit
       self.spline = intp.UnivariateSpline(self.ph,self.I, k=5, s=0)

       # Linear part of the phase curve
       self.linear = polyfit(arange(12e0,35e0,1e0),self.spline.__call__(arange(12e0,35e0,1e0)),deg=1)

   def RandomSet(self,loc,N,SNR):
       from random import uniform, sample, gauss
 
       spline = self.spline

       phase_rand =  sample(arange(loc[0],loc[1],loc[2]),N)
       #phase_rand = array(loc)

       refl_rand = spline.__call__(phase_rand)

       #se = sqrt(2)/float(SNR)

       # Reflectances in intensity
       observations = array([gauss(each,each/float(SNR)) for each in refl_rand])

       return observations, phase_rand

   def Compare(self,**arg):

       I = self.I
       ph = self.ph

       y = arg["y"]

       if arg.has_key("par_abs"): par = arg["par_abs"]
       if arg.has_key("compare"): comp = arg["compare"]       
      

       plt.plot(ph,I,label="Actual curve")
       
       if arg.has_key("x"):  
          plt.plot(arg["x"],y,'bo',label="simulated points")
          plt.plot(ph, func_kaa(comp, ph),label="ajusted phase function (simulated points)")

       plt.legend(loc=0)

       #ax = plt.gca()
       #ax.set_ylim(ax.get_ylim()[::-1])
       plt.show()

   def SeeCurve(self):

       angles = self.ph
       I      = self.I

       plt.plot(angles,I,"b*",label="points")
       plt.plot(angles,self.spline.__call__(angles),label="spline")
       plt.plot(angles,polyval(self.linear,angles),label="linear part")

       plt.legend(loc=0)

       #ax = plt.gca()
       #ax.set_ylim(ax.get_ylim()[::-1])

       plt.show()

   def Dist(self,*entry):
    
       # last inputs must be the labels
       
       n =  len(entry)
       
       plt.figure(figsize=(10,8),dpi=60)
       plt.title("N = 15 | SNR = 50")

       for i in reversed(range(int(n/2))):
           plt.hist(entry[i], bins=2*log2(entry[i].size + 1), label=entry[n - i - 1 ], \
           histtype='step',linewidth=2, normed=True, range=(0,1))
       
       plt.legend(loc=0)
       plt.ylabel("$f$")
       plt.xlim(0,1.5)
       plt.xlabel("residuo")
       
       plt.show()

# END
