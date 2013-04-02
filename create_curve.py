#!/usr/bin/python
# -*- coding: utf-8 -*-

# Escrevendo em python3 e usando python2.6:
from __future__ import print_function, unicode_literals, absolute_import, division

from numpy import arange, polyfit, polyval, degrees, log10, exp
import scipy.optimize as opt
import matplotlib.pyplot as plt

# phase functions:
func_kaa = lambda p, x : p[0]*exp(-x/(float(p[1]) + 1e-20)) + p[2] + p[3]*x

# Sum of square deviations
def ssd(guess, obs, ph):

    res = lambda p, obs, ph: abs(obs - func_kaa(p, ph))
    sol,cov,infodict,mesg,ier = opt.leastsq(res, guess, args=(obs, ph), full_output=True)
       
    return sol

class PhaseCurve:

   def __init__(self):
      from hapke_model_v2 import main
      if __name__ == '__main__':
           print("Main.")
           
           self.args = main()

      else:
           print("Imported.")

   def MakeCurve(self, phase_range,**args):
    
       if len(args) is  0: args = self.args
    
       from hapke_model_v2 import HapkeModel as hapke
   
       h  = hapke(phase_range,**args)

       # Make a phase curve with Hapke Model
       r =  h.Aw(h.g,**h.args)

       self.r = r/r[0]
       self.ph = degrees(h.g)

   def Parameters(self):

       angles = self.ph
       r = self.r

       import scipy.interpolate as intp

       # Fit
       self.spl_curve = intp.UnivariateSpline(angles,r, k=5, s=0)

       # Linear part of the phase curve
       self.linear = polyfit(arange(12e0,20e0,1e0),self.spl_curve.__call__(arange(12e0,20e0,1e0)),deg=1)

   def OEpars(self):

       #Belskaya parameters

       I03I5 = self.spl_curve.__call__(0.3)/self.spl_curve.__call__(5e0)
       I03 = self.spl_curve.__call__(0.3)//polyval(self.linear,0.3)
       b = (polyval(self.linear,20e0) - polyval(self.linear,15e0))/(20e0-15e0)

       diff = lambda y: abs( self.spl_curve.__call__(y) - polyval(self.linear,y) )

       
       a = 0.1
       OE_begin = 0.1
       while diff(a) > 5e-3:
             OE_begin = a
             a = a + 0.2
 
       # Kaasaleinen parameters
       kaa_sol = ssd([1e0, 1e0, 0e0, 0e0],self.r,self.ph)
 
       return I03I5, I03, b, OE_begin, kaa_sol

   def Plot(self):

       angles = self.ph
       r = self.r

       plt.plot(angles,r,"b*",label="points")
       plt.plot(angles,self.spl_curve.__call__(angles),label="spline k=5")
       plt.plot(angles,polyval(self.linear,angles),label="linear part")

       plt.legend(loc=0)
       plt.ylabel("Reflectance")
       plt.xlabel("Phase Angle (degrees)")

       #ax = plt.gca()
       #ax.set_ylim(ax.get_ylim()[::-1])

       plt.show()
       plt.clf()

if __name__ == '__main__':
   ph = PhaseCurve()
   #b  = ph.MakeCurve([0.01,20.01,3.0])
   #c  = ph.Parameters()
   #p  = ph.Plot()
   
   plt.figure(figsize=(9,10), dpi=70)
   
   # CBOE dominance
   for B in arange(0.5,0.9,0.1):
    for x in arange(0.5,3.0,0.5):
     for f in arange(0.4,0.8,0.1):

       b = ph.MakeCurve([0.01,20.01,3.0],Bs0=0.4, Bc0=B, X=x, fill_factor=f, a_ratio=1000, w=0.23, c=-0.5, v=4)
       c = ph.Parameters()
       I03I5, I03, b, OE_begin, sol = ph.OEpars()
       #p = ph.Plot()
       plt.plot(sol[3],sol[1],"bo")

   # SHOE dominance

   for B in arange(0.5,0.9,0.1):
    for x in arange(10.0,100.0,90.0):
     for f in arange(0.4,0.8,0.1):
      for a in arange(500, 1000, 100):

        b = ph.MakeCurve([0.01,20.01,3.0], Bs0=B, Bc0=0.4, X=x, fill_factor=f, a_ratio=a, w=0.23, c=-0.5, v=4)
        c = ph.Parameters()
        I03I5, I03, b, OE_begin, sol = ph.OEpars()
        plt.plot(sol[3],sol[1],"ro")

   #plt.xlim(0,0.08)
   plt.ylim(0,8)

   plt.xlabel("$k$")
   plt.ylabel("$d$")
   plt.show()

