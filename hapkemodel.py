#!/usr/bin/python
# -*- coding: utf-8 -*-
# Filename: hapke_model_v2.py

# Escrevendo em python3 e usando python2.6:
from __future__ import print_function, unicode_literals, absolute_import, division

#############################################################
#################### Shell Parameters #######################
#############################################################

def main():

    import optparse
    parser = optparse.OptionParser()
    parser.add_option('-X', dest="X", type="float")
    parser.add_option('-f', action="store",dest="fill_factor", type="float")
    parser.add_option('--Bs0', action="store", dest="Bs0", type="float")
    parser.add_option('--Bc0', action="store", dest="Bc0", type="float")
    parser.add_option('-v', action="store", dest="v", type="float")
    parser.add_option('-a', action="store", dest="a_ratio", type="float")
    parser.add_option('--hs', action="store", dest="hs", type="float")
    parser.add_option('--hc', action="store", dest="hc", type="float")
    parser.add_option('-w', action="store", dest="w", type="float")
    parser.add_option('--gsca', action="store", dest="gsca", type="float")
    parser.add_option('-c', action="store", dest="c", type="float")
    parser.add_option('--refr', action="store", dest="refr", type="float")

    par, remain = parser.parse_args()

    X       = par.X
    f       = par.fill_factor
    Bs0     = par.Bs0
    Bc0     = par.Bc0
    v       = par.v
    a_ratio = par.a_ratio
    hs      = par.hs
    hc      = par.hc
    w       = par.w
    gsca    = par.gsca
    c       = par.c
    refr    = par.refr

    if X is None:
        return {'Bs0':Bs0,'Bc0':Bc0,'hs':hs,'hc':hc,'w':w,'gsca':gsca,'fill_factor':f,"c":c}
    else:
        return {'X':X,'fill_factor':f,'Bs0':Bs0,'Bc0':Bc0,'v':v,'a_ratio':a_ratio,'w':w,"c":c}

#############################################################
#################### Fortran Compiler #######################
#############################################################

def fcompile():
    from platform import system
    
    if system() == 'Linux':
       import subprocess as sp
       import os

       fnull = open(os.devnull, 'w')
       if sp.call(["ls","bhmie_fortran.so"],stdout=fnull,stderr=fnull) == 2:
          sp.call(["f2py","-c","-m","bhmie_fortran","bhmie.f"],stdout=fnull,stderr=fnull)
          fnull.close()

############################################################
############# Sum of square deviations #####################
############################################################

def ssd(func, guess, obs, ph):
    
    res = lambda par, obs, ph: abs(obs - func(par, ph))
    sol,cov,infodict,mesg,ier = opt.leastsq(res, guess, args=(obs, ph), full_output=True)
       
    return sol
 
class HapkeModel:

  __version__ = 2.00

  '''
   #############################################################
   #########  Written by:  Pedro Henrique Hasselmann  ##########
   #########  Last modified:  March, 2013             ##########
   #########  Version: 2.00                           ##########
   #############################################################
                          PYTHON 2.7.2
   #############################################################
   ################ HAPKE MODEL (Hapke 2008) ###################
   #############################################################

   # r(i,e,g) = K*(w/4*pi)*(m0/m0+m)*[p(g)*Bsh(g) + Bcb(g)*M(m0/K,m/K)]

   # Functions:
   # p(g)    --> Particle Scattering function
   # Bsh(g)  --> Shadow-Hiding function
   # Bcb(g)  --> Coherent Backscattering function
   # M(m0,m) --> Multiple scattering term

   # Input:
   # g        --> phase angle
   # Bs0, Bc0 --> Opposition effect amplitudes
   # w        --> single-scattering albedo
   # f        --> filling factor (1 - porosity)
   # X        --> Size Parameter
   # gsca     --> Asymmetry parameter
   # Qsca     --> Particle Scattering Efficiency
   # v        --> expoent of the particle size distribution
   # a_ratio  --> ratio between largest and smallest size in the
   # particle distribution

   # Output:
   r --> Surface element reflectance
   refl --> Integrated reflectance of a range of phase angles
   
   #############################################################
   ###################### Requirements #########################
   #############################################################
   numpy > 1.5
   scipy > 0.8
   scikits.scattpy
   
  '''

#############################################################
################### GLOBAL MODULES ##########################
#############################################################

  from numpy import array, linspace, arange, log, sqrt, exp, sum, sin, cos, tan, indices
  from numpy import vectorize as vec
  from math import pi
  from scipy.special import legendre as P # Legendre Polynomials
  from scipy.integrate import quad, simps
  from collections import deque

  global deque, pi, vec, array, linspace, arange, P, quad, simps, \
         log, sqrt, sum, exp, sin, cos, tan, indices

  global TINY, order, series
  TINY = 1e-20
  order  = 7 # Legendre Polynomial Order

  series = lambda func, o: array(map(func, o))

  fcompile()

#############################################################
########################## MODULES ##########################
#############################################################

  def __init__(self, phase_range,**args):

      global w, z
      w = args["w"]

      if __name__ == '__main__':
         #print("Hapke Model 2008 is Main.")
         args = main()

      else:
         pass #print("Hapke Model 2008 is Imported.")

      from numpy import radians

      # Convert phase_range to array.
      if isinstance(phase_range,list) == False:
         g = radians(arange(phase_range,phase_range+TINY,1))
      else:
         if phase_range[1] > 5e0:
            g = radians(list(arange(phase_range[0],5e0,0.25*phase_range[2])) + list(arange(5e0,phase_range[1],phase_range[2])))
         else:
            g = radians(arange(phase_range[0],phase_range[0],phase_range[2]))

      self.g    = g
      self.args = args
      self.args["refr"] = 1.45

      # Set the asymetric parameter:
      if args.has_key("X"):
         Qsca, Qe, Qback, gsca = self.Sph_Sca_mie(self.args["X"],self.args["refr"],50)
         
         self.args["Qsca"] = Qsca
         self.args["gsca"] = gsca
      else:
            gsca = args["gsca"]


      # Variable z used in the Opposition effect formulae: 
      z = tan(0.5*g) 

      # r0 term
      self.args["r0"] = (1e0 - sqrt(1e0 - w))/(1e0 + sqrt(1e0 - w))

      # Explicity Porosity
 
      fi = args["fill_factor"]**(2e0/3e0)
      self.args["K"]  = - log(1 - 1.209*fi)/1.209*fi

      #print("Parameters: ",self.args)

#############################################################
######################### Angles ############################
#############################################################

  def SphericalBody(self,phase,N=40):

      from numpy import outer

      # Body Latitude
      La = linspace(-0.5*pi, 0.5*pi, N)

      # Body Longitude
      Lo = linspace(- 0.5*pi, 0.5*pi - phase, N)

      # Observed and Incident Angles
      m0 = outer( cos(Lo + phase), cos(La) )
      m  = outer( cos(Lo), cos(La) )
      
      dA =  cos(La)* (pi/N) * (pi - phase)/N
      
      return m, m0, dA
      

#############################################################
############### Multiple-scattering term ####################
#############################################################

  def M(self,m,m0,b):
      
      from numpy import nditer, ones
      from collections import deque
      
      r0  = self.args["r0"]
      K   = self.args["K"]

      # Ambartsumian–Chandrasekhar H -function - second-order aproximation:
      H_o2 = vec( lambda x: 1e0/(1e0 - w*x*(r0 + 0.5*(1e0 - 2e0*r0*x)*log((1e0 + x)/(x + TINY)) )) )

      # Double Factorial
      fac = vec( lambda n: reduce(int.__mul__,range(n,0,-2)) )

      # Odd coeficient:
      A = vec( lambda n: (1/n)*(fac(n)/fac(n+1))*(-1)**(0.5*(n+1)) )

      # Pn --> Legendre polynomials
      o = arange(1,order,2)  # odd

      A_b  = A(o)*b
      iP   = 1e0 - sum(A_b*A(o))

      P_m0 = ones(m0.shape)
      P_m  = ones(m0.shape)

      for i, n in enumerate(o):
          P_m0 = P_m0 + A_b[i]*P(n)(m0)
          P_m  = P_m  + A_b[i]*P(n)(m) 
          
      return P_m0*(H_o2(m/K) - 1e0) + P_m*(H_o2(m0/K) - 1e0) + iP*(H_o2(m/K) - 1e0)*(H_o2(m0/K) - 1e0)

#############################################################
############ Spherical Scattering Cross-Section #############
#############################################################

  def Sph_Sca(self,a,wv,m):
      import scikits.scattpy as scat
  
      S = scat.Sphere(2*pi*a/wv,m)
      LAB = scat.Lab(S, alpha=pi/4)
      RES = scat.ebcm(LAB)
      Csca_tm, Qsca_tm = scat.LAB.get_Csca(RES.c_sca_tm)
      return Csca_tm, Qsca_tm

  def Sph_Sca_mie(self,x,refr,o=50):
      from bh_mie import bhmie

      s1,s2,qext,qsca,qback,gsca = bhmie(x,refr,o)

      return qsca, qext, qback, gsca

#############################################################
################ Single-scattering function #################
#############################################################

# Rayleigh Scatterers
  def p_ray(self,g):

      pg = vec(lambda x : 1 + 0.5*P(2)(cos(x)))

      # Asymmetric Factor:
      qsi = quad(lambda x: pg(x),0,pi)

      b2 = 5e0*qsi**2

      return pg(g) , b2 , qsi

# Double Henyey-Greenstein scattering function:

  def p_2hg(self, g, gsca, c):
      odd = arange(1,order,2)
      even = arange(2,order+1,2)

      term = lambda n: (2e0*n + 1e0) * P(n)(cos(g)) * gsca**n

      pg = 1e0 + sum( series(term, even), axis=0)  - c*sum( series(term, odd), axis=0)
      
      b_term = lambda n: c*(2e0*n + 1e0)*gsca**n
      b = series(b_term, odd)

      return  pg, b

#############################################################
################### Opposition effect #######################
#############################################################

# shadow-hiding opposition effect coeficient:

# Hapke (1993)
  def Bsh(self,Bs0,**args):

      if args.has_key("v") and args.has_key("fill_factor"):
         v = args["v"]
         fill_factor = args["fill_factor"]
         a_ratio = args["a_ratio"]

         # Regolith Size Distribution
         if v==None: N = sqrt(3e0/8)          # K*a*exp(-a/a_mean)

         # K*a**(-n)
         if v==0: N = 4/(3*sqrt(3e0))                                      # n = 0
         if v==1: N = 3e0/(sqrt(8e0*log(a_ratio)))                         # n = 1
         if v==2: N = 2e0*sqrt(1e0/a_ratio)                                # n = 2
         if v==3: N = sqrt(2e0)*(log(a_ratio)**(3e0/2))*(1e0/a_ratio)      # n = 3
         if v==4: N = sqrt(3e0)/log(a_ratio)                               # n = 4
         if v==5: N = 1e0/sqrt(2e0)                                        # n = 5

         f = 1.209 * fill_factor**(2e0/3e0)

         hs = (-3e0/8) * N * log( 1 - f ) * (fill_factor/f)
         
         return 1e0 + Bs0/(1e0 + z/hs)

      elif args.has_key("hs"):
         hs = args["hs"]

      return 1e0 + Bs0/(1e0 + z/hs)
  
# Coherent backscattering opposition effect coeficient:

# Akkermans et al. (1988)
  def Bcb(self,Bc0,**args):

      if args.has_key('X'):
         f = args["fill_factor"]
         X = args["X"]
         Qsca = args["Qsca"]
         gsca = args["gsca"]

      # Mishchenko (1992)
         hc = 3e0*f*Qsca*(1 - gsca)/(8e0*X)

      elif args.has_key("hc"): 
         hc = args["hc"]

      x = z/hc

      return 1e0 + Bc0*0.5*(1e0 + (1e0 - exp(-x))/(x + TINY))/(1e0 + x)**2
  

#############################################################
############# Integral whole-disk reflectance ###############
#############################################################

  def Aw(self,g,Bs0,Bc0,**args):

      # Phase-angle dependents:
      pg, b  = self.p_2hg(g,args["gsca"],args["c"])
      Bs     = self.Bsh(Bs0,**args)
      Bc     = self.Bcb(Bc0,**args)
      Kw_4pi = args["K"]*w/(4*pi)

      # Helfenstein & Sheppard 2011
      
      # Integral of the Lommel-Seelinger term:
      # Aw1 = K*(w/4*pi)*Bs(g)*pg(g)*integral( m0/(m0 + m) * m * dA)

      # Integral of Multi-scattering Term:
      # Aw2 = K*(w/4*pi)*Bc(g)*integral( m0/(m0 + m) * M * m * dA)

      iMsca = deque()
      iLS   = deque()
      for phase in g:
          m, m0, dA = self.SphericalBody(phase)
          iLS.append( simps( simps( (m0*m/(m0 + m))*dA, axis=1)) )
          iMsca.append( simps( simps( (m0*m/(m0 + m))*self.M(m,m0,b)*dA, axis=1)) )

      Aw1 = Bs*pg*array(iLS)
      Aw2 = Bc*array(iMsca)

      return Kw_4pi*(Aw1 + Aw2)

########################################################################

      # Lommel-Seelinger equation in Lat & Lon coordinates:
      
      #LSE = lambda Lon, a: (cos(a) - sin(a)*tan(Lon)) * (0.5*cos(2e0*Lon) + 0.5) / (1e0 + cos(a) - sin(a)*tan(Lon))

      #icosb = sin(0.5*pi - g) + 1

      #iLSE = array([quad(LSE, -0.5*pi, 0.5*pi, args=(phase,))[0] for phase in g])

# END OF METHOD
