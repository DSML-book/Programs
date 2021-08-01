"""" accrejgamma.py """
from math import exp, gamma, log
from numpy.random import rand

alpha = 1.3
lam = 5.6
f = lambda x: lam**alpha * x**(alpha-1) * exp(-lam*x)/gamma(alpha)
g = lambda x: 4*exp(-4*x)
C = 1.2
found = False
while not found:
   x = - log(rand())/4
   if C*g(x)*rand() <= f(x):
      found = True
print(x)