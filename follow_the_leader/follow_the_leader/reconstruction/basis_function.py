#!/usr/bin/env python3
from numpy.polynomial import polynomial as P
from numpy.polynomial import Polynomial
import numpy as np
import matplotlib.pyplot as plt

lut = [      [1],         
            [1,1],        
           [1,2,1],       
          [1,3,3,1],      
         [1,4,6,4,1],     
        [1,5,10,10,5,1],  
       [1,6,15,20,15,6,1]]

def binomial(n : int, k: int):
  while n >= len(lut):
    s = len(lut)
    nextRow = [1] * s
    for i in range(1, s):
      nextRow[i] = lut[s-1][i-1] + lut[s-1][i]
    nextRow.append(1)
    lut.append(nextRow)
  return lut[n][k]

def bezier_basis(n):
  basis = []
  fig, ax = plt.subplots()
  for k in range (0, n+1):
    print(f"n: {n} k: {k}")
    term1 = P.polypow([1, -1], n-k)
    term2 = P.polypow([0, 1], k)
    mult = P.polymul([binomial(n,k)], term1)
    mult2 = P.polymul(mult, term2)
    p = Polynomial(coef=mult2)
    m = p.linspace(100, [0, 1])
    ax.plot(m[0], m[1])
    basis.append(mult2)
  print(basis)
  plt.show()
  return basis

if __name__ == "__main__":
    bezier_basis(12)