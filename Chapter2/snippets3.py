import random as rd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

beta = np.matrix('10;-14;4;-0.25')
n=100
sig=5

class Learning():
    def __init__(self, n, beta, sig):
        self._n = n
        self._beta = beta
        self._sig = sig
        #self._Xmat = None
        self._y = None
        self._k = np.size(self._beta)
        self._x = None
        self._xx = None
        self._yy = None

    def generate_data(self):
        self._x=10*np.random.uniform(0,1,self._n)
        Xmat=np.matrix([np.ones(self._n, dtype=np.int), self._x, self._x**2, self._x**3] ).transpose()
        self._y=Xmat*self._beta +np.asmatrix(self._sig*np.asmatrix(np.random.normal(size=self._n))).transpose()
        #print(self._x)
        return np.asmatrix(self._x).transpose(),self._y



    def train(self):


        #beta = np.matrix('10;-14;4;-0.25')
        #sig=5
        #n=100
        x,y=self.generate_data()

        plt.scatter([x],[y])
        self._xx=np.arange(np.amin(x), np.amax(x), 0.01).transpose()
        self._yy= np.polyval(np.flip(beta, 0), self._xx)
        plt.plot(self._xx,self._yy)
        plt.show()

    def fitmodels(self):
        self.train()

        K=18
        Xmat=np.matrix([np.ones(self._n, dtype=np.int)])
        betahat = np.divide((Xmat.transpose() * Xmat),(Xmat.transpose() * self._y))
        betahats={}
        for k in range(2,K):
            Xmat = np.matrix([Xmat, self._x**(k-1)]).transpose()
            betahat = np.divide((Xmat.transpose()*Xmat),(Xmat.transpose()*self._y))
            betahats[k] = betahat
        c=np.matrix([2, 4, 26])
        for i in range(1,4):
            yy=np.polyval(np.flip(betahat, 0), self._xx)
            plt.plot(self._xx, yy)
            plt.show()

    def validate(self):
        xval, yval = self.generate_data()
        Xmat = np.matrix([np.ones(self._n, dtype=np.int)])

if __name__ == "__main__":
    learn = Learning(100, np.matrix('10;-14;4;-0.25'), 5)

    learn.train()