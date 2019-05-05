#coding=utf-8
#Version:python3.6.0
#Tools:Pycharm 2017.3.2
import matplotlib.pyplot as plt
import numpy as np
n = 12
X = np.arange(n)  #产生0到n-1的数组
Y1 = (1-X/float(n))*np.random.uniform(0.5,1,n)  #uniform为均匀分布
Y2 = (1-X/float(n))*np.random.uniform(0.5,1,n)
plt.bar(X,Y1,color='#9999ff',edgecolor='white')
plt.bar(X,-Y2,color='#ff9999',edgecolor='white')
for x,y in zip(X,Y1):
    #ha : horizontal alignment
    #va : vertical alignment
    plt.text(x + 0.01,y+0.01,'%.2f'%y,ha = 'center',va='bottom')
for x,y in zip(X,Y2):
    #ha : horizontal alignment
    #va : vertical alignment
    plt.text(x + 0.01,-y-0.1,'%.2f'%-y,ha = 'center',va='bottom')
plt.xlim(-1,12)
plt.xticks([])
plt.ylim(-1,1)
plt.yticks([])
plt.show()