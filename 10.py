#coding=utf-8
#Version:python3.6.0
#Tools:Pycharm 2017.3.2
import  matplotlib.pyplot as plt
import numpy as np
def f(x,y):  #高度函数
    return (1-x/2+x**5+y**3) * np.exp(-x**2-y**2)
n=12
x =np.linspace(-3,3,3.5*n)
y =np.linspace(-3,3,3.0*n)
X,Y = np.meshgrid(x,y)
Z = f(X,Y)
#显示图像
#这里的cmap='bone'等价于plt.cm.bone
plt.imshow(Z,interpolation = 'nearest',cmap = 'bone' ,origin = 'up')
#显示右边的栏
plt.colorbar(shrink = .5)
plt.show()