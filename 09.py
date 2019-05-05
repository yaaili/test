#coding=utf-8
#Version:python3.6.0
#Tools:Pycharm 2017.3.2
import matplotlib.pyplot as plt
import numpy as np
#计算x,y,对应的坐标值
def f(x,y):  #高度函数
    return (1-x/2+x**5+y**3) * np.exp(-x**2-y**2)

#生成X,Y的数据
n = 256
x = np.linspace(-3,3,n)
y = np.linspace(-3,3,n)
# 把x,y数据生成mesh网格状的数据，因为等高线的显示是在网格的基础上添加上高度值
X,Y = np.meshgrid(x,y)

# 填充等高线
#use plt.contourf to filling contours
#X Y and value for (X,Y) point
#这里的8就是说明等高线分成多少个部分，如果是0则分成2半
#则8是分成10半
#cmap找对应的颜色，如果高=0就找0对应的颜色值，
plt.contourf(X,Y,f(X,Y),cmap = plt.cm.hot)

C =plt.contour(X,Y,f(X,Y),8,colors="black",linewidth = .5)
#显示图表
plt.clabel(C,inline=True, fontsize=12)

plt.show()


