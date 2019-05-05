#coding=utf-8
#Version:python3.6.0
#Tools:Pycharm 2017.3.2
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-1,1,50)#从(-1,1)均匀取50个点
y=x**2
y1=x*2
#显示第一条图像，需要定义一个对象figure，即一个对象
plt.figure(num=1)
#简单的使用
p2,=plt.plot(x,y,color='blue',linewidth = 2.5)
p1,=plt.plot(x,y1,color='green',linewidth=2)

#计入标签
#如果允许在shadow--if True, draw a shadow behind legend
legend = plt.legend([p1,p2],["up","down"],loc=1,shadow=True)
frame = legend.get_frame()
frame.set_facecolor('r')


plt.xlim((0,2))  #在plt.show()之前添加
plt.ylim((-2,2))


#gca = ‘get current axis’
#gca指的是获取当前的4个轴
ax = plt.gca()
#设置脊梁(也就是包围在图标四周的默认黑线)
#所以设置脊梁的时候，一共有四个方位
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
#将底部脊梁作为x轴
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
#设置x轴的位置(设置底的时候依据的是y轴)
ax.spines['bottom'].set_position(('data',0))
#设置左脊梁(y轴)依据的是x轴的0位置
ax.spines['left'].set_position(('data',0))

plt.xticks(np.linspace(-1,2,5))
plt.yticks([-2,-1,0,1,2],[r'$really\ bad$',r'$b$',r'$c\ \alpha$','d','good'])
#plt.xlabel("x'lihuanyu")#x轴上的名字
#plt.ylabel("y'yamiaowen")#y轴上的名字

#加入标注
x0 = 1
y0 = 2*x0
plt.plot([x0,x0],[0,y0],linewidth = 2.5,linestyle='--')
ax.annotate('lihuanyu', xy=(x0, y0), xytext=(x0+0.2, y0+0.3),xycoords='data',
            arrowprops=dict(facecolor='green',connectionStyle="arc, rad=1"))

plt.show()

