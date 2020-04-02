import numpy as np
import matplotlib.pyplot as plt

w3t3p5 = np.loadtxt('./mr_out/mr_xw30_T300_P5.out')
w3t3p8 = np.loadtxt('./mr_out/mr_xw30_T300_P8.out')
w3t10p5 = np.loadtxt('./mr_out/mr_xw30_T1000_P5.out')
w3t10p8 = np.loadtxt('./mr_out/mr_xw30_T1000_P8.out')
w7t3p5 = np.loadtxt('./mr_out/mr_xw70_T300_P5.out')
w7t3p8 = np.loadtxt('./mr_out/mr_xw70_T300_P8.out')
w7t10p5 = np.loadtxt('./mr_out/mr_xw70_T1000_P5.out')
w7t10p8 = np.loadtxt('./mr_out/mr_xw70_T1000_P8.out')

s1 = 'H2O = '
s2 = '%, Ts = '
s3 = 'K, Ps = '
s4 = 'Pa'

w3 = '30'
w7 = '70'
t3 = '300'
t10 = '1000'
p5 = '1.0e5'
p8 = '1.0e8'

plt.plot(w3t3p5[:,0],w3t3p5[:,1],lw=2,label=s1+w3+s2+t3+s3+p5+s4,
         ls='solid',color='xkcd:dark blue')
#plt.plot(w3t3p8[:,0],w3t3p8[:,1],lw=2,label=s1+w3+s2+t3+s3+p8+s4,
#         ls='solid')
plt.plot(w3t10p5[:,0],w3t10p5[:,1],lw=2,label=s1+w3+s2+t10+s3+p5+s4,
         ls='solid',color='red')
plt.plot(w3t10p8[:,0],w3t10p8[:,1],lw=2,label=s1+w3+s2+t10+s3+p8+s4,
         ls='solid',color='xkcd:dark orange')

plt.plot(w7t3p5[:,0],w7t3p5[:,1],lw=2,label=s1+w7+s2+t3+s3+p5+s4,
         ls='dashed',color='xkcd:dark blue')
#plt.plot(w7t3p8[:,0],w7t3p8[:,1],lw=2,label=s1+w7+s2+t3+s3+p8+s4,
#         ls='dashdot',color='k')
plt.plot(w7t10p5[:,0],w7t10p5[:,1],lw=2,label=s1+w7+s2+t10+s3+p5+s4,
         ls='dashed',color='red')
plt.plot(w7t10p8[:,0],w7t10p8[:,1],lw=2,label=s1+w7+s2+t10+s3+p8+s4,
         ls='dashed',color='xkcd:dark orange')

plt.xlim(0.5,10)

plt.legend()
plt.xlabel('Mass / Earth')
plt.ylabel('Radius / Earth')
plt.savefig('water_curves.png')
