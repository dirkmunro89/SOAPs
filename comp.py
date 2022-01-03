#
import os
import numpy as np
#
nelx=180
nely=60
#
xs={}
for file in os.listdir("./"):
    filename = os.fsdecode(file)
    if 'xopt_' in filename:
        s=os.path.splitext(filename)[0].split('_')[1]
        if s != 'g':
            xs[int(s)]=np.loadtxt(filename)
        else:
            x_b=np.loadtxt(filename)
            g_b=x_b.reshape((nelx,nely))
#
TV=0e0
for i in range(nelx-1):
    for j in range(nely-1):
        TV=TV+abs( g_b[i][j] - g_b[i][j+1] )
        TV=TV+abs( g_b[i][j] - g_b[i+1][j] )
#
print('---')
print('TV of global', TV)
print('---')
#
dff=[]
for s in xs:
    x_c=xs[s]
    g_c=x_c.reshape((nelx,nely))
    dff.append(np.linalg.norm(x_b-x_c))
#
    TVs=0e0
    for i in range(nelx-1):
        for j in range(nely-1):
            TVs=TVs+abs( g_c[i][j] - g_c[i][j+1] )
            TVs=TVs+abs( g_c[i][j] - g_c[i+1][j] )
#
    print(s,dff[-1],abs(TV-TVs)/TV)
#
