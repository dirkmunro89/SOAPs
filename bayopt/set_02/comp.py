#
import os
import numpy as np
#
nelx=180
nely=60
#
f_0={}
with open('global.txt','r') as file:
    for line in file:
        if '*' in line[0]:
            f_0[int(line.split()[1])]=float(line.split()[2])
#
xs={}
g=0
for file in os.listdir("./"):
    filename = os.fsdecode(file)
    if 'xopt_' in filename:
        s=os.path.splitext(filename)[0].split('_')[1]
#       if s != 'g':
        xs[int(s)]=np.loadtxt(filename)
#
g=58
g_b=xs[g].reshape((nelx,nely))
x_b=xs[g]
#
f_opt=f_0[g]
#
cnt=0
for s in f_0:
    if abs(f_0[s]-f_opt)/f_opt < 0.01:
        cnt=cnt+1
print('based on 1% obj change', cnt)
#
if g==0:
    print('best solution file not found')
    exit()
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
sim=[]
fdf=[]
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
#   print(s,dff[-1],abs(TV-TVs)/TV)
    if abs(TV-TVs)/TV < 0.05:
        sim.append(s)
        fdf.append(abs(f_0[s]-f_opt)/f_opt)
#,(f_0[s]-f_opt)/f_opt])
#
#   0.05 seems to be a good measure of similarity
#
print(sorted(sim))
print(len(sim))
print(max(fdf))
