#
def conv(ni,ri):
#
#   See Snyman (1987)
#
    tmp=1e0; c1 = float(ni-ri); c2 = float(ni+1)
    for i in range(ni):
        c1=c1+1e0; c2=c2+1e0
        tmp=tmp*c1/c2
#
    print(1e0-tmp)
#
def bay(ni,ri,a,b):
#
#   see Bolton (2004)
#
    tmp=1e0; abar=a+b-1; bbar=b-ri-1
    for i in range(1,ni+1):
        tmp=tmp*(ni+i+bbar)/(ni+i+abar)
#
    print(1e0-tmp)
#
bay(200,1,1,1000)
bay(200,2,1,1000)
bay(200,3,1,1000)
bay(200,4,1,1000)
bay(200,5,1,1000)
bay(200,10,1,1000)
#
