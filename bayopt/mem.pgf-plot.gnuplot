set table "mem.pgf-plot.table"; set format "%.5f"
set format "%.7e";;  set samples 501; Binv(p,q)=exp(lgamma(p+q)-lgamma(p)-lgamma(q)); beta(x,p,q)=p<=0||q<=0?1/0:x<0||x>1?0.0:Binv(p,q)*x**(p-1.0)*(1.0-x)**(q-1.0); plot [x=0:1] beta(x,1,100); 
