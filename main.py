# -*- coding: utf-8 -*-
"""
Created on Thu Jul 10 22:06:20 2025

@author: erni_
"""
import numpy as np
import random 
#import itertools
import matplotlib
import matplotlib.pyplot as plt
import time

def charge_data_serial(filename):
    time1=time.time()
    file=open(filename)
    data=[]
    N=0
    for line in file:
        [x,y]=[float(a) for a in line.split(",")]
        data.append([x,y])
        N=N+1
    time_i=time.time()
    print(f'Execution time data charging {time_i-time1:.3f} s')
    return N, np.array(data)    

def distance_point_point(p1,p2):
    
    return np.sqrt(sum((p1-p2)**2)) 


def k_means_serial(data,k,N):
    time1=time.time()
    random.seed(43)
    ps=random.sample(range(0,N),k)
    ass=np.zeros(N,int)
    sums=np.zeros((k,2),float)
    counters=np.zeros(k,int)
    
    change=1
    C=np.zeros((k,2))
    for j, p in enumerate(ps):
        #print(p)
        C[j,:]=data[p,:]
        
    while change>0:
        
        change=0
        
        for i,dato in enumerate(data):
            kmin=distanza_punto2punti(dato, C, k)
            if ass[i]!= kmin:
                change+=1
                sums[ass[i],:]-=dato
                counters[ass[i]]-=1
                sums[kmin,:]+=dato
                counters[kmin]+=1 
                ass[i]=kmin
            assert ass[i]<k
            
        
        for i in range(k):
            C[i,:]=sums[i,:]/counters[i]
            
    time2=time.time()
    print(f'Execution time serial algorithm {time2-time1:.3f} s')
    return ass

def analysis(data, ass,k):
    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(20,8))
    ax1.scatter(data[:,0],data[:,1],color="black")
    for clus in range(k):
        sel=ass==clus
        ax2.scatter(data[sel,0], data[sel,1],label=clus,edgecolors='none')
    ax2.set_xlabel('x')
    ax1.set_xlabel('x')
    ax2.set_ylabel('y')
    ax1.set_ylabel('y')
    plt.savefig('Figures/Clusters.png')
    plt.show()
    
def distanza_punto2punti(p,C,k):
    d_min=np.inf
    k_min=k
    for j,c in enumerate(C):
        d=distance_point_point(p,c)
        if d<d_min:
            d_min=d
            k_min=j
    return k_min


def main():
    
    time1=time.time()
    N, data=charge_data_serial("Resources/dati.csv")
    k=12
    ass=k_means_serial(data, k, N)
    time2=time.time()
    
    
    
    analysis(data,ass,k)
    
    print(f'Execution time main {time2-time1:.3f} s')
    return 0
    
if __name__ == '__main__':
    main()
