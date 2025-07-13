# -*- coding: utf-8 -*-
"""
Created on Sat Jul 12 14:48:00 2025

@author: erni_
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time
import random
from functools import partial
import multiprocessing
from multiprocessing import Value, Array


def distance_point_point(p1,p2):
    
    return np.sqrt(sum((p1-p2)**2)) 


def k_means_serial(data,k,N,D):
    flag=0
    time1=time.time()
    random.seed(433)
    ps=random.sample(range(0,N),k)
    ass=k*np.ones(N,int)
    sums=np.zeros((k,D),float)
    counters=np.zeros(k,int)
    
    change=N
    C=np.zeros((k,D))
    for j, p in enumerate(ps):
        #print(p)
        C[j,:]=data[p,:]
        
    while change>N//100:
        
        change=0
        
        for i,dato in enumerate(data):
            kmin=distance_pont2points(dato, C)
            if ass[i]!= kmin:
                change+=1
                if flag:
                    sums[ass[i],:]-=dato
                    counters[ass[i]]-=1
                sums[kmin,:]+=dato
                counters[kmin]+=1 
                ass[i]=kmin
        
        
        for i in range(k):
            C[i,:]=sums[i,:]/counters[i]
        #analysis(data, ass, k,C)
        #print(change)
        
        flag=1
            
    time2=time.time()
    print(f'Execution time serial algorithm {time2-time1:.3f} s')
    return ass, C



def analysis(data, ass,k,C,flag=0):
    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(20,8))
    ax1.scatter(data[:,0],data[:,1],color="black")
    for clus in range(k):
        sel=ass==clus
        ax2.scatter(data[sel,0], data[sel,1],label=clus,edgecolors='none')
        ax2.scatter(C[clus,0],C[clus,1],marker='o',color="red", s=100)
    ax2.set_xlabel('x')
    ax1.set_xlabel('x')
    ax2.set_ylabel('y')
    ax1.set_ylabel('y')
    if flag:
        plt.savefig('Figures/Clusters.png')
    plt.show()

def k_means_parallel(data,k,N,pool_size,D):
    flag=0
    counters=np.zeros(N)
    time1=time.time()
    
    random.seed(433)
    ps=random.sample(range(0,N),k)
    ass=k*np.ones(N,int)
    sums=np.zeros((k,D),float)
    
    change=N
    C=np.zeros((k,D))
    
    
    
    for j, p in enumerate(ps):
        C[j,:]=data[p,:]
        
    pool = multiprocessing.Pool(processes=pool_size)
                                #,initializer=init, initargs=(counters,))
    while change>N//100:
           
        
        change=0
        
        ass_new=pool.map(partial(distance_pont2points,C=C),data)
        change=sum(ass!=ass_new)
        
        
        for i in range(N):
            if ass[i]!=ass_new[i]:
                sums[ass_new[i],:]+=data[i,:]
                counters[ass_new[i]]+=1
                if flag:
                    counters[ass[i]]-=1 
                    sums[ass[i],:]-=data[i,:]
                
                ass[i]=ass_new[i]

             
        
        C=np.array(pool.map(partial(recenter,sums,counters),range(k)))
        #analysis(data, ass, k, C)
        #print(change)
        flag=1
        
    pool.close()
    pool.join()
    
            
    time2=time.time()
    print(f'Execution time parallel algorithm {time2-time1:.3f} s')
    return ass,C



    
def distance_pont2points(p,C):
    k=len(C)
    d_min=np.inf
    k_min=k
    for j,c in enumerate(C):
        d=distance_point_point(p,c)
        if d<d_min:
            d_min=d
            k_min=j
    return k_min



def recenter(sums,counters,i):
    return sums[i,:]/counters[i]



def generate_cliusters(k,N,D):

    random.seed(11)
    np.random.seed(11)
    data=np.zeros((N,D))
    S=np.random.random((k,D))
    
    for i in range(N):
        p=random.randint(0,k-1)
        data[i,:]=S[p,:]+np.random.normal(0,0.02,D)
        
        
    return data

def k_means_parallel_2(data,k,N,pool_size,D):
    change=Value("i",N)
    time1=time.time()
    
    random.seed(433)
    ps=random.sample(range(0,N),k)
    ass=k*np.ones(N,int)
    
    C=np.zeros((k,D))
    
    
    for j, p in enumerate(ps):
        C[j,:]=data[p,:]
        
    pool = multiprocessing.Pool(processes=pool_size)
    while change.value>N//100:
           
        pool.apply_async(reset)
        
        ass_new=pool.map(partial(distance_pont2points,C=C),data)
        
        for i in range(N):
            if ass[i]!=ass_new[i]:
                pool.apply_async(increment)
        
        
    
        ass=np.copy(ass_new)
        
        
        
        C=np.array(pool.map(partial(recenter_2,data,ass),range(k)))
        
        
        
    pool.close()
    pool.join()
    
            
    time2=time.time()
    print(f'Execution time parallel algorithm {time2-time1:.3f} s')
    return ass,C


def recenter_2(data,ass,cluster):
    
    return sum(data[ass==cluster])/sum(ass==cluster)
