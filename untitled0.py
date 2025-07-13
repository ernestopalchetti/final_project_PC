# -*- coding: utf-8 -*-
"""
Created on Fri Jul 11 16:01:54 2025

@author: erni_
"""

import numpy as np
import random 
#import itertools
import matplotlib
import matplotlib.pyplot as plt
import time

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


        
def k_means_parallel_2(data,k,N,pool_size,D):
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
    return ass




def recenter(sums,counters,i):
    return sums[i,:]/counters[i]

def generate_cliusters(k,N,D):

    random.seed(11)
    np.random.seed(11)
    data=np.zeros((N,D))
    C=np.random.random((k,D))
    
    for i in range(N):
        p=random.randint(0,11)
        data[i,:]=C[p,:]+np.random.normal(0,0.02,D)
        
        
    return data
        
    

def main():
    Ns=[10000, 100000, 1000000]
    #Ns=[100000]
    D=2
    k=12
    speed_N=[]
    for N in Ns:
        print("N=",N)
        data=generate_cliusters(k,N,D)
        
        np.savetxt("new_cluster.csv", data, delimiter=",", fmt="%f")
        
        time_start_s=time.time()
        ass,C=k_means_serial(data, k, N,D)
        time_end_s=time.time()
        analysis(data, ass, k,C)
        
        
        speedups=[]
        for rep in range(0,5):
            
            pool_size=rep
            time1=time.time()
            ass=k_means_parallel(data, k, N,2**rep,D)
            time2=time.time()
            
            speedup=(time_end_s-time_start_s)/(time2-time1)
            
            print(f'Processes: {2**rep:2d}, Speedup: {speedup:.5f} ')
            
            speedups.append([2**rep , speedup])
        speed_N.append(speedups)

    t=np.array([1,2,4,8,16])    
    name='Figures/Speedups.png'
    speed_N=np.array(speed_N)
    fig, ax, =plt.subplots(figsize=(10, 8))
    for i,N in enumerate(Ns):
        ax.plot(speed_N[i,:,0],speed_N[i,:,1],label=str(N))
    ax.plot(t,t,label=None,ls=":")
    plt.title("Speedups")
    plt.ylim(top=3)
    plt.xlabel("Threads")
    plt.ylabel("Speedup")
    plt.legend(title='N',loc='upper right')
    plt.grid(True)
    plt.savefig(name)
    plt.show()
    
    
if __name__ == '__main__':
    main()