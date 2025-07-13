# -*- coding: utf-8 -*-
"""
Created on Fri Jul 11 16:01:54 2025

@author: erni_
"""

import numpy as np
import random 
import matplotlib
import matplotlib.pyplot as plt
import time

from multiprocessing import Value, Array

from utils import *



    

def main():
    #Ns=[10000, 100000, 1000000]
    Ns=[10000,100000,1000000]
    ks=[10,20,40]
    D=2
    for k in ks:
        
        speed_N=[]
        for N in Ns:
            print("N=",N)
            data=generate_cliusters(k,N,D)
            nome='Datasets/new_cluster_k_'+str(k)+'_N_'+str(N)+'.csv'
            np.savetxt(nome, data, delimiter=",", fmt="%f")
            
            time_start_s=time.time()
            ass,C=k_means_serial(data, k, N,D)
            time_end_s=time.time()
            analysis(data, ass, k,C)
            
            
            speedups=[]
            for rep in range(0,5):
                
                pool_size=rep
                time1=time.time()
                
                ass1,C=k_means_parallel(data, k, N,2**rep,D)
                time2=time.time()
                
                assert np.all(ass==ass1)
                
                speedup=(time_end_s-time_start_s)/(time2-time1)
                
                print(f'Processes: {2**rep:2d}, Speedup: {speedup:.5f} ')
                
                speedups.append([2**rep , speedup])
            speed_N.append(speedups)
    
        t=np.array([1,2,4,8,16])    
        name='Figures/Speedups_k_'+str(k)+'.png'
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