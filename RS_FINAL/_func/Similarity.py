# 유사도

import numpy as np
import pandas as pd

class _similarity:
    def __init__(self, data):
        self.data=data
        
    def COS(self):
        NumUsers = np.size(self.data, axis = 0)
        Sim = np.full((NumUsers, NumUsers), 0.0)
        
        for u in range(0, NumUsers):
            arridx_u = np.where(self.data[u,] == 0)
            for v in range(u+1, NumUsers):
                arridx_v = np.where(self.data[v, ] == 0)
                arridx = np.unique(np.concatenate((arridx_u, arridx_v), axis = None))

                U = np.delete(self.data[u, ], arridx)
                V = np.delete(self.data[v, ], arridx)

                if(np.linalg.norm(U) == 0 or np.linalg.norm(V) == 0):
                    Sim[u,v] = 0
                else :
                    Sim[u,v] = np.dot(U, V) / (np.linalg.norm(U) * np.linalg.norm(V))
                Sim[v, u] = Sim[u, v]

        return Sim
    
    def PCC(self):
        NumUsers = np.size(self.data, axis = 0)
        Sim = np.full((NumUsers, NumUsers), -1.0) 

        mean = np.nanmean(np.where(self.data != 0.0,self.data, np.nan), axis = 1)

        for u in range(0, NumUsers) : 
            arridx_u = np.where(self.data[u,] == 0)
            for v in range(u+1, NumUsers) :
                arridx_v = np.where(self.data[v, ] == 0)
                arridx = np.unique(np.concatenate((arridx_u, arridx_v), axis = None))

                U = np.delete(self.data[u, ], arridx) - mean[u]
                V = np.delete(self.data[v, ], arridx) - mean[v]

                if(np.linalg.norm(U) == 0 or np.linalg.norm(V) == 0):
                    Sim[u, v] = 0
                else:
                    Sim[u, v] = np.dot(U,V) / (np.linalg.norm(U) * np.linalg.norm(V))

                Sim[v, u] = Sim[u, v]

        return Sim

    def JAC(self):
        NumUsers =np.size(self.data, axis = 0)
        Sim = np.full((NumUsers, NumUsers), 0.0)
        for u in range(0, NumUsers):
            for v in range(u , NumUsers):
                U = np.array(self.data[u, ] > 0, dtype = np.int)
                V = np.array(self.data[v, ] > 0, dtype = np.int)
                SumUV = U + V
                Inter = np.sum(np.array(SumUV > 1 , dtype = np.int))
                Union = np.sum(np.array(SumUV > 0, dtype = np.int))
                tmp = Inter / Union
                Sim[u, v] = tmp 
                Sim[v, u] = Sim[u,v]
                
        return Sim
        
        
    def MSD(self):
        NumUsers = np.size(self.data, axis = 0)
        Sim = np.full((NumUsers, NumUsers), 0.0)
        for u in range(0, NumUsers):
            for v in range(u, NumUsers):
                U = np.where(self.data[u,] == 0, np.nan, self.data[u, ])
                V = np.where(self.data[v, ] == 0 , np.nan, self.data[v, ])
                SquaredSum = np.square(U-V)
                SquaredSum = SquaredSum[~np.isnan(SquaredSum)]
                AllItems = np.size(SquaredSum, axis = 0)
                tmp = np.sum(SquaredSum)/AllItems
                Sim[u,v] = tmp 
                Sim[v, u] = Sim[u, v]
        return Sim
    
    def JMSD(self,max):
        return self.JAC() * (1 - ((self.MSD()/max)))     
    