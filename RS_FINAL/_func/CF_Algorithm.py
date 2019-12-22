# CF_알고리즘

import numpy as np
import pandas as pd

class _CF:
    def __init__(self, data ,extra_data):
      
        self.data =np.array(data)
        self.mu   =extra_data.mu
        self.mu_u =extra_data.mu_u
        self.mu_i =extra_data.mu_i   
            
    def basic_baseline_user(self,sim, k):

        predicted_rating = np.array([[0.0 for col in range(100)] for row in range(200)]) # 100 200

        Sim=sim

        k_neighbors = np.argsort(-Sim)
        k_neighbors = np.delete(k_neighbors, np.s_[k:], 1) 

        NumUsers = np.size(self.data, axis = 0) 

        for u in range(0,NumUsers) :
            list_sim = Sim[u, k_neighbors[u,]]
            list_rating = self.data[k_neighbors[u,],].astype('float64')
            baseline_mean = self.mu_u[k_neighbors[u,],]

            denominator = np.sum(list_sim)
            numerator = np.sum(list_sim.reshape(-1,1)*(list_rating - baseline_mean.reshape(-1,1)), axis = 0)        
            predicted_rating[u,] = self.mu_i + self.mu_u[u] + self.mu + numerator / denominator


        return predicted_rating; 

    def basic_baseline_items(self,sim, k):
    
        predicted_rating = np.array([[0.0 for col in range(200)] for row in range(100)]) # 200 100

        Sim=sim

        k_neighbors = np.argsort(-Sim)
        k_neighbors = np.delete(k_neighbors, np.s_[k:], 1) 

        NumItems = np.size(self.data, axis = 0) 

        for i in range(0,NumItems) :
            list_sim = Sim[i, k_neighbors[i,]]
            list_rating =  self.data[k_neighbors[i,],].astype('float64')
            baseline_mean = self.mu_i[k_neighbors[i,],]

            denominator = np.sum(list_sim)
            numerator = np.sum(list_sim.reshape(-1,1)*(list_rating - baseline_mean.reshape(-1,1)), axis = 0)        
            predicted_rating[i,] = self.mu_i[i] + self.mu_u + self.mu + numerator / denominator

        
        return predicted_rating;
    
    