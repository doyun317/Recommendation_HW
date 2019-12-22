# CF_Algorithm

import numpy as np
import pandas as pd
import os

from surprise import SVD
from surprise import Dataset
from surprise import Reader

from surprise import KNNBasic
from surprise import KNNWithMeans
from surprise import KNNWithZScore

from surprise.model_selection import cross_validate
from surprise.model_selection import KFold

from collections import defaultdict

class _CF:
    def __init__(self, data ,extradata):
      
        self.data = data
        self.list = extradata.list
        self.trainset = extradata.trainset
            
    def Basic_CF(self) :
        sim_options = {'name':'cosine','user_based':True}
        algo = KNNBasic(k=40, min_k = 1, sim_options = sim_options)
        algo.fit(self.trainset)

        for uid in (self.list):
            lids = self.data[self.data.uid ==uid]
            a = self.data[self.data.uid==uid]

            for i in range (1, len(a)):
                lid = lids[i-1:i].lid.values[0]
                r_ui = lids[i-1:i].rate.values[0]
                pred = algo.predict(uid, lid, r_ui, verbose = True)
                
        return pred
                
    
    def CFM(self) : 
        sim_options = {'name':'cosine','user_based':True}
        algo = KNNWithMeans(k=40, min_k = 1, sim_options = sim_options)
        algo.fit(self.trainset)

        for uid in (self.list):
            lids = self.data[self.data.uid ==uid]
            a = self.data[self.data.uid==uid]

            for i in range (1, len(a)):
                lid = lids[i-1:i].lid.values[0]
                r_ui = lids[i-1:i].rate.values[0]
                pred = algo.predict(uid, lid, r_ui, verbose = True)
                
        return pred
    
    def CFZ(self) :
        sim_options = {'name':'cosine','user_based':True}
        algo = KNNWithZScore(k=40, min_k = 1, sim_options = sim_options)
        algo.fit(self.trainset)

        for uid in (self.list):
            lids = self.data[self.data.uid ==uid]
            a = self.data[self.data.uid==uid]

            for i in range (1, len(a)):
                lid = lids[i-1:i].lid.values[0]
                r_ui = lids[i-1:i].rate.values[0]
                pred = algo.predict(uid, lid, r_ui, verbose = True)
    
        return pred
    
    def SVD(self) : 
        algo = SVD(n_factors = 100, n_epochs = 20, biased = False, lr_all = 0.005, reg_all = 0)
        algo.fit(self.trainset)

        for uid in (self.list):
            lids = self.data[self.data.uid ==uid]
            a = self.data[self.data.uid==uid]

            for i in range (1, len(a)):
                lid = lids[i-1:i].lid.values[0]
                r_ui = lids[i-1:i].rate.values[0]
                pred = algo.predict(uid, lid, r_ui, verbose = True)
    
        return pred
    
    
    def PMF(self) :
        algo = SVD(n_factors = 100, n_epochs = 20, biased = False, lr_all = 0.005, reg_all = 0.02)
        algo.fit(self.trainset)

        for uid in (self.list):
            lids = self.data[self.data.uid ==uid]
            a = self.data[self.data.uid==uid]

            for i in range (1, len(a)):
                lid = lids[i-1:i].lid.values[0]
                r_ui = lids[i-1:i].rate.values[0]
                pred = algo.predict(uid, lid, r_ui, verbose = True)
                
        return pred
    
    
    def PMFB(self) :
        algo = SVD(n_factors = 100, n_epochs = 20, biased = True, lr_all = 0.005, reg_all = 0.02)
        algo.fit(self.trainset)

        for uid in (self.list):
            lids = self.data[self.data.uid ==uid]
            a = self.data[self.data.uid==uid]

            for i in range (1, len(a)):
                lid = lids[i-1:i].lid.values[0]
                r_ui = lids[i-1:i].rate.values[0]
                pred = algo.predict(uid, lid, r_ui, verbose = True)
                
        return pred