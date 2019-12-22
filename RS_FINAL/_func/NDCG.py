# NDCG

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


class _NDCG:
    
    def __init__(self, data ,extradata):
     
        self.data = data
        self.list = extradata.list
        self.trainset = extradata.trainset
        self.id_no       =[]
        self.CF_ndcg_    =[]
        self.CFWM_ndcg_  =[]  
        self.CFWZ_ndcg_  =[]
        self.SVD_ndcg_   =[]
        self.PMF_ndcg_   =[]
        self.PMFWB_ndcg_ =[]
        
    def discounted_cumulative_gain_score(self,rel_list, p):
        
        dcg = rel_list[0]
        for idx in range(1, p):
            dcg += (rel_list[idx] / np.log2(idx+1))
        return dcg
    
    def Calculate_NDCG(self):

        if(len(self.id_no)==0):
            flag = 1
        else:
            flag = 0
            
        for i in self.arr:
            temp = self.df_est[self.df_est['uid']==int(i)]
            temp2 = temp.sort_values(by=['est'],ascending = False).reset_index(drop=True)
            temp3 = temp.sort_values(by=['r_ui'],ascending = False).reset_index(drop=True)

            rank_relevant_score = temp2['r_ui']
            ideal_vector        = temp3['r_ui']

            dcg = self.discounted_cumulative_gain_score(rank_relevant_score, p=len(rank_relevant_score))
            idcg = self.discounted_cumulative_gain_score(ideal_vector, p=len(ideal_vector))
            ndcg = dcg / idcg
            
            if(flag ==1):
                self.id_no.append(int(i))
            
        return ndcg
    
    def Basic_CF(self) :
        u_id = []
        I_id = []
        r_ui_= np.array([])
        _est = np.array([])
         
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
                u_id.append(int(pred.uid))
                I_id.append(int(pred.iid))
                r_ui_ = np.append(r_ui_,pred.r_ui)
                _est = np.append(_est,pred.est)

        self.df_est = pd.DataFrame({ 'uid' :u_id, 'Iid':I_id, 'r_ui' : r_ui_, 'est': _est})
        self.arr = self.df_est['uid'].unique()
        
        self.CF_ndcg_= self.Calculate_NDCG()
        
    def CFM(self) :
        u_id = []
        I_id = []
        r_ui_= np.array([])
        _est = np.array([])
        
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
                u_id.append(int(pred.uid))
                I_id.append(int(pred.iid))
                r_ui_ = np.append(r_ui_,pred.r_ui)
                _est = np.append(_est,pred.est)

        self.df_est = pd.DataFrame({ 'uid' :u_id, 'Iid':I_id, 'r_ui' : r_ui_, 'est': _est})
        self.arr = self.df_est['uid'].unique()
        
        self.CFWM_ndcg_= self.Calculate_NDCG()            
        
    def CFZ(self) :
        u_id = []
        I_id = []
        r_ui_= np.array([])
        _est = np.array([])
        
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
                u_id.append(int(pred.uid))
                I_id.append(int(pred.iid))
                r_ui_ = np.append(r_ui_,pred.r_ui)
                _est = np.append(_est,pred.est)

        self.df_est = pd.DataFrame({ 'uid' :u_id, 'Iid':I_id, 'r_ui' : r_ui_, 'est': _est})
        self.arr = self.df_est['uid'].unique()
        
        self.CFWZ_ndcg_= self.Calculate_NDCG()   
        
    def SVD(self) :    
        u_id = []
        I_id = []
        r_ui_= np.array([])
        _est = np.array([])
        
        algo = SVD()
        algo.fit(self.trainset)

        for uid in (self.list):
            lids = self.data[self.data.uid ==uid]
            a = self.data[self.data.uid==uid]

            for i in range (1, len(a)):
                lid = lids[i-1:i].lid.values[0]
                r_ui = lids[i-1:i].rate.values[0]
                pred = algo.predict(uid, lid, r_ui, verbose = True)
                u_id.append(int(pred.uid))
                I_id.append(int(pred.iid))
                r_ui_ = np.append(r_ui_,pred.r_ui)
                _est = np.append(_est,pred.est)

        self.df_est = pd.DataFrame({ 'uid' :u_id, 'Iid':I_id, 'r_ui' : r_ui_, 'est': _est})
        self.arr = self.df_est['uid'].unique()
        
        self.SVD_ndcg_= self.Calculate_NDCG()   
        
    def PMF(self) :
        u_id = []
        I_id = []
        r_ui_= np.array([])
        _est = np.array([])
        
        algo = SVD(n_factors = 100, n_epochs = 20, biased = False, lr_all = 0.005, reg_all = 0.02)
        algo.fit(self.trainset)

        for uid in (self.list):
            lids = self.data[self.data.uid ==uid]
            a = self.data[self.data.uid==uid]

            for i in range (1, len(a)):
                lid = lids[i-1:i].lid.values[0]
                r_ui = lids[i-1:i].rate.values[0]
                pred = algo.predict(uid, lid, r_ui, verbose = True)
                u_id.append(int(pred.uid))
                I_id.append(int(pred.iid))
                r_ui_ = np.append(r_ui_,pred.r_ui)
                _est = np.append(_est,pred.est)

        self.df_est = pd.DataFrame({ 'uid' :u_id, 'Iid':I_id, 'r_ui' : r_ui_, 'est': _est})
        self.arr = self.df_est['uid'].unique()
        
        self.PMF_ndcg_= self.Calculate_NDCG() 
        
    def PMFB(self) :   
        u_id = []
        I_id = []
        r_ui_= np.array([])
        _est = np.array([])
        
        algo = SVD(n_factors = 100, n_epochs = 20, biased = True, lr_all = 0.005, reg_all = 0.02)
        algo.fit(self.trainset)

        for uid in (self.list):
            lids = self.data[self.data.uid ==uid]
            a = self.data[self.data.uid==uid]

            for i in range (1, len(a)):
                lid = lids[i-1:i].lid.values[0]
                r_ui = lids[i-1:i].rate.values[0]
                pred = algo.predict(uid, lid, r_ui, verbose = True)
                u_id.append(int(pred.uid))
                I_id.append(int(pred.iid))
                r_ui_ = np.append(r_ui_,pred.r_ui)
                _est = np.append(_est,pred.est)

        self.df_est = pd.DataFrame({ 'uid' :u_id, 'Iid':I_id, 'r_ui' : r_ui_, 'est': _est})
        self.arr = self.df_est['uid'].unique()
        
        self.PMFWB_ndcg_= self.Calculate_NDCG()    
        
        
   