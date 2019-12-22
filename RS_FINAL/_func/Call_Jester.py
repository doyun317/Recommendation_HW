# jester 데이터 로딩 Class

import numpy as np
import pandas as pd

class _Import_Data:
    def __init__(self, filepath):
        self.filepath=filepath
        
    def _Call_Jester(self):
        datasize=200
        j_data = pd.read_csv(self.filepath,header=None)
        
        j_data = j_data.replace(99,np.nan) 
        # 99, 즉 Rating을 하지 않는 말장난의 점수를 np.nan으로 바꿈.
        
        for i in range(datasize):
            x =  j_data.iloc[i,1:]
            j_data.iloc[i,1:] = np.where(x!=np.nan,(x-np.min(x))/(np.max(x)-np.min(x)),x)   
        
        # 데이터가 24983개 이기 때문에 200개까지만 사용하고,
        # 반복문으로 200개의 사용자를 보면서 nan값을 제외하고 각 줄의 레이팅을 0부터 1사이 값으로 스케일링 해준다
        
        data = j_data.iloc[:datasize,1:].values 

        data2 = data

        data[np.isnan(data)]=0
        
        # 위에서 구한 값을 data에 다시 넣어주고 아래 유사도 측정 전에 nan값을 다시 0으로 바꾸어 주었다.
        
        self.mu   = np.nanmean(data2)

        self.mu_u = np.nanmean(data2,axis=1)

        self.mu_i = np.nanmean(data2,axis=0)

        self.mu_u = self.mu_u - self.mu
        self.mu_i = self.mu_i - self.mu
        
        # 평균값을 구할땐 nan값을 0으로 바꾼 값이 아닌 데이터(data2)로 사용하여 구한다.
        # mu 는 전체 데이터프레임의 평균
        #mu_u는 사용자가 레이팅한 값의 평균
        #mu_i는 아이템이 가진 레이팅값의 평균
        
        data = j_data.iloc[:datasize,1:]
        data = data.values
        
        #데이터프레임이 아닌 matrix 형태로 변환
    
        return data
    

