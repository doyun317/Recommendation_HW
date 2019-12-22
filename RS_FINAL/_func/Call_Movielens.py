# MovieLens 데이터 로딩 Class

import numpy as np
import pandas as pd
import os

from surprise import Dataset
from surprise import Reader


class _Import_Data:
    def __init__(self, filepath):
        self.filepath=filepath
        
    def _Call_Movielens(self):
        
        file_path = os.path.expanduser(self.filepath)
        reader = Reader(line_format = 'user item rating timestamp',sep='\t')
        data = Dataset.load_from_file(file_path, reader = reader)
        trainset = data.build_full_trainset()

        df = pd.DataFrame(data.raw_ratings,columns = ['uid',"lid","rate","timestamp"])
        
        
        # 데이터 로딩 및, 데이터의 크기가 너무 커 레이팅 횟수가 많은 상위 40명을 추려서 필터링
        
        _filter = df["uid"].value_counts()[:40].index.tolist()
        df = df[df["uid"].isin(_filter)].reset_index(drop=True)

        _list = df['uid'].unique()
        _list = sorted(_list)
        
        self.list = _list
        self.trainset = trainset
        
        # 데이터 확인 및, 유저 아이디 개별적으로 확인하기 위해 unique()함수를 사용하여 중복값을 제거 한 후에 변수에 저장한다

        #원 데이터 uid가 무작위 순서(timestamp에 따라서)로 배치돼있기 때문에 sorted()함수를 이용하여 정렬

        #이때 uid 자체가 문자열로 저장되어 있기 때문에 정렬이 ASCII 코드를 따른다 (순서상 '1' 다음에 '2' 가아닌 '10'이 존재 ['1','10','2'] 순서 )
        
        return df
    

