# MovieLens 데이터 로딩 Class

import numpy as np
import pandas as pd
import os

from surprise import Dataset

class _Import_Data:
    def __init__(self, filepath):
        self.filepath=filepath
        
    def _Call_Movielens_for_Evaluation(self):
        
        data = Dataset.load_builtin('ml-100k')
        return data
    

