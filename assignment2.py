# Code for task 2 : 

import pandas as pd
import numpy as np

#Read Data
df = df = pd.read_csv('C:/Users/Dell/Desktop/cours_S2/comput_tools/current_dataset.csv')

#Clean the data : Select INDPRO
INDPRO = df['INDPRO']
#Drop first Row
INDPRO = INDPRO.drop(index=0)
#transform INDPRO using log differences
INDPRO = np.log(INDPRO).diff()

print(df)

