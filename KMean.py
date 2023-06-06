import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

# import matplotlib.pyplot as plt
# import seaborn as sns
# import plotly.express as px
# from mpl_toolkits.mplot3d import Axes3D
# import plotly as py
# import plotly.graph_objs as go


customers = pd.read_csv('./datasets/Mall_Customers.csv')

customers = customers.rename(columns={'Annual Income (k$)': 'Annual_income', 'Spending Score (1-100)': 'Spending_score'})

df_3d = customers.drop(columns=(['CustomerID', 'Gender']))
# df_shuffle = df_3d.sample(frac=1)
X = np.array(df_3d.astype(float))


model = KMeans(n_clusters=6, n_init=10, max_iter=30000, tol=0.0001, random_state=111)
model.fit(X)
labels3 = model.labels_

customers["label"] = labels3;
print(customers)

