import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["figure.figsize"] = (7.5,5.5)

df = pd.read_csv('data_queries_postgres_stat_synt.txt',sep="|")


print(df.head())
print(df.describe())
print(df['act_row'].describe())
print(list(df))

plt.scatter(df['est_cost'],df['exec_time'], c="black", label='postgres',s=10)
plt.xlabel('estimated cost')
plt.ylabel('execution time ')
plt.legend()
plt.show()

plt.scatter(df['est_row'],df['act_row'],c="black", label='postgres',s=10)
plt.xlabel('predicted # of rows')
plt.ylabel('actual # of rows')
plt.legend()
plt.show()

