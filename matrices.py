#%%
import numpy as np
import time
from scipy.sparse import coo_matrix
import sys
import matplotlib.pyplot as plt


def fill_mat_V1(n,m):
  A = np.zeros((n,m), dtype=float)
  for i in range(0,n):
    for j in range(0,m):
      if i == j and (i>=0 and j>=0):
        A[i,j] = 2.0
      if i == j+1 and (i>=0 and j>=0):
        A[i,j] = -1.0
      if (i == j-1) and (i>=0 and j>=0):
        A[i,j] = 1.0
  return A

def fill_mat_V2(n,m):
  A = np.zeros((n,m), dtype=float)
  for i in range(0,n):
    for j in range(0,m):
      if i == j:
        A[i,j] = 2.0
      if i == j+1:
        A[i,j] = -1.0
      if (i == j-1):
        A[i,j] = 1.0
  return A

def fill_mat_V3(n):
  A = np.zeros((n,n), dtype=float)
  A[0,0] = 2.0
  A[0,1] = 1.0
  A[n-1,n-1] = 2.0
  A[n-1,n-2] = -1.0
  for i in range(1,n-1):
      A[i,i] = 2.0
      A[i, i-1] = -1.0
      A[i, i+1] = 1.0
  return A

def fill_mat_V4(n):
  row = []
  col = []
  data = []
  row.append(0)
  col.append(0)
  data.append(2.0)

  #row[0] = 0
  #col[0] = 0
  #data[0] = 2.0
  #for i in range(0,n):
  #  row.append(i)
  #  col.append(i)
  #  data.append(2.0)
  
  for i in range(1,n-1):
    row.append(i)
    col.append(i)
    data.append(2.0)

    row.append(i+1)
    col.append(i-1)
    data.append(1.0)

    row.append(i-1)
    col.append(i+1)
    data.append(-1.0)

  row.append(n-1)
  col.append(n-1)
  data.append(2.0)

  row = np.array(row)
  col = np.array(col)
  data = np.array(data)

  A = coo_matrix((data, (row, col)), shape=(n,n))

  return A





#%%
n = 10000
m = 10000
number_tests = 1

A = None
elapsed_time_total = 0.0
elapsed_time_avg = 0.0

for i in range(0,number_tests):
  tic = time.perf_counter()
  A = fill_mat_V3(n)
  toc = time.perf_counter()
  elapsed_time_total = elapsed_time_total + (toc - tic)
  print(f"================= V3 total elapsed time: {elapsed_time_total*1000} ms")
  print(f"====================== size of A is: {sys.getsizeof(A)/(1024**2)} MB")

  tic = time.perf_counter()
  A = fill_mat_V4(n)
  toc = time.perf_counter()
  elapsed_time_total = elapsed_time_total + (toc - tic)
  print(f"================= V4 total elapsed time: {elapsed_time_total*1000} ms")
  print(f"====================== size of A is: {sys.getsizeof(A)/(1024**2)} MB")

elapsed_time_avg = elapsed_time_total/number_tests
print(f"================= avg elapsed time: {elapsed_time_avg*1000} ms")



plt.plot()
plt.show


# %%
"""
Homework TODOs (pro Gruppe, nächste Woche wird eine Gruppe ausgewählt, welche die Ergebnisse herzeigt):
-) Einlesen in COO sparse Datenformat (google)
-) fill_mat_V4 fertig implementieren, dort soll die A-Matrix als sparse Datenstruktur erzeugt werden
-) Create performance plot (also mention the number_tests variable)
    -) y-axis elapsed_time_avg, x-axis number of rows
    -) create one line for every version in one plot (don't forget the legend)
-) Create memory usage plots (also state the number_tests variable)
    -) y-axis is memory consumption in MB, x-axis is number of rows
    -) create one line for every version in one plot (don't forget the legend)

number of rows and number_tests is up to you
"""
