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
n = 100
m = 10
number_tests = 5


A = None
elapsed_time = 0.0
elapsed_time_avg = 0.0

n_values = []
v1_times = []
v2_times = []
v3_times = []
v4_times = []
v1_sizes = []
v2_sizes = []
v3_sizes = []
v4_sizes = []


for j in range (0, 3): #nicht groesser als 10 -> dauert zu lange
  n_values.append(n)

  elapsed_time = 0.0
  elapsed_time_avg = 0.0
  for i in range(0,number_tests):
    tic = time.perf_counter()
    A = fill_mat_V1(n,n)
    toc = time.perf_counter()
    elapsed_time = elapsed_time + (toc - tic)

  elapsed_time_avg = elapsed_time/number_tests
  v1_times.append(elapsed_time_avg*1000)
  v1_sizes.append(sys.getsizeof(A) / (1024 ** 2))
  print(f"================= V1 avg total elapsed time: {elapsed_time_avg*1000} ms")
  print(f"====================== size of A (V1) is: {sys.getsizeof(A)/(1024**2)} MB") 

  elapsed_time = 0.0
  elapsed_time_avg = 0.0

  for i in range(0,number_tests):
    tic = time.perf_counter()
    A = fill_mat_V2(n,n)
    toc = time.perf_counter()
    elapsed_time = elapsed_time + (toc - tic)

  elapsed_time_avg = elapsed_time/number_tests
  v2_times.append(elapsed_time_avg*1000)
  v2_sizes.append(sys.getsizeof(A) / (1024 ** 2))
  print(f"================= V2 avg total elapsed time: {elapsed_time_avg*1000} ms")
  print(f"====================== size of A (V2) is: {sys.getsizeof(A)/(1024**2)} MB") 

  elapsed_time = 0.0
  elapsed_time_avg = 0.0

  for i in range(0,number_tests):
    tic = time.perf_counter()
    A = fill_mat_V3(n)
    toc = time.perf_counter()
    elapsed_time = elapsed_time + (toc - tic)

  elapsed_time_avg = elapsed_time/number_tests
  v3_times.append(elapsed_time_avg*1000)
  v3_sizes.append(sys.getsizeof(A) / (1024 ** 2))
  print(f"================= V3 avg total elapsed time: {elapsed_time_avg*1000} ms")
  print(f"====================== size of A (V3) is: {sys.getsizeof(A)/(1024**2)} MB") 

  elapsed_time = 0.0
  elapsed_time_avg = 0.0

  for i in range(0,number_tests):
    tic = time.perf_counter()
    A = fill_mat_V4(n)
    toc = time.perf_counter()
    elapsed_time = elapsed_time + (toc - tic)

  elapsed_time_avg = elapsed_time/number_tests
  v4_times.append(elapsed_time_avg*1000)
  v4_sizes.append(sys.getsizeof(A) / (1024 ** 2))
  print(f"================= V4 avg total elapsed time: {elapsed_time_avg*1000} ms")
  print(f"====================== size of A (V4) is: {sys.getsizeof(A)/(1024**2)} MB") 


  n = n + 500

# # === PLOT: Laufzeiten ===
plt.figure(figsize=(10, 6))
plt.plot(n_values, v1_times, label='fill_mat_V1', marker='o')
plt.plot(n_values, v2_times, label='fill_mat_V2', marker='o')
plt.plot(n_values, v3_times, label='fill_mat_V3', marker='o')
plt.plot(n_values, v4_times, label='fill_mat_V4 (Sparse)', marker='o')



#plt.xscale('log')
#plt.yscale()
plt.xlabel("Matrixgröße n")
plt.ylabel(f"Laufzeit in Millisekunden bei {number_tests} Durchlaeufen pro Funktion pro n")
plt.title("Laufzeitvergleich der Funktionen V1 bis V4")
plt.legend()
plt.grid(True, which="both", linestyle='--', linewidth=0.5)
#plt.tight_layout()
plt.show()

# === PLOT: Speicherverbrauch ===
plt.figure(figsize=(10, 6))
plt.plot(n_values, v1_sizes, label='fill_mat_V1', marker='o')
plt.plot(n_values, v2_sizes, label='fill_mat_V2', marker='o')
plt.plot(n_values, v3_sizes, label='fill_mat_V3', marker='o')
plt.plot(n_values, v4_sizes, label='fill_mat_V4 (Sparse)', marker='o')


#plt.xscale('log')
#plt.yscale('log')
plt.xlabel("Matrixgröße n")
plt.ylabel("Speicherverbrauch in MB")
plt.title("Speicherverbrauch der Funktionen V1 bis V4")
plt.legend()
plt.grid(True, which="both", linestyle='--', linewidth=0.5)
#plt.tight_layout()
plt.show()

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
