
Numpy Tutorial
# Import Numpy Library
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from IPython.display import Image
Numpy Array Creation
list1 = [10,20,30,40,50,60]
list1
[10, 20, 30, 40, 50, 60]
# Display the type of an object
type(list1)
list
#Convert list to Numpy Array
arr1 = np.array(list1)
arr1
array([10, 20, 30, 40, 50, 60])
#Memory address of an array object
arr1.data
<memory at 0x000001C2B747E348>
# Display type of an object
type(arr1)
numpy.ndarray
#Datatype of array
arr1.dtype
dtype('int32')
# Convert Integer Array to FLOAT
arr1.astype(float)
array([10., 20., 30., 40., 50., 60.])
# Generate evenly spaced numbers (space =1) between 0 to 10
np.arange(0,10)
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
# Generate numbers between 0 to 100 with a space of 10
np.arange(0,100,10)
array([ 0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
# Generate numbers between 10 to 100 with a space of 10 in descending order
np.arange(100, 10, -10)
array([100,  90,  80,  70,  60,  50,  40,  30,  20])
#Shape of Array
arr3 = np.arange(0,10)
arr3.shape
(10,)
arr3
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
# Size of array
arr3.size
10
# Dimension 
arr3.ndim
1
# Datatype of object
arr3.dtype
dtype('int32')
# Bytes consumed by one element of an array object
arr3.itemsize
4
# Bytes consumed by an array object
arr3.nbytes
40
# Length of array
len(arr3)
10
# Generate an array of zeros
np.zeros(10)
array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
# Generate an array of ones with given shape
np.ones(10)
array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
# Repeat 10 five times in an array
np.repeat(10,5)
array([10, 10, 10, 10, 10])
# Repeat each element in array 'a' thrice
a= np.array([10,20,30])
np.repeat(a,3)
array([10, 10, 10, 20, 20, 20, 30, 30, 30])
# Array of 10's
np.full(5,10)
array([10, 10, 10, 10, 10])
# Generate array of Odd numbers
ar1 = np.arange(1,20)
ar1[ar1%2 ==1]
array([ 1,  3,  5,  7,  9, 11, 13, 15, 17, 19])
# Generate array of even numbers
ar1 = np.arange(1,20)
ar1[ar1%2 == 0]
array([ 2,  4,  6,  8, 10, 12, 14, 16, 18])
# Generate evenly spaced 4 numbers between 10 to 20.
np.linspace(10,20,4)
array([10.        , 13.33333333, 16.66666667, 20.        ])
# Generate evenly spaced 11 numbers between 10 to 20.
np.linspace(10,20,11)
array([10., 11., 12., 13., 14., 15., 16., 17., 18., 19., 20.])
# Create an array of random values
np.random.random(4)
array([0.61387161, 0.7734601 , 0.48868515, 0.05535259])
# Generate an array of Random Integer numbers
np.random.randint(0,500,5)
array([359,   3, 200, 437, 400])
# Generate an array of Random Integer numbers
np.random.randint(0,500,10)
array([402, 196, 481, 426, 245,  19, 292, 233, 399, 175])
# Using random.seed we can generate same number of Random numbers
np.random.seed(123)
np.random.randint(0,100,10)
array([66, 92, 98, 17, 83, 57, 86, 97, 96, 47])
# Using random.seed we can generate same number of Random numbers
np.random.seed(123)
np.random.randint(0,100,10)
array([66, 92, 98, 17, 83, 57, 86, 97, 96, 47])
# Using random.seed we can generate same number of Random numbers
np.random.seed(101)
np.random.randint(0,100,10)
array([95, 11, 81, 70, 63, 87, 75,  9, 77, 40])
# Using random.seed we can generate same number of Random numbers
np.random.seed(101)
np.random.randint(0,100,10)
array([95, 11, 81, 70, 63, 87, 75,  9, 77, 40])
# Generate array of Random float numbers
f1 = np.random.uniform(5,10, size=(10))
f1
array([6.5348311 , 9.4680654 , 8.60771931, 5.94969477, 7.77113796,
       6.76065977, 5.90946201, 8.92800881, 9.82741611, 6.16176831])
# Extract Integer part
np.floor(f1)
array([6., 9., 8., 5., 7., 6., 5., 8., 9., 6.])
# Truncate decimal part
np.trunc(f1)
array([6., 9., 8., 5., 7., 6., 5., 8., 9., 6.])
# Convert Float Array to Integer array
f1.astype(int)
array([6, 9, 8, 5, 7, 6, 5, 8, 9, 6])
# Normal distribution (mean=0 and variance=1)
b2 =np.random.randn(10)
b2
array([ 0.18869531, -0.75887206, -0.93323722,  0.95505651,  0.19079432,
        1.97875732,  2.60596728,  0.68350889,  0.30266545,  1.69372293])
arr1
array([10, 20, 30, 40, 50, 60])
# Enumerate for Numpy Arrays
for index, value in np.ndenumerate(arr1):
    print(index, value)
(0,) 10
(1,) 20
(2,) 30
(3,) 40
(4,) 50
(5,) 60
Operations on an Array
arr2 = np.arange(1,20)
arr2
array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
       18, 19])
# Sum of all elements in an array
arr2.sum()
190
# Cumulative Sum
np.cumsum(arr2)
array([  1,   3,   6,  10,  15,  21,  28,  36,  45,  55,  66,  78,  91,
       105, 120, 136, 153, 171, 190], dtype=int32)
# Find Minimum number in an array
arr2.min()
1
# Find MAX number in an array
arr2.max()
19
# Find INDEX of Minimum number in an array
arr2.argmin()
0
# Find INDEX of MAX number in an array
arr2.argmax()
18
# Find mean of all numbers in an array
arr2.mean()
10.0
# Find median of all numbers present in arr2
np.median(arr2)
10.0
# Variance
np.var(arr2)
30.0
# Standard deviation
np.std(arr2)
5.477225575051661
# Calculating percentiles
np.percentile(arr2,70)
13.6
# 10th & 70th percentile
np.percentile(arr2,[10,70])
array([ 2.8, 13.6])
Operations on a 2D Array
A = np.array([[1,2,3,0] , [5,6,7,22] , [10 , 11 , 1 ,13] , [14,15,16,3]])
A
array([[ 1,  2,  3,  0],
       [ 5,  6,  7, 22],
       [10, 11,  1, 13],
       [14, 15, 16,  3]])
# SUM of all numbers in a 2D array
A.sum()
129
# MAX number in a 2D array
A.max()
22
# Minimum
A.min()
0
# Column wise mimimum value 
np.amin(A, axis=0)
array([1, 2, 1, 0])
# Row wise mimimum value 
np.amin(A, axis=1)
array([0, 5, 1, 3])
# Mean of all numbers in a 2D array
A.mean()
8.0625
# Mean
np.mean(A)
8.0625
# Median
np.median(A)
6.5
# 50 percentile = Median
np.percentile(A,50)
6.5
np.var(A)
40.30859375
np.std(A)
6.348904925260734
np.percentile(arr2,70)
13.6
# Enumerate for Numpy 2D Arrays
for index, value in np.ndenumerate(A):
    print(index, value)
(0, 0) 1
(0, 1) 2
(0, 2) 3
(0, 3) 0
(1, 0) 5
(1, 1) 6
(1, 2) 7
(1, 3) 22
(2, 0) 10
(2, 1) 11
(2, 2) 1
(2, 3) 13
(3, 0) 14
(3, 1) 15
(3, 2) 16
(3, 3) 3
Reading elements of an array
a = np.array([7,5,3,9,0,2])
# Access first element of the array
a[0]
7
# Access all elements of Array except first one.
a[1:]
array([5, 3, 9, 0, 2])
# Fetch 2nd , 3rd & 4th value from the Array
a[1:4]
array([5, 3, 9])
# Get last element of the array
a[-1]
2
a[-3]
9
a[-6]
7
a[-3:-1]
array([9, 0])
Replace elements in array
ar = np.arange(1,20)
ar
array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
       18, 19])
# Replace EVEN numbers with ZERO
rep1 = np.where(ar % 2 == 0, 0 , ar)
print(rep1)
[ 1  0  3  0  5  0  7  0  9  0 11  0 13  0 15  0 17  0 19]
ar2 = np.array([10, 20 , 30 , 10 ,10 ,20, 20])
ar2
array([10, 20, 30, 10, 10, 20, 20])
# Replace 10 with value 99
rep2 = np.where(ar2 == 10, 99 , ar2)
print(rep2)
[99 20 30 99 99 20 20]
p2 = np.arange(0,100,10)
p2
array([ 0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
# Replace values at INDEX loc 0,3,5 with 33,55,99
np.put(p2, [0, 3 , 5], [33, 55, 99])
p2
array([33, 10, 20, 55, 40, 99, 60, 70, 80, 90])
Missing Values in an array
a = np.array([10 ,np.nan,20,30,60,np.nan,90,np.inf])
a
array([10., nan, 20., 30., 60., nan, 90., inf])
# Search for missing values and return as a boolean array
np.isnan(a)
array([False,  True, False, False, False,  True, False, False])
# Index of missing values in an array
np.where(np.isnan(a))
(array([1, 5], dtype=int64),)
# Replace all missing values with 99
a[np.isnan(a)] = 99
a
array([10., 99., 20., 30., 60., 99., 90., inf])
# Check if array has any NULL value
np.isnan(a).any()
False
A = np.array([[1,2,np.nan,4] , [np.nan,6,7,8] , [10 , np.nan , 12 ,13] , [14,15,16,17]])
A
array([[ 1.,  2., nan,  4.],
       [nan,  6.,  7.,  8.],
       [10., nan, 12., 13.],
       [14., 15., 16., 17.]])
# Search for missing values and return as a boolean array
np.isnan(A)
array([[False, False,  True, False],
       [ True, False, False, False],
       [False,  True, False, False],
       [False, False, False, False]])
# Index of missing values in an array
np.where(np.isnan(A))
(array([0, 1, 2], dtype=int64), array([2, 0, 1], dtype=int64))
Stack Arrays Vertically
a = np.zeros(20).reshape(2,-1)
b = np.repeat(1, 20).reshape(2,-1)
a
array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
b
array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
np.vstack([a,b])
array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
       [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]])
a1 = np.array([[1], [2], [3]])
b1 = np.array([[4], [5], [6]])
a1
array([[1],
       [2],
       [3]])
b1
array([[4],
       [5],
       [6]])
np.vstack([a1,b1])
array([[1],
       [2],
       [3],
       [4],
       [5],
       [6]])
Stack Arrays Horizontally
np.hstack([a,b])
array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1.]])
np.hstack([a1,b1])
array([[1, 4],
       [2, 5],
       [3, 6]])
### hstack & vstack

arr1 = np.array([[7,13,14],[18,10,17],[11,12,19]])
arr2= np.array([16,6,1])
arr3= np.array([[5,8,4,3]])

np.hstack((np.vstack((arr1,arr2)),np.transpose(arr3)))
array([[ 7, 13, 14,  5],
       [18, 10, 17,  8],
       [11, 12, 19,  4],
       [16,  6,  1,  3]])
Common items between two Arrays
c1 = np.array([10,20,30,40,50,60])
c2 = np.array([12,20,33,40,55,60])
np.intersect1d(c1,c2)
array([20, 40, 60])
Remove Common Elements
# Remove common elements of C1 & C2 array from C1

np.setdiff1d(c1,c2)
array([10, 30, 50])
Process Elements on Conditions
a = np.array([1,2,3,6,8])
b = np.array([10,2,30,60,8])

np.where(a == b) # returns the indices of elements in an input array where the given condition is satisfied.
(array([1, 4], dtype=int64),)
# Return an array where condition is satisfied
a[np.where(a == b)]
array([2, 8])
# Return all numbers betweeen 20 & 35
a1 = np.arange(0,60)
a1[np.where ((a1>20) & (a1<35))]
array([21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34])
# Return all numbers betweeen 20 & 35 OR numbers divisible by 10
a1 = np.arange(0,60)
a1[np.where (((a1>20) & (a1<35)) | (a1 % 10 ==0)) ]
array([ 0, 10, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
       40, 50])
# Return all numbers betweeen 20 & 35 using np.logical_and
a1[np.where(np.logical_and(a1>20, a1<35))]
array([21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34])
Check for elements in an Array using isin()
a = np.array([10,20,30,40,50,60,70])
a
array([10, 20, 30, 40, 50, 60, 70])
# Check whether number 11 & 20 are present in an array
np.isin(a, [11,20])
array([False,  True, False, False, False, False, False])
#Display the matching numbers
a[np.isin(a,20)]
array([20])
# Check whether number 33 is present in an array
np.isin(a, 33)
array([False, False, False, False, False, False, False])
a[np.isin(a, 33)]
array([], dtype=int32)
b = np.array([10,20,30,40,10,10,70,80,70,90])
b
array([10, 20, 30, 40, 10, 10, 70, 80, 70, 90])
# Check whether number 10 & 70 are present in an array
np.isin(b, [10,70])
array([ True, False, False, False,  True,  True,  True, False,  True,
       False])
# Display the indices where match occurred
np.where(np.isin(b, [10,70]))
(array([0, 4, 5, 6, 8], dtype=int64),)
# Display the matching values
b[np.where(np.isin(b, [10,70]))]
array([10, 10, 10, 70, 70])
# Display the matching values
b[np.isin(b, [10,70])]
array([10, 10, 10, 70, 70])
Reverse Array
a4 = np.arange(10,30)
a4
array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
       27, 28, 29])
# Reverse the array
a4[::-1]
array([29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13,
       12, 11, 10])
# Reverse the array
np.flip(a4)
array([29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13,
       12, 11, 10])
a3 = np.array([[3,2,8,1] , [70,50,10,67] , [45,25,75,15] , [12,9,77,4]])
a3
array([[ 3,  2,  8,  1],
       [70, 50, 10, 67],
       [45, 25, 75, 15],
       [12,  9, 77,  4]])
# Reverse ROW positions
a3[::-1,]
array([[12,  9, 77,  4],
       [45, 25, 75, 15],
       [70, 50, 10, 67],
       [ 3,  2,  8,  1]])
# Reverse COLUMN positions
a3[:,::-1]
array([[ 1,  8,  2,  3],
       [67, 10, 50, 70],
       [15, 75, 25, 45],
       [ 4, 77,  9, 12]])
# Reverse both ROW & COLUMN positions
a3[::-1,::-1]
array([[ 4, 77,  9, 12],
       [15, 75, 25, 45],
       [67, 10, 50, 70],
       [ 1,  8,  2,  3]])
Sorting Array
a = np.array([10,5,2,22,12,92,17,33])
# Sort array in ascending order
np.sort(a)
array([ 2,  5, 10, 12, 17, 22, 33, 92])
a3 = np.array([[3,2,8,1] , [70,50,10,67] , [45,25,75,15]])
a3
array([[ 3,  2,  8,  1],
       [70, 50, 10, 67],
       [45, 25, 75, 15]])
# Sort along rows
np.sort(a3)
array([[ 1,  2,  3,  8],
       [10, 50, 67, 70],
       [15, 25, 45, 75]])
# Sort along rows
np.sort(a3,axis =1)
array([[ 1,  2,  3,  8],
       [10, 50, 67, 70],
       [15, 25, 45, 75]])
# Sort along columns
np.sort(a3,axis =0)
array([[ 3,  2,  8,  1],
       [45, 25, 10, 15],
       [70, 50, 75, 67]])
# Sort in descending order
b = np.sort(a)
b = b[::-1]
b
array([92, 33, 22, 17, 12, 10,  5,  2])
# Sort in descending order
c = np.sort(a)
np.flip(c)
array([92, 33, 22, 17, 12, 10,  5,  2])
# Sort in descending order
a[::-1].sort()
a
array([92, 33, 22, 17, 12, 10,  5,  2])
"N" Largest & Smallest Numbers in an Array
p = np.arange(0,50)
p
array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
       34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49])
np.random.shuffle(p)
p
array([33, 48, 14, 20, 44, 29,  4, 46, 18, 45, 21,  2,  7, 30, 17, 40, 37,
       42, 34, 25, 35, 38, 43,  8, 24, 32, 10, 36,  0, 26, 12,  9,  3, 39,
        6, 49, 23, 13,  1,  5, 19, 27, 47, 15, 22, 11, 41, 31, 16, 28])
# Return "n" largest numbers in an Array
n = 4
p[np.argsort(p)[-nth:]]
array([46, 47, 48, 49])
# Return "n" largest numbers in an Array
p[np.argpartition(-p,n)[:n]]
array([48, 47, 49, 46])
# Return "n" smallest numbers in an Array
p[np.argsort(-p)[-n:]]
array([3, 2, 1, 0])
# Return "n" smallest numbers in an Array
p[np.argpartition(p,n)[:n]]
array([1, 0, 2, 3])
Repeating Sequences
a5 = [10,20,30] 
a5
[10, 20, 30]
# Repeat whole array twice
np.tile(a5, 2)
array([10, 20, 30, 10, 20, 30])
# Repeat each element in an array thrice
np.repeat(a5, 3)
array([10, 10, 10, 20, 20, 20, 30, 30, 30])
Compare Arrays
d1 = np.arange(0,10)
d1
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
d2 = np.arange(0,10)
d2
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
d3 = np.arange(10,20)
d3
array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
d4 = d1[::-1]
d4
array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0])
# Compare arrays using "allclose" function. If this function returns True then Arrays are equal
res1 = np.allclose(d1,d2)
res1
True
# Compare arrays using "allclose" function. If this function returns False then Arrays are not equal
res2 = np.allclose(d1,d3)
res2
False
# Compare arrays using "allclose" function.
res3 = np.allclose(d1,d4)
res3
False
Frequent Values in an Array
# unique numbers in an array
b = np.array([10,10,10,20,30,20,30,30,20,10,10,30,10])
np.unique(b)
array([10, 20, 30])
# unique numbers in an array along with the count E.g value 10 occurred maximum times (5 times) in an array "b"
val , count = np.unique(b,return_counts=True)
val,count
(array([10, 20, 30]), array([6, 3, 4], dtype=int64))
# 10 is the most frequent value 
np.bincount(b).argmax()
10
Read-Only Array
d5 = np.arange(10,100,10)
d5
array([10, 20, 30, 40, 50, 60, 70, 80, 90])
# Make arrays immutable  
d5.flags.writeable = False
d5[0] = 99
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
<ipython-input-712-cff52f234eeb> in <module>
----> 1 d5[0] = 99

ValueError: assignment destination is read-only
d5[2] = 11
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
<ipython-input-713-a3b6c959d563> in <module>
----> 1 d5[2] = 11

ValueError: assignment destination is read-only
Load & Save
# Load data from a text file using loadtext
p4 = np.loadtxt('sample.txt', 
                dtype = np.integer # Decides the datatype of resulting array
               )
p4
array([[24, 29, 88],
       [ 1,  0,  8],
       [33,  7, 99],
       [39, 11, 98],
       [22, 76, 87]])
# Load data from a text file using genfromtxt
p5 = np.genfromtxt('sample0.txt',dtype='str')
p5
array([['Asif', 'India', 'Cricket'],
       ['John', 'USA', 'Hockey'],
       ['Ramiro', 'Canada', 'Football']], dtype='<U8')
# Accessing specific rows
p5[0]
array(['Asif', 'India', 'Cricket'], dtype='<U8')
# Accessing specific columns
p5[:,0]
array(['Asif', 'John', 'Ramiro'], dtype='<U8')
p6 = np.genfromtxt('sample2.txt', 
                   delimiter=' ', 
                   dtype=None, 
                   names=('Name', 'ID', 'Age')
                  )
p6
array([(b'Name', b'ID', b'Age'), (b'Asif', b'22', b'29'),
       (b'John', b'45', b'33'), (b'Ramiro', b'55', b'67'),
       (b'Michael', b'67', b'55'), (b'Klaus', b'44', b'32'),
       (b'Sajad', b'23', b'53')],
      dtype=[('Name', 'S7'), ('ID', 'S2'), ('Age', 'S3')])
# Skip header using "skiprows" parameter
p6 = np.loadtxt('sample2.txt', 
                   delimiter=' ', 
                   dtype=[('Name', str, 50), ('ID', np.integer), ('Age', np.integer)], 
                   skiprows=1
                  )
p6
array([('Asif', 22, 29), ('John', 45, 33), ('Ramiro', 55, 67),
       ('Michael', 67, 55), ('Klaus', 44, 32), ('Sajad', 23, 53)],
      dtype=[('Name', '<U50'), ('ID', '<i4'), ('Age', '<i4')])
# Return only first & third column using "usecols" parameter
np.loadtxt('sample.txt', delimiter =' ', usecols =(0, 2)) 
array([[24., 88.],
       [ 1.,  8.],
       [33., 99.],
       [39., 98.],
       [22., 87.]])
# Return only three rows using "max_rows" parameter
p6 = np.loadtxt('sample2.txt', 
                   delimiter=' ', 
                   dtype=[('Name', str, 50), ('ID', np.integer), ('Age', np.integer)], 
                   skiprows=1,
                   max_rows = 3
                  )
p6
array([('Asif', 22, 29), ('John', 45, 33), ('Ramiro', 55, 67)],
      dtype=[('Name', '<U50'), ('ID', '<i4'), ('Age', '<i4')])
# Skip header using "skip_header" parameter
p6 = np.genfromtxt('sample2.txt', 
                   delimiter=' ', 
                   dtype=[('Name', str, 50), ('ID', np.integer), ('Age', np.float)], 
                   names=('Name', 'ID', 'Age'),
                   skip_header=1
                  )
p6
array([('Asif', 22, 29.), ('John', 45, 33.), ('Ramiro', 55, 67.),
       ('Michael', 67, 55.), ('Klaus', 44, 32.), ('Sajad', 23, 53.)],
      dtype=[('Name', '<U50'), ('ID', '<i4'), ('Age', '<f8')])
p7 = np.arange(10,200,11)
p7
array([ 10,  21,  32,  43,  54,  65,  76,  87,  98, 109, 120, 131, 142,
       153, 164, 175, 186, 197])
np.savetxt('test3.csv', p7, delimiter=',')
p8 = np.arange(0,121).reshape(11,11)
p8
array([[  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10],
       [ 11,  12,  13,  14,  15,  16,  17,  18,  19,  20,  21],
       [ 22,  23,  24,  25,  26,  27,  28,  29,  30,  31,  32],
       [ 33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43],
       [ 44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54],
       [ 55,  56,  57,  58,  59,  60,  61,  62,  63,  64,  65],
       [ 66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76],
       [ 77,  78,  79,  80,  81,  82,  83,  84,  85,  86,  87],
       [ 88,  89,  90,  91,  92,  93,  94,  95,  96,  97,  98],
       [ 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
       [110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120]])
np.save('test4.npy', p8)
p9 = np.load('test4.npy')
p9
array([[  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10],
       [ 11,  12,  13,  14,  15,  16,  17,  18,  19,  20,  21],
       [ 22,  23,  24,  25,  26,  27,  28,  29,  30,  31,  32],
       [ 33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43],
       [ 44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54],
       [ 55,  56,  57,  58,  59,  60,  61,  62,  63,  64,  65],
       [ 66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76],
       [ 77,  78,  79,  80,  81,  82,  83,  84,  85,  86,  87],
       [ 88,  89,  90,  91,  92,  93,  94,  95,  96,  97,  98],
       [ 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
       [110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120]])
np.save('numpyfile', p8)
p10 = np.load('numpyfile.npy')
p10
array([[  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10],
       [ 11,  12,  13,  14,  15,  16,  17,  18,  19,  20,  21],
       [ 22,  23,  24,  25,  26,  27,  28,  29,  30,  31,  32],
       [ 33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43],
       [ 44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54],
       [ 55,  56,  57,  58,  59,  60,  61,  62,  63,  64,  65],
       [ 66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76],
       [ 77,  78,  79,  80,  81,  82,  83,  84,  85,  86,  87],
       [ 88,  89,  90,  91,  92,  93,  94,  95,  96,  97,  98],
       [ 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
       [110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120]])
p11 = np.arange(0,1000000).reshape(1000,1000)
p11
array([[     0,      1,      2, ...,    997,    998,    999],
       [  1000,   1001,   1002, ...,   1997,   1998,   1999],
       [  2000,   2001,   2002, ...,   2997,   2998,   2999],
       ...,
       [997000, 997001, 997002, ..., 997997, 997998, 997999],
       [998000, 998001, 998002, ..., 998997, 998998, 998999],
       [999000, 999001, 999002, ..., 999997, 999998, 999999]])
# Save Numpy array to a compressed file
np.savez_compressed('test6.npz', p11)
# Save Numpy array to a npy file
np.save('test7.npy', p11)
# Compressed file size is much lesser than normal npy file
Image(filename='load_save.PNG')

Printing Options
# Display values upto 4 decimal place
np.set_printoptions(precision=4)
a = np.array([12.654398765 , 90.7864098354674])
a
array([12.6544, 90.7864])
# Display values upto 2 decimal place
np.set_printoptions(precision=2)
a = np.array([12.654398765 , 90.7864098354674])
a
array([12.65, 90.79])
# Array Summarization
np.set_printoptions(threshold=3)
np.arange(200)
array([  0,   1,   2, ..., 197, 198, 199])
# Reset Formatter
np.set_printoptions(precision=8,suppress=False, threshold=1000, formatter=None)
a = np.array([12.654398765 , 90.7864098354674])
a
array([12.65439876, 90.78640984])
np.arange(1,1100)
array([   1,    2,    3, ..., 1097, 1098, 1099])
# Display all values
np.set_printoptions(threshold=np.inf)
np.arange(1,1100)
array([   1,    2,    3,    4,    5,    6,    7,    8,    9,   10,   11,
         12,   13,   14,   15,   16,   17,   18,   19,   20,   21,   22,
         23,   24,   25,   26,   27,   28,   29,   30,   31,   32,   33,
         34,   35,   36,   37,   38,   39,   40,   41,   42,   43,   44,
         45,   46,   47,   48,   49,   50,   51,   52,   53,   54,   55,
         56,   57,   58,   59,   60,   61,   62,   63,   64,   65,   66,
         67,   68,   69,   70,   71,   72,   73,   74,   75,   76,   77,
         78,   79,   80,   81,   82,   83,   84,   85,   86,   87,   88,
         89,   90,   91,   92,   93,   94,   95,   96,   97,   98,   99,
        100,  101,  102,  103,  104,  105,  106,  107,  108,  109,  110,
        111,  112,  113,  114,  115,  116,  117,  118,  119,  120,  121,
        122,  123,  124,  125,  126,  127,  128,  129,  130,  131,  132,
        133,  134,  135,  136,  137,  138,  139,  140,  141,  142,  143,
        144,  145,  146,  147,  148,  149,  150,  151,  152,  153,  154,
        155,  156,  157,  158,  159,  160,  161,  162,  163,  164,  165,
        166,  167,  168,  169,  170,  171,  172,  173,  174,  175,  176,
        177,  178,  179,  180,  181,  182,  183,  184,  185,  186,  187,
        188,  189,  190,  191,  192,  193,  194,  195,  196,  197,  198,
        199,  200,  201,  202,  203,  204,  205,  206,  207,  208,  209,
        210,  211,  212,  213,  214,  215,  216,  217,  218,  219,  220,
        221,  222,  223,  224,  225,  226,  227,  228,  229,  230,  231,
        232,  233,  234,  235,  236,  237,  238,  239,  240,  241,  242,
        243,  244,  245,  246,  247,  248,  249,  250,  251,  252,  253,
        254,  255,  256,  257,  258,  259,  260,  261,  262,  263,  264,
        265,  266,  267,  268,  269,  270,  271,  272,  273,  274,  275,
        276,  277,  278,  279,  280,  281,  282,  283,  284,  285,  286,
        287,  288,  289,  290,  291,  292,  293,  294,  295,  296,  297,
        298,  299,  300,  301,  302,  303,  304,  305,  306,  307,  308,
        309,  310,  311,  312,  313,  314,  315,  316,  317,  318,  319,
        320,  321,  322,  323,  324,  325,  326,  327,  328,  329,  330,
        331,  332,  333,  334,  335,  336,  337,  338,  339,  340,  341,
        342,  343,  344,  345,  346,  347,  348,  349,  350,  351,  352,
        353,  354,  355,  356,  357,  358,  359,  360,  361,  362,  363,
        364,  365,  366,  367,  368,  369,  370,  371,  372,  373,  374,
        375,  376,  377,  378,  379,  380,  381,  382,  383,  384,  385,
        386,  387,  388,  389,  390,  391,  392,  393,  394,  395,  396,
        397,  398,  399,  400,  401,  402,  403,  404,  405,  406,  407,
        408,  409,  410,  411,  412,  413,  414,  415,  416,  417,  418,
        419,  420,  421,  422,  423,  424,  425,  426,  427,  428,  429,
        430,  431,  432,  433,  434,  435,  436,  437,  438,  439,  440,
        441,  442,  443,  444,  445,  446,  447,  448,  449,  450,  451,
        452,  453,  454,  455,  456,  457,  458,  459,  460,  461,  462,
        463,  464,  465,  466,  467,  468,  469,  470,  471,  472,  473,
        474,  475,  476,  477,  478,  479,  480,  481,  482,  483,  484,
        485,  486,  487,  488,  489,  490,  491,  492,  493,  494,  495,
        496,  497,  498,  499,  500,  501,  502,  503,  504,  505,  506,
        507,  508,  509,  510,  511,  512,  513,  514,  515,  516,  517,
        518,  519,  520,  521,  522,  523,  524,  525,  526,  527,  528,
        529,  530,  531,  532,  533,  534,  535,  536,  537,  538,  539,
        540,  541,  542,  543,  544,  545,  546,  547,  548,  549,  550,
        551,  552,  553,  554,  555,  556,  557,  558,  559,  560,  561,
        562,  563,  564,  565,  566,  567,  568,  569,  570,  571,  572,
        573,  574,  575,  576,  577,  578,  579,  580,  581,  582,  583,
        584,  585,  586,  587,  588,  589,  590,  591,  592,  593,  594,
        595,  596,  597,  598,  599,  600,  601,  602,  603,  604,  605,
        606,  607,  608,  609,  610,  611,  612,  613,  614,  615,  616,
        617,  618,  619,  620,  621,  622,  623,  624,  625,  626,  627,
        628,  629,  630,  631,  632,  633,  634,  635,  636,  637,  638,
        639,  640,  641,  642,  643,  644,  645,  646,  647,  648,  649,
        650,  651,  652,  653,  654,  655,  656,  657,  658,  659,  660,
        661,  662,  663,  664,  665,  666,  667,  668,  669,  670,  671,
        672,  673,  674,  675,  676,  677,  678,  679,  680,  681,  682,
        683,  684,  685,  686,  687,  688,  689,  690,  691,  692,  693,
        694,  695,  696,  697,  698,  699,  700,  701,  702,  703,  704,
        705,  706,  707,  708,  709,  710,  711,  712,  713,  714,  715,
        716,  717,  718,  719,  720,  721,  722,  723,  724,  725,  726,
        727,  728,  729,  730,  731,  732,  733,  734,  735,  736,  737,
        738,  739,  740,  741,  742,  743,  744,  745,  746,  747,  748,
        749,  750,  751,  752,  753,  754,  755,  756,  757,  758,  759,
        760,  761,  762,  763,  764,  765,  766,  767,  768,  769,  770,
        771,  772,  773,  774,  775,  776,  777,  778,  779,  780,  781,
        782,  783,  784,  785,  786,  787,  788,  789,  790,  791,  792,
        793,  794,  795,  796,  797,  798,  799,  800,  801,  802,  803,
        804,  805,  806,  807,  808,  809,  810,  811,  812,  813,  814,
        815,  816,  817,  818,  819,  820,  821,  822,  823,  824,  825,
        826,  827,  828,  829,  830,  831,  832,  833,  834,  835,  836,
        837,  838,  839,  840,  841,  842,  843,  844,  845,  846,  847,
        848,  849,  850,  851,  852,  853,  854,  855,  856,  857,  858,
        859,  860,  861,  862,  863,  864,  865,  866,  867,  868,  869,
        870,  871,  872,  873,  874,  875,  876,  877,  878,  879,  880,
        881,  882,  883,  884,  885,  886,  887,  888,  889,  890,  891,
        892,  893,  894,  895,  896,  897,  898,  899,  900,  901,  902,
        903,  904,  905,  906,  907,  908,  909,  910,  911,  912,  913,
        914,  915,  916,  917,  918,  919,  920,  921,  922,  923,  924,
        925,  926,  927,  928,  929,  930,  931,  932,  933,  934,  935,
        936,  937,  938,  939,  940,  941,  942,  943,  944,  945,  946,
        947,  948,  949,  950,  951,  952,  953,  954,  955,  956,  957,
        958,  959,  960,  961,  962,  963,  964,  965,  966,  967,  968,
        969,  970,  971,  972,  973,  974,  975,  976,  977,  978,  979,
        980,  981,  982,  983,  984,  985,  986,  987,  988,  989,  990,
        991,  992,  993,  994,  995,  996,  997,  998,  999, 1000, 1001,
       1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012,
       1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023,
       1024, 1025, 1026, 1027, 1028, 1029, 1030, 1031, 1032, 1033, 1034,
       1035, 1036, 1037, 1038, 1039, 1040, 1041, 1042, 1043, 1044, 1045,
       1046, 1047, 1048, 1049, 1050, 1051, 1052, 1053, 1054, 1055, 1056,
       1057, 1058, 1059, 1060, 1061, 1062, 1063, 1064, 1065, 1066, 1067,
       1068, 1069, 1070, 1071, 1072, 1073, 1074, 1075, 1076, 1077, 1078,
       1079, 1080, 1081, 1082, 1083, 1084, 1085, 1086, 1087, 1088, 1089,
       1090, 1091, 1092, 1093, 1094, 1095, 1096, 1097, 1098, 1099])
Vector Addition
v1 = np.array([1,2])
v2 = np.array([3,4])
v3 = v1+v2
v3 = np.add(v1,v2)
print('V3 =' ,v3)
V3 = [4 6]
Multiplication of vectors
a1 = [5 , 6 ,8]
a2 = [4, 7 , 9]
print(np.multiply(a1,a2))
[20 42 72]
Dot Product
https://www.youtube.com/watch?v=WNuIhXo39_k

https://www.youtube.com/watch?v=LyGKycYT2v0

a1 = np.array([1,2,3])
a2 = np.array([4,5,6])

dotp = a1@a2
print(" Dot product - ",dotp)

dotp = np.dot(a1,a2)
print(" Dot product usign np.dot",dotp)

dotp = np.inner(a1,a2)
print(" Dot product usign np.inner", dotp)

dotp = sum(np.multiply(a1,a2))
print(" Dot product usign np.multiply & sum",dotp)

dotp = np.matmul(a1,a2)
print(" Dot product usign np.matmul",dotp)

dotp = 0
for i in range(len(a1)):
    dotp = dotp + a1[i]*a2[i]
print(" Dot product usign for loop" , dotp)
 Dot product -  32
 Dot product usign np.dot 32
 Dot product usign np.inner 32
 Dot product usign np.multiply & sum 32
 Dot product usign np.matmul 32
 Dot product usign for loop 32
Length of Vector
v3 = np.array([1,2,3,4,5,6])
length = np.sqrt(np.dot(v3,v3))
length
9.539392014169456
v3 = np.array([1,2,3,4,5,6])
length = np.sqrt(sum(np.multiply(v3,v3)))
length
9.539392014169456
v3 = np.array([1,2,3,4,5,6])
length = np.sqrt(np.matmul(v3,v3))
length
9.539392014169456
Normalized Vector
How to normalize a vector : https://www.youtube.com/watch?v=7fn03DIW3Ak

#First Method
v1 = [2,3]
length_v1 = np.sqrt(np.dot(v1,v1))
norm_v1 = v1/length_v1
length_v1 , norm_v1
(3.605551275463989, array([0.5547002 , 0.83205029]))
#Second Method
v1 = [2,3]
norm_v1 = v1/np.linalg.norm(v1)
norm_v1
array([0.5547002 , 0.83205029])
Angle between vectors
#First Method
v1 = np.array([8,4])
v2 = np.array([-4,8])
ang = np.rad2deg(np.arccos( np.dot(v1,v2) / (np.linalg.norm(v1)*np.linalg.norm(v2))))
ang
90.0
#Second Method
v1 = np.array([4,3])
v2 = np.array([-3,4])
lengthV1 = np.sqrt(np.dot(v1,v1)) 
lengthV2  = np.sqrt(np.dot(v2,v2))
ang = np.rad2deg(np.arccos( np.dot(v1,v2) / (lengthV1 * lengthV2)))
print('Angle between Vectors - %s' %ang)
Angle between Vectors - 90.0
Inner & outer products
Inner and Outer Product :

https://www.youtube.com/watch?v=FCmH4MqbFGs&t=2s

https://www.youtube.com/watch?v=FCmH4MqbFGs

v1 = np.array([1,2,3])
v2 = np.array([4,5,6])
np.inner(v1,v2)

print("\n Inner Product ==>  \n", np.inner(v1,v2))
print("\n Outer Product ==>  \n", np.outer(v1,v2))
 Inner Product ==>  
 32

 Outer Product ==>  
 [[ 4  5  6]
 [ 8 10 12]
 [12 15 18]]
Vector Cross Product
v1 = np.array([1,2,3])
v2 = np.array([4,5,6])
print("\nVector Cross Product ==>  \n", np.cross(v1,v2))
Vector Cross Product ==>  
 [-3  6 -3]
Matrix Creation
# Create a 4x4 matrix
A = np.array([[1,2,3,4] , [5,6,7,8] , [10 , 11 , 12 ,13] , [14,15,16,17]])
A
array([[ 1,  2,  3,  4],
       [ 5,  6,  7,  8],
       [10, 11, 12, 13],
       [14, 15, 16, 17]])
# Datatype of Matrix
A.dtype
dtype('int32')
B = np.array([[1.5,2.07,3,4] , [5,6,7,8] , [10 , 11 , 12 ,13] , [14,15,16,17]])
B
array([[ 1.5 ,  2.07,  3.  ,  4.  ],
       [ 5.  ,  6.  ,  7.  ,  8.  ],
       [10.  , 11.  , 12.  , 13.  ],
       [14.  , 15.  , 16.  , 17.  ]])
# Datatype of Matrix
B.dtype
dtype('float64')
# Shape of Matrix
A.shape
(4, 4)
# Generate a 4x4 zero matrix
np.zeros((4,4))
array([[0., 0., 0., 0.],
       [0., 0., 0., 0.],
       [0., 0., 0., 0.],
       [0., 0., 0., 0.]])
#Shape of Matrix
z1 = np.zeros((4,4))
z1.shape
(4, 4)
# Generate a 5x5 matrix filled with ones
np.ones((5,5))
array([[1., 1., 1., 1., 1.],
       [1., 1., 1., 1., 1.],
       [1., 1., 1., 1., 1.],
       [1., 1., 1., 1., 1.],
       [1., 1., 1., 1., 1.]])
# Return 10x10 matrix of random integer numbers between 0 to 500
np.random.randint(0,500, (10,10))
array([[229, 366,  71, 357, 452, 244, 407, 163, 207, 226],
       [451, 338, 441, 461,  46, 131,  46, 485, 285, 470],
       [149, 378,  21, 465,  23, 235, 254, 383,  94, 356],
       [199, 276,  27, 459,   5, 305, 470, 217, 191,  82],
       [ 77, 358, 131, 184, 383, 142, 383,  49, 343,  52],
       [253, 397, 431, 433, 280, 404, 448, 180, 316, 303],
       [370, 285, 316, 309, 395,  40, 219, 301,  97, 408],
       [292, 166, 137, 125,  52,  67, 299, 129,  79,  68],
       [196, 484,  61, 146, 307, 270, 412, 401,  87,  46],
       [ 52, 144, 454, 455,  84,  10, 190, 362,  96, 122]])
arr2
array([644, 575, 936, 757, 316, 732, 704, 110,   5, 908, 477,  40,  49,
       851, 623, 506, 136, 371, 925, 883])
arr2.reshape(5,4)
array([[644, 575, 936, 757],
       [316, 732, 704, 110],
       [  5, 908, 477,  40],
       [ 49, 851, 623, 506],
       [136, 371, 925, 883]])
mat1 = np.random.randint(0,1000,100).reshape(10,10)
mat1
array([[ 92, 907, 507, 394, 625, 478, 419, 540,   3, 851],
       [340, 303, 526, 250, 709, 505, 956, 197, 632, 947],
       [262, 984, 103, 229, 366,  71, 357, 964, 244, 919],
       [675, 207, 226, 451, 850, 953, 461,  46, 643, 558],
       [508, 997, 797, 470, 149, 378,  21, 465, 535, 235],
       [254, 383,  94, 356, 711, 788, 539, 971,   5, 305],
       [982, 217, 703,  82, 589, 358, 643, 696, 895, 654],
       [383, 561, 855,  52, 253, 397, 943, 945, 280, 404],
       [960, 692, 828, 815, 370, 285, 828, 309, 395,  40],
       [219, 813, 609, 920, 804, 678, 649, 125, 564,  67]])
mat1[0,0]
644
mat1[mat1 > 500]
array([644, 575, 936, 757, 732, 704, 908, 851, 623, 506, 925, 883, 556,
       840, 638, 906, 735, 619, 896, 503, 574, 676, 979, 831, 519, 906,
       615, 750, 503, 615, 911, 512, 628, 760, 865, 989, 664, 676, 892,
       703, 542, 956, 615, 923, 776, 854, 794, 855, 686, 950, 741, 685,
       570])
# Identity Matrix : https://en.wikipedia.org/wiki/Identity_matrix

I = np.eye(9)
I
array([[1., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 1., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 1., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 1., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 1., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 1., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 1., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 1., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 1.]])
# Diagonal Matrix : https://en.wikipedia.org/wiki/Diagonal_matrix

D = np.diag([1,2,3,4,5,6,7,8])
D
array([[1, 0, 0, 0, 0, 0, 0, 0],
       [0, 2, 0, 0, 0, 0, 0, 0],
       [0, 0, 3, 0, 0, 0, 0, 0],
       [0, 0, 0, 4, 0, 0, 0, 0],
       [0, 0, 0, 0, 5, 0, 0, 0],
       [0, 0, 0, 0, 0, 6, 0, 0],
       [0, 0, 0, 0, 0, 0, 7, 0],
       [0, 0, 0, 0, 0, 0, 0, 8]])
# Traingular Matrices (lower & Upper triangular matrix) : https://en.wikipedia.org/wiki/Triangular_matrix

M = np.random.randn(5,5)
U = np.triu(M)
L = np.tril(M)
print("lower triangular matrix - \n" , M)
print("\n")


print("lower triangular matrix - \n" , L)
print("\n")

print("Upper triangular matrix - \n" , U)
lower triangular matrix - 
 [[ 0.65111795 -0.31931804 -0.84807698  0.60596535 -2.01816824]
 [ 0.74012206  0.52881349 -0.58900053  0.18869531 -0.75887206]
 [-0.93323722  0.95505651  0.19079432  1.97875732  2.60596728]
 [ 0.68350889  0.30266545  1.69372293 -1.70608593 -1.15911942]
 [-0.13484072  0.39052784  0.16690464  0.18450186  0.80770591]]


lower triangular matrix - 
 [[ 0.65111795  0.          0.          0.          0.        ]
 [ 0.74012206  0.52881349  0.          0.          0.        ]
 [-0.93323722  0.95505651  0.19079432  0.          0.        ]
 [ 0.68350889  0.30266545  1.69372293 -1.70608593  0.        ]
 [-0.13484072  0.39052784  0.16690464  0.18450186  0.80770591]]


Upper triangular matrix - 
 [[ 0.65111795 -0.31931804 -0.84807698  0.60596535 -2.01816824]
 [ 0.          0.52881349 -0.58900053  0.18869531 -0.75887206]
 [ 0.          0.          0.19079432  1.97875732  2.60596728]
 [ 0.          0.          0.         -1.70608593 -1.15911942]
 [ 0.          0.          0.          0.          0.80770591]]
# Generate a 5X5 matrix with a given fill value of 8
np.full((5,5) , 8)
array([[8, 8, 8, 8, 8],
       [8, 8, 8, 8, 8],
       [8, 8, 8, 8, 8],
       [8, 8, 8, 8, 8],
       [8, 8, 8, 8, 8]])
# Generate 5X5 matrix of Random float numbers between 10 to 20
np.random.uniform(10,20, size=(5,5))
array([[13.51434265, 17.33567613, 19.13889527, 17.00987494, 13.88531272],
       [19.42259289, 17.36491331, 12.38464388, 18.23773728, 17.60613445],
       [13.94709074, 12.00187917, 17.12596473, 18.45308897, 13.68646541],
       [14.36980119, 13.56597664, 12.39737407, 16.53378141, 13.90439201],
       [16.57783018, 13.62273355, 13.56502014, 11.952516  , 19.87312751]])
A
array([[ 1,  2,  3,  4],
       [ 5,  6,  7,  8],
       [10, 11, 12, 13],
       [14, 15, 16, 17]])
# Collapse Matrix into one dimension array
A.flatten()
array([ 1,  2,  3,  4,  5,  6,  7,  8, 10, 11, 12, 13, 14, 15, 16, 17])
# Collapse Matrix into one dimension array
A.ravel()
array([ 1,  2,  3,  4,  5,  6,  7,  8, 10, 11, 12, 13, 14, 15, 16, 17])
Reading elements of a Matrix
A
array([[ 1,  2,  3,  4],
       [ 5,  6,  7,  8],
       [10, 11, 12, 13],
       [14, 15, 16, 17]])
# Fetch first row of matrix
A[0,]
array([1, 2, 3, 4])
# Fetch first column of matrix
A[:,0]
array([ 1,  5, 10, 14])
# Fetch first element of the matrix
A[0,0]
1
A[1:3 , 1:3]
array([[ 6,  7],
       [11, 12]])
Reverse Rows / Columns of a Matrix
arr = np.arange(16).reshape(4,4)
arr
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11],
       [12, 13, 14, 15]])
# Reverse rows
arr[::-1]
array([[12, 13, 14, 15],
       [ 8,  9, 10, 11],
       [ 4,  5,  6,  7],
       [ 0,  1,  2,  3]])
#Reverse Columns
arr[:, ::-1]
array([[ 3,  2,  1,  0],
       [ 7,  6,  5,  4],
       [11, 10,  9,  8],
       [15, 14, 13, 12]])
SWAP Rows & Columns
m1 = np.arange(0,16).reshape(4,4)
m1
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11],
       [12, 13, 14, 15]])
# SWAP rows 0 & 1
m1[[0,1]] = m1[[1,0]]
m1
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11],
       [12, 13, 14, 15]])
# SWAP rows 2 & 3
m1[[3,2]] = m1[[2,3]]
m1
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [12, 13, 14, 15],
       [ 8,  9, 10, 11]])
m2 = np.arange(0,36).reshape(6,6)
m2
array([[ 0,  1,  2,  3,  4,  5],
       [ 6,  7,  8,  9, 10, 11],
       [12, 13, 14, 15, 16, 17],
       [18, 19, 20, 21, 22, 23],
       [24, 25, 26, 27, 28, 29],
       [30, 31, 32, 33, 34, 35]])
# Swap columns 0 & 1
m2[:,[0, 1]] = m2[:,[1, 0]]
m2
array([[ 6,  0,  2,  3,  4,  5],
       [ 7,  6,  8,  9, 10, 11],
       [13, 12, 14, 15, 16, 17],
       [19, 18, 20, 21, 22, 23],
       [25, 24, 26, 27, 28, 29],
       [31, 30, 32, 33, 34, 35]])
# Swap columns 2 & 3
m2[:,[2, 3]] = m2[:,[3, 2]]
m2
array([[ 6,  0,  3,  2,  4,  5],
       [ 7,  6,  9,  8, 10, 11],
       [13, 12, 15, 14, 16, 17],
       [19, 18, 21, 20, 22, 23],
       [25, 24, 27, 26, 28, 29],
       [31, 30, 33, 32, 34, 35]])
Concatenate Matrices
Matrix Concatenation : https://docs.scipy.org/doc/numpy/reference/generated/numpy.concatenate.html

A = np.array([[1,2] , [3,4] ,[5,6]])
B = np.array([[1,1] , [1,1]])
C = np.concatenate((A,B))
C
array([[1, 2],
       [3, 4],
       [5, 6],
       [1, 1],
       [1, 1]])
Matrix Addition
Matrix Addition : https://www.youtube.com/watch?v=ZCmVpGv6_1g

#********************************************************#
M = np.array([[1,2,3],[4,-3,6],[7,8,0]])
N = np.array([[1,1,1],[2,2,2],[3,3,3]])

print("\n First Matrix (M)  ==>  \n", M)
print("\n Second Matrix (N)  ==>  \n", N)

C = M+N
print("\n Matrix Addition (M+N)  ==>  \n", C)

# OR

C = np.add(M,N,dtype = np.float64)
print("\n Matrix Addition using np.add  ==>  \n", C)

#********************************************************#
 First Matrix (M)  ==>  
 [[ 1  2  3]
 [ 4 -3  6]
 [ 7  8  0]]

 Second Matrix (N)  ==>  
 [[1 1 1]
 [2 2 2]
 [3 3 3]]

 Matrix Addition (M+N)  ==>  
 [[ 2  3  4]
 [ 6 -1  8]
 [10 11  3]]

 Matrix Addition using np.add  ==>  
 [[ 2.  3.  4.]
 [ 6. -1.  8.]
 [10. 11.  3.]]
Matrix subtraction
Matrix subtraction : https://www.youtube.com/watch?v=7jb_AO_hRc8&list=PLmdFyQYShrjcoVkhCCIwxNj9N4rW1-T5I&index=8

#********************************************************#
M = np.array([[1,2,3],[4,-3,6],[7,8,0]])
N = np.array([[1,1,1],[2,2,2],[3,3,3]])

print("\n First Matrix (M)  ==>  \n", M)
print("\n Second Matrix (N)  ==>  \n", N)

C = M-N
print("\n Matrix Subtraction (M-N)  ==>  \n", C)

# OR

C = np.subtract(M,N,dtype = np.float64)
print("\n Matrix Subtraction using np.subtract  ==>  \n", C)

#********************************************************#
 First Matrix (M)  ==>  
 [[ 1  2  3]
 [ 4 -3  6]
 [ 7  8  0]]

 Second Matrix (N)  ==>  
 [[1 1 1]
 [2 2 2]
 [3 3 3]]

 Matrix Subtraction (M-N)  ==>  
 [[ 0  1  2]
 [ 2 -5  4]
 [ 4  5 -3]]

 Matrix Subtraction using np.subtract  ==>  
 [[ 0.  1.  2.]
 [ 2. -5.  4.]
 [ 4.  5. -3.]]
Matrices Scalar Multiplication
Matrices Scalar Multiplication : https://www.youtube.com/watch?v=4lHyTQH1iS8&list=PLmdFyQYShrjcoVkhCCIwxNj9N4rW1-T5I&index=9

M = np.array([[1,2,3],[4,-3,6],[7,8,0]])

C = 10

print("\n Matrix (M)  ==>  \n", M)

print("\nMatrices Scalar Multiplication ==>  \n", C*M)

# OR

print("\nMatrices Scalar Multiplication ==>  \n", np.multiply(C,M))
 Matrix (M)  ==>  
 [[ 1  2  3]
 [ 4 -3  6]
 [ 7  8  0]]

Matrices Scalar Multiplication ==>  
 [[ 10  20  30]
 [ 40 -30  60]
 [ 70  80   0]]

Matrices Scalar Multiplication ==>  
 [[ 10  20  30]
 [ 40 -30  60]
 [ 70  80   0]]
Transpose of a matrix
Transpose of a matrix : https://www.youtube.com/watch?v=g_Rz94DXvNo&list=PLmdFyQYShrjcoVkhCCIwxNj9N4rW1-T5I&index=13

M = np.array([[1,2,3],[4,-3,6],[7,8,0]])

print("\n Matrix (M)  ==>  \n", M)

print("\nTranspose of M ==>  \n", np.transpose(M))

# OR

print("\nTranspose of M ==>  \n", M.T)
 Matrix (M)  ==>  
 [[ 1  2  3]
 [ 4 -3  6]
 [ 7  8  0]]

Transpose of M ==>  
 [[ 1  4  7]
 [ 2 -3  8]
 [ 3  6  0]]

Transpose of M ==>  
 [[ 1  4  7]
 [ 2 -3  8]
 [ 3  6  0]]
Determinant of a matrix
Determinant of a matrix :

https://www.youtube.com/watch?v=21LWuY8i6Hw&t=88s

https://www.youtube.com/watch?v=Ip3X9LOh2dk&list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab&index=6

M = np.array([[1,2,3],[4,-3,6],[7,8,0]])

print("\n Matrix (M)  ==>  \n", M)

print("\nDeterminant of M ==>  ", np.linalg.det(M))
 Matrix (M)  ==>  
 [[ 1  2  3]
 [ 4 -3  6]
 [ 7  8  0]]

Determinant of M ==>   195.0
Rank of a matrix
M = np.array([[1,2,3],[4,-3,6],[7,8,0]])

print("\n Matrix (M)  ==>  \n", M)

print("\nRank of M ==> ", np.linalg.matrix_rank(M))
 Matrix (M)  ==>  
 [[ 1  2  3]
 [ 4 -3  6]
 [ 7  8  0]]

Rank of M ==>  3
Trace of matrix
M = np.array([[1,2,3],[4,-3,6],[7,8,0]])

print("\n Matrix (M)  ==>  \n", M)

print("\nTrace of M ==> ", np.trace(M))
 Matrix (M)  ==>  
 [[ 1  2  3]
 [ 4 -3  6]
 [ 7  8  0]]

Trace of M ==>  -2
Inverse of matrix A
Inverse of matrix : https://www.youtube.com/watch?v=pKZyszzmyeQ

M = np.array([[1,2,3],[4,-3,6],[7,8,0]])

print("\n Matrix (M)  ==>  \n", M)

print("\nInverse of M ==> \n", np.linalg.inv(M))
 Matrix (M)  ==>  
 [[ 1  2  3]
 [ 4 -3  6]
 [ 7  8  0]]

Inverse of M ==> 
 [[-0.24615385  0.12307692  0.10769231]
 [ 0.21538462 -0.10769231  0.03076923]
 [ 0.27179487  0.03076923 -0.05641026]]
Matrix Multiplication (pointwise multiplication)
M = np.array([[1,2,3],[4,-3,6],[7,8,0]])
N = np.array([[1,1,1],[2,2,2],[3,3,3]])

print("\n First Matrix (M)  ==>  \n", M)
print("\n Second Matrix (N)  ==>  \n", N)

print("\n Point-Wise Multiplication of M & N  ==> \n", M*N)

# OR

print("\n Point-Wise Multiplication of M & N  ==> \n", np.multiply(M,N))
 First Matrix (M)  ==>  
 [[ 1  2  3]
 [ 4 -3  6]
 [ 7  8  0]]

 Second Matrix (N)  ==>  
 [[1 1 1]
 [2 2 2]
 [3 3 3]]

 Point-Wise Multiplication of M & N  ==> 
 [[ 1  2  3]
 [ 8 -6 12]
 [21 24  0]]

 Point-Wise Multiplication of M & N  ==> 
 [[ 1  2  3]
 [ 8 -6 12]
 [21 24  0]]
Matrix dot product
Matrix Multiplication :

https://www.youtube.com/watch?v=vzt9c7iWPxs&t=207s

https://www.youtube.com/watch?v=XkY2DOUCWMU&list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab&index=4

M = np.array([[1,2,3],[4,-3,6],[7,8,0]])
N = np.array([[1,1,1],[2,2,2],[3,3,3]])

print("\n First Matrix (M)  ==>  \n", M)
print("\n Second Matrix (N)  ==>  \n", N)

print("\n Matrix Dot Product ==> \n", M@N)

# OR

print("\n Matrix Dot Product using np.matmul ==> \n", np.matmul(M,N))

# OR

print("\n Matrix Dot Product using np.dot ==> \n", np.dot(M,N))
 First Matrix (M)  ==>  
 [[ 1  2  3]
 [ 4 -3  6]
 [ 7  8  0]]

 Second Matrix (N)  ==>  
 [[1 1 1]
 [2 2 2]
 [3 3 3]]

 Matrix Dot Product ==> 
 [[14 14 14]
 [16 16 16]
 [23 23 23]]

 Matrix Dot Product using np.matmul ==> 
 [[14 14 14]
 [16 16 16]
 [23 23 23]]

 Matrix Dot Product using np.dot ==> 
 [[14 14 14]
 [16 16 16]
 [23 23 23]]
Matrix Division
M = np.array([[1,2,3],[4,-3,6],[7,8,0]])
N = np.array([[1,1,1],[2,2,2],[3,3,3]])

print("\n First Matrix (M)  ==>  \n", M)
print("\n Second Matrix (N)  ==>  \n", N)


print("\n Matrix Division (M/N)   ==> \n", M/N)

# OR

print("\n Matrix Division (M/N)   ==> \n", np.divide(M,N))
 First Matrix (M)  ==>  
 [[ 1  2  3]
 [ 4 -3  6]
 [ 7  8  0]]

 Second Matrix (N)  ==>  
 [[1 1 1]
 [2 2 2]
 [3 3 3]]

 Matrix Division (M/N)   ==> 
 [[ 1.          2.          3.        ]
 [ 2.         -1.5         3.        ]
 [ 2.33333333  2.66666667  0.        ]]

 Matrix Division (M/N)   ==> 
 [[ 1.          2.          3.        ]
 [ 2.         -1.5         3.        ]
 [ 2.33333333  2.66666667  0.        ]]
Sum of all elements in a matrix
N = np.array([[1,1,1],[2,2,2],[3,3,3]])

print("\n Matrix (N)  ==>  \n", N)


print ("Sum of all elements in a Matrix  ==>")
print (np.sum(N))
 Matrix (N)  ==>  
 [[1 1 1]
 [2 2 2]
 [3 3 3]]
Sum of all elements in a Matrix  ==>
18
Column-Wise Addition
N = np.array([[1,1,1],[2,2,2],[3,3,3]])

print("\n Matrix (N)  ==>  \n", N)

print ("Column-Wise summation ==> ")
print (np.sum(N,axis=0))
 Matrix (N)  ==>  
 [[1 1 1]
 [2 2 2]
 [3 3 3]]
Column-Wise summation ==> 
[6 6 6]
Row-Wise Addition
N = np.array([[1,1,1],[2,2,2],[3,3,3]])

print("\n Matrix (N)  ==>  \n", N)

print ("Row-Wise summation  ==>")
print (np.sum(N,axis=1))
 Matrix (N)  ==>  
 [[1 1 1]
 [2 2 2]
 [3 3 3]]
Row-Wise summation  ==>
[3 6 9]
Kronecker Product of matrices
Kronecker Product of matrices : https://www.youtube.com/watch?v=e1UJXvu8VZk

M1 = np.array([[1,2,3] , [4,5,6]]) 
M1
array([[1, 2, 3],
       [4, 5, 6]])
M2 = np.array([[10,10,10],[10,10,10]])
M2
array([[10, 10, 10],
       [10, 10, 10]])
np.kron(M1,M2)
array([[10, 10, 10, 20, 20, 20, 30, 30, 30],
       [10, 10, 10, 20, 20, 20, 30, 30, 30],
       [40, 40, 40, 50, 50, 50, 60, 60, 60],
       [40, 40, 40, 50, 50, 50, 60, 60, 60]])
Matrix Powers
M1 = np.array([[1,2],[4,5]])
M1
array([[1, 2],
       [4, 5]])
#Matrix to the power 3

M1@M1@M1
array([[ 57,  78],
       [156, 213]])
#Matrix to the power 3

np.linalg.matrix_power(M1,3)
array([[ 57,  78],
       [156, 213]])
Tensor
What is Tensor :

https://www.youtube.com/watch?v=f5liqUk0ZTw
https://www.youtube.com/watch?v=bpG3gqDM80w&t=634s
https://www.youtube.com/watch?v=uaQeXi4E7gA
# Create Tensor

T1 = np.array([
  [[1,2,3],    [4,5,6],    [7,8,9]],
  [[10,20,30], [40,50,60], [70,80,90]],
  [[100,200,300], [400,500,600], [700,800,900]],
  ])

T1
array([[[  1,   2,   3],
        [  4,   5,   6],
        [  7,   8,   9]],

       [[ 10,  20,  30],
        [ 40,  50,  60],
        [ 70,  80,  90]],

       [[100, 200, 300],
        [400, 500, 600],
        [700, 800, 900]]])
T2 = np.array([
  [[0,0,0] , [0,0,0] , [0,0,0]],
  [[1,1,1] , [1,1,1] , [1,1,1]],
  [[2,2,2] , [2,2,2] , [2,2,2]]
    
])

T2
array([[[0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]],

       [[1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]],

       [[2, 2, 2],
        [2, 2, 2],
        [2, 2, 2]]])
Tensor Addition
A = T1+T2
A
array([[[  1,   2,   3],
        [  4,   5,   6],
        [  7,   8,   9]],

       [[ 11,  21,  31],
        [ 41,  51,  61],
        [ 71,  81,  91]],

       [[102, 202, 302],
        [402, 502, 602],
        [702, 802, 902]]])
np.add(T1,T2)
array([[[  1,   2,   3],
        [  4,   5,   6],
        [  7,   8,   9]],

       [[ 11,  21,  31],
        [ 41,  51,  61],
        [ 71,  81,  91]],

       [[102, 202, 302],
        [402, 502, 602],
        [702, 802, 902]]])
Tensor Subtraction
S = T1-T2
S
array([[[  1,   2,   3],
        [  4,   5,   6],
        [  7,   8,   9]],

       [[  9,  19,  29],
        [ 39,  49,  59],
        [ 69,  79,  89]],

       [[ 98, 198, 298],
        [398, 498, 598],
        [698, 798, 898]]])
np.subtract(T1,T2)
array([[[  1,   2,   3],
        [  4,   5,   6],
        [  7,   8,   9]],

       [[  9,  19,  29],
        [ 39,  49,  59],
        [ 69,  79,  89]],

       [[ 98, 198, 298],
        [398, 498, 598],
        [698, 798, 898]]])
Tensor Element-Wise Product
P = T1*T2
P
array([[[   0,    0,    0],
        [   0,    0,    0],
        [   0,    0,    0]],

       [[  10,   20,   30],
        [  40,   50,   60],
        [  70,   80,   90]],

       [[ 200,  400,  600],
        [ 800, 1000, 1200],
        [1400, 1600, 1800]]])
np.multiply(T1,T2)
array([[[   0,    0,    0],
        [   0,    0,    0],
        [   0,    0,    0]],

       [[  10,   20,   30],
        [  40,   50,   60],
        [  70,   80,   90]],

       [[ 200,  400,  600],
        [ 800, 1000, 1200],
        [1400, 1600, 1800]]])
Tensor Element-Wise Division
D = T1/T2
D
C:\Anaconda\lib\site-packages\ipykernel_launcher.py:1: RuntimeWarning: divide by zero encountered in true_divide
  """Entry point for launching an IPython kernel.
array([[[ inf,  inf,  inf],
        [ inf,  inf,  inf],
        [ inf,  inf,  inf]],

       [[ 10.,  20.,  30.],
        [ 40.,  50.,  60.],
        [ 70.,  80.,  90.]],

       [[ 50., 100., 150.],
        [200., 250., 300.],
        [350., 400., 450.]]])
np.divide(T1,T2)
C:\Anaconda\lib\site-packages\ipykernel_launcher.py:1: RuntimeWarning: divide by zero encountered in true_divide
  """Entry point for launching an IPython kernel.
array([[[ inf,  inf,  inf],
        [ inf,  inf,  inf],
        [ inf,  inf,  inf]],

       [[ 10.,  20.,  30.],
        [ 40.,  50.,  60.],
        [ 70.,  80.,  90.]],

       [[ 50., 100., 150.],
        [200., 250., 300.],
        [350., 400., 450.]]])
Tensor Dot Product
T1
array([[[  1,   2,   3],
        [  4,   5,   6],
        [  7,   8,   9]],

       [[ 10,  20,  30],
        [ 40,  50,  60],
        [ 70,  80,  90]],

       [[100, 200, 300],
        [400, 500, 600],
        [700, 800, 900]]])
T2
array([[[0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]],

       [[1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]],

       [[2, 2, 2],
        [2, 2, 2],
        [2, 2, 2]]])
np.tensordot(T1,T2)
array([[  63,   63,   63],
       [ 630,  630,  630],
       [6300, 6300, 6300]])
Solving Equations
Solving Equations :

https://www.youtube.com/watch?v=NNmiOoWt86M
https://www.youtube.com/watch?v=a2z7sZ4MSqo
A = np.array([[1,2,3] , [4,5,6] , [7,8,9]])
A
array([[1, 2, 3],
       [4, 5, 6],
       [7, 8, 9]])
B = np.random.random((3,1))
B
array([[0.09714648],
       [0.10284749],
       [0.7015073 ]])
# Ist Method
X = np.dot(np.linalg.inv(A) , B)
X
array([[ 1.86931429e+15],
       [-3.73862857e+15],
       [ 1.86931429e+15]])
# 2nd Method
X = np.matmul(np.linalg.inv(A) , B)
X
array([[ 1.86931429e+15],
       [-3.73862857e+15],
       [ 1.86931429e+15]])
# 3rd Method
X = np.linalg.inv(A)@B
X
array([[ 1.86931429e+15],
       [-3.73862857e+15],
       [ 1.86931429e+15]])
# 4th Method
X = np.linalg.solve(A,B)
X
array([[ 1.86931429e+15],
       [-3.73862857e+15],
       [ 1.86931429e+15]])
END
