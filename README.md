# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:

### FEATURE SCALING
```python
import pandas as pd
from scipy import stats
import numpy as np
df = pd.read_csv("bmi.csv")
df.head()
```
<img width="411" height="294" alt="image" src="https://github.com/user-attachments/assets/80b0a2ae-f85c-45b8-a31f-ad334b06f445" />

```python
df.dropna()
```
<img width="485" height="605" alt="image" src="https://github.com/user-attachments/assets/06c033ef-918b-43d1-bff8-44c6caa87a88" />

```python
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
df[['Height','Weight']]=sc.fit_transform(df[['Height','Weight']])
df.head(10)
```
<img width="457" height="520" alt="image" src="https://github.com/user-attachments/assets/a1d4d79b-4691-4f31-a402-f6cddb36beee" />

```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df[['Height','Weight']] = scaler.fit_transform(df[['Height','Weight']])
df.head(10)
```
<img width="418" height="519" alt="image" src="https://github.com/user-attachments/assets/0ea54be5-d8c0-46ca-a281-e77795858417" />

```python
from sklearn.preprocessing import Normalizer
scaler = Normalizer()
df1 = df.copy()
df1[['Height','Weight']] = scaler.fit_transform(df1[['Height','Weight']])
df1
```
<img width="454" height="570" alt="image" src="https://github.com/user-attachments/assets/450386ad-17fe-4b56-a3b2-fbc0b8e55277" />

```python
from sklearn.preprocessing import MaxAbsScaler
scaler = MaxAbsScaler()
df2 = df1.copy()
df2[['Height','Weight']] = scaler.fit_transform(df2[['Height','Weight']])
df2
```
<img width="465" height="599" alt="image" src="https://github.com/user-attachments/assets/b4b23a7f-4fa7-4e8a-9c52-7d6c174041e7" />

```python
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
df3 = df2.copy()
df3[['Height','Weight']] = scaler.fit_transform(df3[['Height','Weight']])
df3
```
<img width="465" height="596" alt="image" src="https://github.com/user-attachments/assets/085c366c-2161-4536-bd2c-12b2755c8680" />

### FEATURE SELECTION
```python








       
# RESULT:
       # INCLUDE YOUR RESULT HERE
