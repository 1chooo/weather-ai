# %% [markdown]
# # 20220421 機器學習實作2 資料清理數據前處理 - 2

# %% [markdown]
# ## Section 1

# %%
import os
import numpy as np
import pandas as pd

# %%
from google.colab import drive
drive.mount('/content/drive')

# %%
# 設定 data_path
dir_data = '/content/drive/MyDrive/中大講師/中大講師/中大 - 天氣與人工智慧/111-1/20221109 機器學習實作2 資料清理數據前處理/data'
f_app_train = os.path.join(dir_data, 'application_train.csv')
f_app_test = os.path.join(dir_data, 'application_test.csv')

app_train = pd.read_csv(f_app_train)
app_test = pd.read_csv(f_app_test)

# %% [markdown]
# 檢視資料中各個欄位類型的數量

# %%
app_train.head()

# %%
app_train.dtypes.value_counts()

# %% [markdown]
# 檢視資料中類別型欄位各自類別的數量

# %%
app_train.select_dtypes(include=["object"]).apply(pd.Series.nunique, axis = 0)

# %% [markdown]
# #### Label encoding
# 有仔細閱讀[參考資料](https://medium.com/@contactsunny/label-encoder-vs-one-hot-encoder-in-machine-learning-3fc273365621)的人可以發現，Label encoding 的表示方式會讓同一個欄位底下的類別之間有大小關係 (0<1<2<...)，所以在這裡我們只對有類別數量小於等於 2 的類別型欄位示範使用 Label encoding，但不表示這樣處理是最好的，一切取決於欄位本身的意義適合哪一種表示方法

# %%
from sklearn.preprocessing import LabelEncoder

# %%
# Create a label encoder object
le = LabelEncoder()
le_count = 0

# Iterate through the columns
for col in app_train:
    if app_train[col].dtype == 'object':
        # If 2 or fewer unique categories
        if len(list(app_train[col].unique())) <= 2:
            # Train on the training data
            le.fit(app_train[col])
            # Transform both training and testing data
            app_train[col] = le.transform(app_train[col])
            app_test[col] = le.transform(app_test[col])
            
            # Keep track of how many columns were label encoded
            le_count += 1
            
print('%d columns were label encoded.' % le_count)

# %% [markdown]
# #### One Hot encoding
# pandas 中的 one hot encoding 非常方便，一行程式碼就搞定

# %%
app_train = pd.get_dummies(app_train)
app_test = pd.get_dummies(app_test)

print(app_train['CODE_GENDER_F'].head())
print(app_train['CODE_GENDER_M'].head())
print(app_train['NAME_EDUCATION_TYPE_Academic degree'].head())

# %% [markdown]
# 可以觀察到原來的類別型欄位都轉為 0/1 了

# %% [markdown]
# ## 練習
# 將下列部分資料片段 sub_train 使用 One Hot encoding, 並觀察轉換前後的欄位數量 (使用 shape) 與欄位名稱 (使用 head) 變化
# 

# %%
app_train = pd.read_csv(f_app_train)
sub_train = pd.DataFrame(app_train['WEEKDAY_APPR_PROCESS_START'])
print(sub_train.shape)
sub_train.head()

# %%
sub_train = pd.get_dummies(sub_train)
print(sub_train.shape)
sub_train.head()

# %%


# %% [markdown]
# # Section 2 

# %% [markdown]
# # [練習目標]
# - 對資料做更多處理 : 顯示特定欄位的統計值與直方圖

# %% [markdown]
# # [練習重點]
# - 試著顯示特定欄位的基礎統計數值 
# - 試著顯示特定欄位的直方圖 

# %%
# Import 需要的套件
import os
import numpy as np
import pandas as pd

# 設定 data_path
dir_data = '/content/drive/MyDrive/中大講師/中大講師/中大 - 天氣與人工智慧/111-1/20221109 機器學習實作2 資料清理數據前處理/data'

# %%
f_app_train = os.path.join(dir_data, 'application_train.csv')
app_train = pd.read_csv(f_app_train)

# %%
import matplotlib.pyplot as plt
%matplotlib inline

# %% [markdown]
# ## 練習時間

# %% [markdown]
# 觀察有興趣的欄位的資料分佈，並嘗試找出有趣的訊息
# #### Eg
# - 計算任意欄位的平均數及標準差等等統計量，學習觀察是否有異常
# - 畫出任意欄位的[直方圖](https://zh.wikipedia.org/zh-tw/%E7%9B%B4%E6%96%B9%E5%9B%BE)
# 
# ### Hints:
# - [Descriptive Statistics For pandas Dataframe](https://chrisalbon.com/python/data_wrangling/pandas_dataframe_descriptive_stats/)
# - [pandas 中的繪圖函數](https://amaozhao.gitbooks.io/pandas-notebook/content/pandas%E4%B8%AD%E7%9A%84%E7%BB%98%E5%9B%BE%E5%87%BD%E6%95%B0.html)
# 

# %%
app_train['AMT_INCOME_TOTAL'].describe()

# %%
app_train['AMT_INCOME_TOTAL'].hist()
plt.xlabel('AMT_INCOME_TOTAL')

# %% [markdown]
# #### 注意到該欄位的最大值和 75% 百分位數的值有異常大的差距，所以直接畫直方圖會看不出所以然來，可以先過濾掉再重新畫圖來看

# %%
app_train.loc[app_train['AMT_INCOME_TOTAL']<app_train['AMT_INCOME_TOTAL'].quantile(0.99)]['AMT_INCOME_TOTAL'].hist()
plt.xlabel('AMT_INCOME_TOTAL')

# %% [markdown]
# # Section 3

# %% [markdown]
# # 檢視 Outliers
# ### 為何會有 outliers, 常見的 outlier 原因
# * 未知值，隨意填補 (約定俗成的代入)，如年齡常見 0,999
# * 可能的錯誤紀錄/手誤/系統性錯誤，如某本書在某筆訂單的銷售量 = 1000 本

# %%
# Import 需要的套件
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

%matplotlib inline

# 設定 data_path
dir_data = '/content/drive/MyDrive/中大講師/中大講師/中大 - 天氣與人工智慧/111-1/20221109 機器學習實作2 資料清理數據前處理/data'

# %%
f_app = os.path.join(dir_data, 'application_train.csv')
print('Path of read in data: %s' % (f_app))
app_train = pd.read_csv(f_app)
app_train.head()

# %% [markdown]
# ## 開始檢視不同欄位是否有異常值

# %%
app_train['DAYS_BIRTH'].head()

# %%
# DAYS_BIRTH: 客戶申請貸款時的年齡
(app_train['DAYS_BIRTH'] / (-365)).describe()

# %%
# DAYS_EMPLOYED: 申請貸款前，申請人已在現職工作的時間
print((app_train['DAYS_EMPLOYED'] / 365).describe())
plt.hist(app_train['DAYS_EMPLOYED'])
plt.show()
app_train['DAYS_EMPLOYED'].value_counts()

# %% [markdown]
# ### 從上面的圖與數值可以看出 365243 顯然是個奇怪的數值

# %%
anom = app_train[app_train['DAYS_EMPLOYED'] == 365243]
non_anom = app_train[app_train['DAYS_EMPLOYED'] != 365243]
print('The non-anomalies default on %0.2f%% of loans' % (100 * non_anom['TARGET'].mean()))
print('The anomalies default on %0.2f%% of loans' % (100 * anom['TARGET'].mean()))
print('There are %d anomalous days of employment' % len(anom))

# %%
# 新增一個欄位: DAYS_EMPLOYED_ANOM 來標記 DAYS_EMPLOYED 是否異常
app_train['DAYS_EMPLOYED_ANOM'] = app_train["DAYS_EMPLOYED"] == 365243
print(app_train['DAYS_EMPLOYED_ANOM'].value_counts())

# 這邊我們用 nan 將異常值取代
app_train['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)

app_train['DAYS_EMPLOYED'].plot.hist(title = 'Days Employment Histogram');
plt.xlabel('Days Employment');

# %%
# 檢查 OWN_CAR_AGE: 貸款人的車齡
plt.hist(app_train[~app_train.OWN_CAR_AGE.isnull()]['OWN_CAR_AGE'])
plt.show()
app_train['OWN_CAR_AGE'].value_counts()

# %%
app_train[app_train['OWN_CAR_AGE'] > 50]['OWN_CAR_AGE'].value_counts()

# %% [markdown]
# ### 從上面我們可以發現車齡為 64, 65 的人特別多，是否合理?
# * 記得，這沒有正確答案 - 但我們總是可以給他們一些標記，讓最後的模型來決定

# %%
print("Target of OWN_CAR_AGE >= 50: %.2f%%" % (app_train[app_train['OWN_CAR_AGE'] >= 50 ]['TARGET'].mean() * 100 ))
print("Target of OWN_CAR_AGE < 50: %.2f%%" % (app_train[app_train['OWN_CAR_AGE'] < 50]['TARGET'].mean() * 100))

app_train['OWN_CAR_AGE_ANOM'] = app_train['OWN_CAR_AGE'] >= 50

# %% [markdown]
# ## 請參考 HomeCredit_columns_description.csv 的欄位說明，觀察並列出三個你覺得可能有 outlier 的欄位並解釋可能的原因

# %%
app_train.dtypes.isin([np.dtype('int64')])

# %%
# 先篩選數值型的欄位
import matplotlib.pyplot as plt

"""
YOUR CODE HERE, fill correct data types (for example str, float, int, ...)
"""
dtype_select = [np.dtype("int64"), np.dtype("float64")]

numeric_columns = list(app_train.columns[list(app_train.dtypes.isin(dtype_select))])

# 再把只有 2 值 (通常是 0,1) 的欄位去掉
numeric_columns = list(app_train[numeric_columns].columns[list(app_train[numeric_columns].apply(lambda x:len(x.unique())!=2 ))])
print("Numbers of remain columns %i" % len(numeric_columns))

# 檢視這些欄位的數值範圍
for col in numeric_columns:
    """
    Your CODE HERE, make the box plot
    """
    #print(col)
    plt.boxplot(app_train[col])
    plt.show()

# %%
# 從上面的圖檢查的結果，至少這三個欄位好像有點可疑

# AMT_INCOME_TOTAL
# REGION_POPULATION_RELATIVE
# OBS_60_CNT_SOCIAL_CIRCLE

# %% [markdown]
# ### Hints: Emprical Cumulative Density Plot, [ECDF](https://zh.wikipedia.org/wiki/%E7%BB%8F%E9%AA%8C%E5%88%86%E5%B8%83%E5%87%BD%E6%95%B0), [ECDF with Python](https://stackoverflow.com/questions/14006520/ecdf-in-python-without-step-function)

# %%
# 最大值離平均與中位數很遠
print(app_train['AMT_INCOME_TOTAL'].describe())

# 繪製 Empirical Cumulative Density Plot (ECDF)
cdf = app_train.AMT_INCOME_TOTAL.value_counts().sort_index().cumsum()
plt.plot(list(cdf.index), cdf/cdf.max())
plt.xlabel('Value')
plt.ylabel('ECDF')
plt.xlim([cdf.index.min(), cdf.index.max() * 1.05]) # 限制顯示圖片的範圍
plt.ylim([-0.05,1.05]) # 限制顯示圖片的範圍

plt.show()

plt.plot(np.log(list(cdf.index)), cdf/cdf.max())
plt.xlabel('Value (log-scale)')
plt.ylabel('ECDF')

plt.ylim([-0.05,1.05]) # 限制顯示圖片的範圍

plt.show()

# %% [markdown]
# ## 補充：Normal dist 的 ECDF
# ![ecdf_normal](https://au.mathworks.com/help/examples/stats/win64/PlotEmpiricalCdfAndCompareWithSamplingDistributionExample_01.png)

# %%
# 最大值落在分布之外
print(app_train['REGION_POPULATION_RELATIVE'].describe())

# 繪製 Empirical Cumulative Density Plot (ECDF)
"""
Your Code Here
"""
cdf = 


plt.plot(list(cdf.index), cdf/cdf.max())
plt.xlabel('Value')
plt.ylabel('ECDF')
plt.ylim([-0.05,1.05]) # 限制顯示圖片的範圍
plt.show()

app_train['REGION_POPULATION_RELATIVE'].hist()
plt.show()

app_train['REGION_POPULATION_RELATIVE'].value_counts()

# 就以這個欄位來說，雖然有資料掉在分布以外，也不算異常，僅代表這間公司在稍微熱鬧的地區有的據點較少，
# 導致 region population relative 在少的部分較為密集，但在大的部分較為疏漏

# %%
# 最大值落在分布之外
print(app_train['OBS_60_CNT_SOCIAL_CIRCLE'].describe())

# 繪製 Empirical Cumulative Density Plot (ECDF)
"""
Your Code Here

cdf = 


plt.plot(list(cdf.index), cdf/cdf.max())
plt.xlabel('Value')
plt.ylabel('ECDF')
plt.xlim([cdf.index.min() * 0.95, cdf.index.max() * 1.05])
plt.ylim([-0.05,1.05]) # 限制顯示圖片的範圍
plt.show()
"""
app_train['OBS_60_CNT_SOCIAL_CIRCLE'].hist()
plt.show()
print(app_train['OBS_60_CNT_SOCIAL_CIRCLE'].value_counts().sort_index(ascending = False))

# %% [markdown]
# ## 注意：當 histogram 畫出上面這種圖 (只出現一條，但是 x 軸延伸很長導致右邊有一大片空白時，代表右邊有值但是數量稀少。這時可以考慮用 value_counts 去找到這些數值

# %%
# 把一些極端值暫時去掉，再繪製一次 Histogram
# 選擇 OBS_60_CNT_SOCIAL_CIRCLE 小於 20 的資料點繪製
"""
Your Code Here
"""
loc_a = 
loc_b = 

app_train.loc[loc_a, loc_b].hist()
plt.show()


