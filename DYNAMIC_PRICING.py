################################################################
# PROJECT: DYNAMIC PRICING
################################################################

import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from helpers.helpers import *
from scipy.stats import shapiro, levene, ttest_ind, mannwhitneyu

from scipy.stats import shapiro
import scipy.stats as stats
import statsmodels.stats.api as sms
import itertools

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 170)

df = pd.read_csv("dynamic_pricing/pricing.csv", sep=";")
df.head()

################################################################
# EXPLORATORY DATA ANALYSIS:
################################################################

check_df(df)
# ##################### Shape #####################
# (3448, 2)
# ##################### Types #####################
# category_id      int64
# price          float64
# dtype: object
# ##################### Head #####################
#    category_id  price
# 0       489756 32.118
# 1       361254 30.711
# 2       361254 31.573
# 3       489756 34.544
# 4       489756 47.206
# ##################### Tail #####################
#       category_id  price
# 3443       489756 37.617
# 3444       874521 33.674
# 3445       489756 58.399
# 3446       874521 65.723
# 3447       489756 30.000
# ##################### NA #####################
# category_id    0
# price          0
# dtype: int64
# ##################### Quantiles #####################
#                  0.000      0.050      0.500      0.950      0.990      1.000
# category_id 201436.000 326584.000 489756.000 874521.000 874521.000 874521.000
# price           10.000     30.000     34.799     92.978 201436.464 201436.991

cat_cols, num_cols, cat_but_car = grab_col_names(df)
# Observations: 3448
# Variables: 2
# cat_cols: 1
# num_cols: 1
# cat_but_car: 0
# num_but_cat: 1

for col in num_cols:
    print({f"{col}"})
    num_summary(df, col)
# {'price'}
# count     3448.000
# mean      3254.476
# std      25235.799
# min         10.000
# 5%          30.000
# 10%         30.000
# 25%         31.890
# 50%         34.799
# 75%         41.536
# 80%         45.461
# 90%         62.506
# 95%         92.978
# 99%     201436.464
# max     201436.991
# Name: price, dtype: float64

df.describe().T
#                count       mean        std        min        25%        50%        75%        max
# category_id 3448.000 542415.172 192805.690 201436.000 457630.500 489756.000 675201.000 874521.000
# price       3448.000   3254.476  25235.799     10.000     31.890     34.799     41.536 201436.991

df.groupby("category_id").agg({"price": ["mean", "median"]})
#                price
#                 mean median
# category_id
# 201436        36.175 33.535
# 326584      1424.665 31.748
# 361254      1659.681 34.459
# 489756      3589.809 35.636
# 675201      3112.240 33.836
# 874521      4605.357 34.401

def graphic(dataframe, col):
    sns.distplot(dataframe[col])
    plt.show()

graphic(df, "price")

################################################################
# TASK 1:  Does the price of the product differ according to its categories?
################################################################

test_istatistigi, pvalue = shapiro(df.loc[df["category_id"] == 201436, "price"])
print('Test İstatistiği = %.4f, p-değeri = %.4f' % (test_istatistigi, pvalue))  # H0: RED
# Test İstatistiği = 0.6190, p-değeri = 0.0000

test_istatistigi, pvalue = shapiro(df.loc[df["category_id"] == 326584, "price"])
print('Test İstatistiği = %.4f, p-değeri = %.4f' % (test_istatistigi, pvalue)) # H0: RED
# Test İstatistiği = 0.0568, p-değeri = 0.0000

# Normallik varsayımı sağlanmamaktadır. H0: Red. :)))
for group in list(df["category_id"].unique()):
    pvalue = shapiro(df.loc[df["category_id"] == group, "price"])[1]
    print(group, 'p-value: %.4f' % pvalue)
# 489756 p-value: 0.0000
# 361254 p-value: 0.0000
# 874521 p-value: 0.0000
# 326584 p-value: 0.0000
# 675201 p-value: 0.0000
# 201436 p-value: 0.0000

# NON-PARAMETRIC TEST:
lst = list(df["category_id"].unique())

for x,y in itertools.combinations(lst, 2):
    print(f" {x, y} ".center(50, "*"))
    test_stat, pvalue = mannwhitneyu(df.loc[df["category_id"] == x, "price"],
                 df.loc[df["category_id"] == y, "price"])
    print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
# **************** (489756, 361254) ****************
# Test Stat = 380129.5000, p-value = 0.0000
# **************** (489756, 874521) ****************
# Test Stat = 519465.0000, p-value = 0.0000
# **************** (489756, 326584) ****************
# Test Stat = 70008.0000, p-value = 0.0000
# **************** (489756, 675201) ****************
# Test Stat = 86744.5000, p-value = 0.0000
# **************** (489756, 201436) ****************
# Test Stat = 60158.0000, p-value = 0.0000
# **************** (361254, 874521) ****************
# Test Stat = 218127.0000, p-value = 0.0242
# **************** (361254, 326584) ****************
# Test Stat = 33158.0000, p-value = 0.0000
# **************** (361254, 675201) ****************
# Test Stat = 39587.0000, p-value = 0.3251
# **************** (361254, 201436) ****************
# Test Stat = 30006.0000, p-value = 0.4866
# **************** (874521, 326584) ****************
# Test Stat = 38752.0000, p-value = 0.0000
# **************** (874521, 675201) ****************
# Test Stat = 47530.0000, p-value = 0.2762
# **************** (874521, 201436) ****************
# Test Stat = 34006.0000, p-value = 0.1478
# **************** (326584, 675201) ****************
# Test Stat = 6963.5000, p-value = 0.0001
# **************** (326584, 201436) ****************
# Test Stat = 5301.0000, p-value = 0.0005
# **************** (675201, 201436) ****************
# Test Stat = 6121.0000, p-value = 0.3185


################################################################
# TASK 2: According to the first question, what should the price of the product be?
################################################################

# Categories 489756 and 326584 are different from the others. No difference in others.
# We can apply the same price to all but these two ids.
# It would be better if we charge different prices to the others as there is no statistical difference between the other IDs.
# Since I will give a price range since I set the prices according to the 95% confidence interval, I prefer not to suppress the outliers since lowering the mean of the confidence intervals will not make a difference.

################################################################
# TASK 3: In terms of price, it is desirable to be "moving". Build a decision support system for price strategy.
################################################################
lst = list(df["category_id"].unique())

for x in lst:
    print(f" {x} ".center(50, "*"))
    print(sms.DescrStatsW(df.loc[df["category_id"] == x, "price"]).tconfint_mean())
# ********************* 489756 *********************
# (2331.739928545868, 4847.8771233317875)
# ********************* 361254 *********************
# (237.85682624118522, 3081.5044995737935)
# ********************* 874521 *********************
# (2455.170723966923, 6755.543792149052)
# ********************* 326584 *********************
# (-1320.7429694257642, 4170.073333101381)
# ********************* 675201 *********************
# (-1172.5979883294285, 7397.078713300637)
# ********************* 201436 *********************
# (34.381720084633564, 37.96927659690045)

# Functional Status:
def ab_test(dataframe, category, target):
    AB = pd.DataFrame()

    lst = df["category_id"].unique()
    for x, y in list(itertools.combinations(lst, 2)):
        a = dataframe[dataframe[category] == [x][0]][target]
        b = dataframe[dataframe[category] == [y][0]][target]

        pvalue1 = shapiro(a)[1]
        pvalue2 = shapiro(b)[1]

        print(pvalue1)
        print(pvalue2)
        # If the assumption of normality is met:
        if (pvalue1 or pvalue2) > 0.05:
            print(pvalue1, pvalue2)

            # If the assumptions are homogeneous:
            pvalue3 = levene(a, b)[1]

            if pvalue3 > 0.05:
                c = ttest_ind(a, b, equal_var=True)[1]
                if c > 0.05:
                    print('ttest result p-value = %.4f' % (c))
                    print(F"Failed to reject null hypothesis (H0). The difference between the two groups is statistically insignificant.")
                else:
                    print('ttest result p-value = %.4f' % (c))
                    print(F"Reject null hypothesis (H0). The difference between the two groups is statistically significant.")

            # If the assumptions are not homogeneous:
            else:
                c = ttest_ind(a, b, equal_var=False)[1]
                if c > 0.05:
                    print('ttest result p-value = %.4f' % (c))
                    print(F"Failed to reject null hypothesis (H0). The difference between the two groups is statistically insignificant.")
                else:
                    print('ttest result p-value = %.4f' % (c))
                    print(F"Reject null hypothesis (H0). The difference between the two groups is statistically significant.")

        # If the assumption of normality is not met:
        else:
            c = mannwhitneyu(a, b)[1]
            if c > 0.05:
                print('mannwhitneyu result p-value = %.4f' % (c))
                print(F"Failed to reject null hypothesis (H0). The difference between the two groups is statistically insignificant.")
            else:
                print('mannwhitneyu result p-value = %.4f' % (c))
                print(F"Reject null hypothesis (H0). The difference between the two groups is statistically significant.")

        temp = pd.DataFrame({"Group Comparison": [c < 0.05],
                             "p-value": c,
                             "GroupA Mean": [a.mean()], "GroupB Mean": [b.mean()],
                             "GroupA Median": [a.median()], "GroupB Median": [b.median()],
                             "GroupA Count": [a.count()], "GroupB Count": [b.count()]})

        temp["Group Comparison"] = np.where(temp["Group Comparison"] == True, "There is a difference", "No difference")
        temp["Test Type"] = np.where((pvalue1 == 0.05) & (pvalue2 == 0.05), "Parametric", "Non-Parametric")

        AB = pd.concat([AB, temp[
            ["Test Type", "Group Comparison", "p-value", "GroupA Mean", "GroupB Mean", "GroupA Median",
             "GroupB Median", "GroupA Count", "GroupB Count"]]])
    return AB

ab_test(df, "category_id","price")

# OUTPUT:
#      Test Tipi       Group Comparison  p-value  GroupA Mean  GroupB Mean  GroupA Median  GroupB Median  GroupA Count  GroupB Count
#   Non-Parametrik  There is a difference    0.000     3589.809     1659.681         35.636         34.459          1705           620
#   Non-Parametrik  There is a difference    0.000     3589.809     4605.357         35.636         34.401          1705           750
#   Non-Parametrik  There is a difference    0.000     3589.809     1424.665         35.636         31.748          1705           145
#   Non-Parametrik  There is a difference    0.000     3589.809     3112.240         35.636         33.836          1705           131
#   Non-Parametrik  There is a difference    0.000     3589.809       36.175         35.636         33.535          1705            97
#   Non-Parametrik  There is a difference    0.024     1659.681     4605.357         34.459         34.401           620           750
#   Non-Parametrik  There is a difference    0.000     1659.681     1424.665         34.459         31.748           620           145
#   Non-Parametrik          No difference    0.325     1659.681     3112.240         34.459         33.836           620           131
#   Non-Parametrik          No difference    0.487     1659.681       36.175         34.459         33.535           620            97
#   Non-Parametrik  There is a difference    0.000     4605.357     1424.665         34.401         31.748           750           145
#   Non-Parametrik          No difference    0.276     4605.357     3112.240         34.401         33.836           750           131
#   Non-Parametrik          No difference    0.148     4605.357       36.175         34.401         33.535           750            97
#   Non-Parametrik  There is a difference    0.000     1424.665     3112.240         31.748         33.836           145           131
#   Non-Parametrik  There is a difference    0.001     1424.665       36.175         31.748         33.535           145            97
#   Non-Parametrik          No difference    0.319     3112.240       36.175         33.836         33.535           131            97