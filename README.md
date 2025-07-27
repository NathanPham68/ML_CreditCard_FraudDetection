# 💳 Credit Card Fraud Detection using Machine Learning

<img width="1280" height="720" alt="image" src="https://github.com/user-attachments/assets/56ba39ba-05bf-4e49-af8d-895d5d7b9863" />

This project focuses on building a basic Machine Learning model to detect fraudulent credit card transactions. We use a structured dataset containing real-world transaction details, including customer demographics, transaction metadata, and merchant information.

The key objective is to classify whether a transaction is fraudulent (1) or legitimate (0) using various features in the dataset.

## 📁 Dataset Overview

[**Link to dataset**](https://drive.google.com/drive/folders/13BTS7p_jQvnd45g4fhg1uvz-T6RW3gV1?usp=sharing)

The dataset consists of transaction records labeled as fraud or not. It includes both categorical and numerical features, which gives us a good opportunity to practice feature encoding, preprocessing, and supervised learning.

<details>
<summary> <strong>Key Features</strong></summary>

| Feature Name                     | Description                                 |
| -------------------------------- | ------------------------------------------- |
| `trans_date_trans_time`          | Timestamp of the transaction                |
| `cc_num`                         | Credit card number                          |
| `merchant`                       | Name of the merchant                        |
| `category`                       | Category/business type of the merchant      |
| `amt`                            | Transaction amount (USD)                    |
| `first`, `last`                  | First and last name of cardholder           |
| `gender`                         | Gender of the cardholder (`Male`/`Female`)  |
| `street`, `city`, `state`, `zip` | Cardholder's address                        |
| `lat`, `long`                    | Geolocation of the cardholder               |
| `city_pop`                       | Population of the city                      |
| `job`                            | Cardholder's job                            |
| `dob`                            | Date of birth of cardholder                 |
| `trans_num`                      | Unique transaction ID                       |
| `unix_time`                      | Unix timestamp (seconds since 1970)         |
| `merch_lat`, `merch_long`        | Geolocation of the merchant                 |
| `is_fraud`                       | Target label: `1` = fraud, `0` = legitimate |

</details>

## 🧠 Goal

The main goal of this project is to:

- Preprocess the dataset for modeling.

- Encode high-cardinality categorical features (like merchant, job, category) effectively.

- Train and evaluate a supervised learning model to detect fraudulent transactions.

## 📊 Tools & Technologies

- Python (Pandas, NumPy, Scikit-learn, Matplotlib/Seaborn)

- Google Colab

- Machine Learning algorithms: Logistic Regression, Random Forest, or XGBoost

## 📈 Analysis & Results

[**Link to code**](https://drive.google.com/file/d/1vdyGBJQgKGCFb16Cn9IhqvpkLlDFqf3F/view?usp=sharing)

### ✅ I. Import libraries

```ruby
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import warnings
import time
warnings.filterwarnings('ignore')
```

### ✅ II. Overview the data

<img width="1280" height="490" alt="image" src="https://github.com/user-attachments/assets/c0d3e7ea-8b59-4f4e-b9f4-075647fc8141" />

#### 1. Check info & missing values

<img width="1280" height="675" alt="image" src="https://github.com/user-attachments/assets/0d6d66f2-93da-4b9c-94cf-92dd2b5dfce3" />

&#8594; There's no missing values in this dataset

#### 2. Check duplicated values

<img width="1280" height="99" alt="image" src="https://github.com/user-attachments/assets/05d1cd49-6b65-4c3e-bec9-0297706e5251" />

#### 3. Drop 2 columns Unnamed

```ruby
df.drop(df.columns[[0,1]], axis=1, inplace =True)
```

#### 4. Check imbalanced

<img width="1280" height="275" alt="image" src="https://github.com/user-attachments/assets/f2ac9375-5291-4f0e-872b-02ab76977ac1" />

**The ration of label 1 on total is 7.6%** → We can continue with the EDA and ML model

### ✅ III. Define type of data

<img width="1280" height="665" alt="image" src="https://github.com/user-attachments/assets/addf27cc-8af9-4123-bc6e-38ac9bd24673" />

As showing the unique values of each numeric columns, there's no columns have low unique values (less than 10 values) → There's no column has dtype = numeric but have category meaning.

### ✅ IV. Start Feature Engineering & EDA

#### 1. Numeric Features

##### 1.1. Check distribution (density) of some quantity features (amt, city_pop)

```ruby
cols = ['amt','city_pop']

plt.figure(figsize=(10, 6))

for i, col in enumerate(cols, 1):
    plt.subplot(len(cols), 1, i)
    sns.kdeplot(num_cols[col], shade=True)
    plt.title(f'Density Distribution of {col}')
    plt.xlabel('Value')
    plt.ylabel('Density')

plt.tight_layout()
plt.show()
```

<img width="1280" height="656" alt="image" src="https://github.com/user-attachments/assets/391f0d3e-ded3-4799-98aa-97176f5be4c5" />

From this distribution, we can see:

* amt: most amount of transactions centralize from 0 to 1000 &#8594; it can have some outliers in this columns, it can related to fraud transactions

* city_pop: most the city population centralize from < 100000 to 200000 &#8594; in practice, population of the city may not affect the fraud transaction. However, we still check if any.

##### 1.2. Check correlation

<img width="1280" height="641" alt="image" src="https://github.com/user-attachments/assets/b2959bfe-bb88-4c5b-89ab-c3ccba8c1d2b" />

Based on correlation heatmap, we see the amt has high correlation with is_fraud

<img width="1280" height="579" alt="image" src="https://github.com/user-attachments/assets/1f22b1c1-8d50-4869-8593-d8cdb0c111e6" />

* For fraud trns, the number of total amount centralize from about 200 - 2000

* For non-fraud, the number of total amount have more higher values

#### 2. Category Features

```ruby
def count_percentage_tns(df, column, target, count):
    '''
    This function to create the table calculate the percentage of fraud/non-fraud transaction on total transaction group by category values

    '''

    # Create 2 dataframes of fraud and non-fraud
    fraud = df[df[target]==1].groupby(column)[[count]].count().reset_index().sort_values(ascending=False, by = count)
    not_fraud = df[df[target]==0].groupby(column)[[count]].count().reset_index().sort_values(ascending=False, by = count)

    #Merge 2 dataframe into one:
    cate_df = fraud.merge(not_fraud, on = column , how = 'outer')
    cate_df = cate_df.fillna(0)
    cate_df.rename(columns = {count+'_x':'fraud',count+'_y':'not_fraud'}, inplace = True)

    #Caculate the percentage:
    cate_df['%'] = cate_df['fraud']/(cate_df['fraud']+cate_df['not_fraud'])

    return cate_df

def count_percentage_cc(df, column, target, count):
    '''
    This function to create the table calculate the percentage of fraud/non-fraud card number on total card number group by category values

    '''

    # Create 2 dataframes of fraud and non-fraud
    fraud = df[df[target]==1].groupby(column)[[count]].nunique().reset_index().sort_values(ascending=False, by = count)
    not_fraud = df[df[target]==0].groupby(column)[[count]].nunique().reset_index().sort_values(ascending=False, by = count)

    #Merge 2 dataframe into one:
    cate_df = fraud.merge(not_fraud, on = column , how = 'outer')
    cate_df = cate_df.fillna(0)
    cate_df.rename(columns = {count+'_x':'fraud',count+'_y':'not_fraud'}, inplace = True)

    #Caculate the percentage:
    cate_df['%'] = cate_df['fraud']/(cate_df['fraud']+cate_df['not_fraud'])

    return cate_df
```

##### 2.1. Category Features: Features related to card holders (job, dob, city, state)

<img width="1280" height="395" alt="image" src="https://github.com/user-attachments/assets/20a37fd3-add3-44f6-bac7-6a72e2dee30d" />

<img width="1280" height="695" alt="image" src="https://github.com/user-attachments/assets/148fdd31-8fc5-4533-9922-45271721e435" />

Since the category columns have many values (job, city, state,..), we will calculate the percentage of fraud transaction for each values in that features, and plot the boxplot to see whether any outliers (outliers in this mean will show us there're some values have high fraud ratio and can be insightful)

```ruby
cols = ['job','state','city','dob_year'] #excluded columns with the same meaning

plt.figure(figsize=(10, 8))

for i, col in enumerate(cols, 1):
    cate_df = count_percentage_cc(df, col, 'is_fraud','cc_num') #apply the function to calculate the fraud ratio
    plt.subplot(len(cols), 1, i)
    sns.boxplot(data = cate_df, y='%')
    plt.title(f'Fraud ratio Distribution of {col}')
    plt.xlabel('Value')
    plt.ylabel('Fraud Ratio')

plt.tight_layout()
plt.show()
```

<img width="1280" height="665" alt="image" src="https://github.com/user-attachments/assets/5b54e8fa-9a38-4b6c-9940-43adb0a3c065" />

All of this features have some highest values as outliers → should check each of them

###### 2.1.1. JOB: not related to fraud card number

<img width="1280" height="596" alt="image" src="https://github.com/user-attachments/assets/6f380009-a90b-441b-9a08-d09f8f443588" />

As we see from 2 tables:

* Job values with % = 1 has very small number of card number fraud (1, and 2)

* Job values with high number of fraud has nearly 50/50 chance of fraud (fraud ratio = 60%)

&#8594; JOB is not related to fraud card number

###### 2.1.2. STATE: not related to fraud card number

<img width="1280" height="611" alt="image" src="https://github.com/user-attachments/assets/f5116fdc-4b78-4536-9898-324be789a408" />

The same pattern with JOB feature &#8594; STATE is not related to fraud card number

###### 2.1.3. CITY: not related to fraud card number

<img width="1280" height="603" alt="image" src="https://github.com/user-attachments/assets/985b3160-4b10-4c9d-abb1-1e50d9170963" />

The same pattern with JOB feature &#8594; STATE is not related to fraud card number

###### 2.1.4. YEAR OF BIRTH: not related to fraud card number

<img width="1280" height="617" alt="image" src="https://github.com/user-attachments/assets/b3e29b6c-0d7d-4443-b72f-695985f733b7" />

The same pattern with JOB feature &#8594; STATE is not related to fraud card number

###### 2.1.5. GENDER: not related to fraud

<img width="1280" height="199" alt="image" src="https://github.com/user-attachments/assets/e23f0bb2-6676-4561-84de-2e1dcc166625" />

The ration between Male and Female nearly the same &#8594; Gender is not related to fraud

##### 2.2. Category Features: Features related to merchant (category, merch_lat, merch_long)

###### 2.2.1. DISTANCE FROM CARD HOLDER TRANSACTION TO MERCHANT LOCATION: not related to fraud

We have the hypothesis: the distance between card holder location and merchant location can affect the fraud behavior

```ruby
# Calculate the distance:
import math

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points on the Earth.

    Parameters:
    lat1, lon1 - Latitude and longitude of the first point in decimal degreesA
    lat2, lon2 - Latitude and longitude of the second point in decimal degrees

    Returns:
    Distance between the two points in kilometers
    """
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.asin(math.sqrt(a))

    # Radius of Earth in kilometers. Use 3956 for miles. Determines return value units.
    r = 6371.0

    # Calculate the result
    distance = r * c

    return distance
```

<img width="1280" height="651" alt="image" src="https://github.com/user-attachments/assets/18a4ca90-3d2c-4f76-b40c-d6c5c8ce1d89" />

From this boxplot, we see in fraud and non-fraud transactions, they have the same range of distance

&#8594; Reject our hypothesis, distance is not affect the fraud

###### 2.2.2. CATEGORY: related to fraud transaction

```ruby
cols = ['category'] #excluded columns with the same meaning

plt.figure(figsize=(8, 6))

for i, col in enumerate(cols, 1):
    cate_df = count_percentage_tns(df, col, 'is_fraud','cc_num') #apply the function to calculate the fraud ratio
    plt.subplot(len(cols), 1, i)
    sns.boxplot(data = cate_df, y='%')
    plt.title(f'Fraud ratio Distribution of {col}')
    plt.xlabel('Value')
    plt.ylabel('Fraud Ratio')

plt.tight_layout()
plt.show()
```

<img width="1280" height="644" alt="image" src="https://github.com/user-attachments/assets/3e3f29bc-3e9a-4ab3-8b17-7a996218e94b" />

<img width="1280" height="624" alt="image" src="https://github.com/user-attachments/assets/1781b1c0-2ce9-4885-bb35-f1628fc8d4ab" />

From 2 tables, we can see:

* Nearly 20% transactions of shopping_net, misc_ner and grocery_pos are fraud

* 20% transactions in each of 3 values come from nearly 50% card holders

&#8594; Category affect the fraud transaction: shopping_net, misc_ner and grocery_pos

##### 2.3. Category Features: Features related to transaction (trans_time)

TRANSACTION HOURS: related to fraud

We have the hypothesis: the time (hour) of transaction can related to fraud

<img width="1280" height="489" alt="image" src="https://github.com/user-attachments/assets/0e8d5747-cfa8-4f44-8161-fb0d47bb2b31" />

From this chart, we see that: From 22 to 4 o'clock: the ratio of fraud transactions is higher than other

&#8594; Transaction hour related to fraud

### ✅ V. Conclusion

We have some insights below for fraud behaviors:

1. Fraud transactions most have the higher amount of values: (from 200 to 2000)

2. Transactions spend on these categories can have higher chance of fraud: shopping_net, misc_net, grocery_pos

3. Transactions spend on time between 22 to 4 o'clock can have higher chance of fraud

### ✅ VI. Apply model

#### 1. Feature Transforming

<img width="1280" height="238" alt="image" src="https://github.com/user-attachments/assets/f4b7d1dd-64a3-4451-a18f-2778f022d45b" />

<img width="1280" height="552" alt="image" src="https://github.com/user-attachments/assets/7ebd0849-1462-4c47-a79f-143459043e4f" />

#### 2. Model training & Evaluation

```ruby
from sklearn.model_selection import train_test_split
x=scaled_df.drop('is_fraud', axis = 1)
y=scaled_df[['is_fraud']]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

from sklearn.ensemble import RandomForestClassifier

clf_rand = RandomForestClassifier(max_depth=2, random_state=0)

clf_rand.fit(x_train, y_train)
y_ranf_pre_train = clf_rand.predict(x_train)
y_ranf_pre_test = clf_rand.predict(x_test)

from sklearn.metrics import balanced_accuracy_score

print(f'Balance accuracy of train set: {balanced_accuracy_score(y_train, y_ranf_pre_train)}')
print(f'Balance accuracy of test set: {balanced_accuracy_score(y_test, y_ranf_pre_test)}')
```

<img width="1280" height="85" alt="image" src="https://github.com/user-attachments/assets/bc06c995-b31b-423e-b511-e4b8802e887b" />


