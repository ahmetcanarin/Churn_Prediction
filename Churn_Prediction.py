import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, RocCurveDisplay
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
import warnings
warnings.simplefilter(action="ignore")

pd.set_option("display.max_columns", None)
# pd.set_option("display.expand_frame_repr", False)
pd.set_option("display.width", 500)
pd.set_option("display.max_rows", 20)
pd.set_option("display.float_format", lambda x: "%.2f" % x)

# Load the Telco churn dataset.
df = pd.read_csv("Telco-Customer-Churn.csv")

# =========================
# 1. EXPLORATORY DATA ANALYSIS
# =========================

def check_df(dataframe, head=5):
    """
    Print a general overview of the dataset:
    shape, data types, sample rows, missing values, and quantiles.
    """
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.describe([0.05, 0.50, 0.95, 0.99]).T)

check_df(df)


def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    Return categorical, numerical, and cardinal categorical column names.

    cat_cols: categorical columns, including numeric-looking categorical features
    num_cols: numerical columns
    cat_but_car: categorical columns with high cardinality
    """

    # Identify object-based categorical columns.
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]

    # Identify numeric columns that should be treated as categorical.
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]

    # Identify high-cardinality categorical columns.
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # Pure numerical columns.
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)


# TotalCharges is stored as object because of blank strings.
# Convert blanks to NaN first, then cast to float.
df["TotalCharges"].value_counts()
df["TotalCharges"] = df["TotalCharges"].replace(" ", np.nan)
df["TotalCharges"].isnull().sum()
df["TotalCharges"].info()
df["TotalCharges"] = df["TotalCharges"].astype(float)

# Standardize service-related "No internet service" and "No phone service" labels.
# This reduces unnecessary category fragmentation.
no_internet = [col for col in df.columns if df[col].astype(str).str.contains("no internet service", case=False).any()]
no_phone = [col for col in df.columns if df[col].astype(str).str.contains("no phone service", case=False).any()]

df[no_internet] = df[no_internet].replace("No internet service", "No")
df[no_phone] = df[no_phone].replace("No phone service", "No")

# Convert Yes/No style columns to binary where applicable.
# Original logic is preserved here.
for col in df.columns:
    if (df[df[col] == "Yes"][col]).any():
        df[col] = df[col].apply(lambda x: 1 if x == "Yes" else 0)

cat_cols, num_cols, cat_but_car = grab_col_names(df)
df.info()


def cat_summary(dataframe, col_name, plot=False):
    """
    Display class counts, ratios, and data type for a categorical feature.
    Optionally plot the distribution.
    """
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe),
                        "Dtype": dataframe[col_name].dtypes}))
    print("##########################################")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

for col in cat_cols:
    cat_summary(df, col, False)


def num_summary(dataframe, numerical_col, plot=False):
    """
    Display descriptive statistics for a numerical feature.
    Optionally plot its histogram.
    """
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

for col in num_cols:
    num_summary(df, col, False)


def target_summary_with_cat(dataframe, target, categorical_col):
    """
    Compare average target rate across category levels.
    This helps identify churn-sensitive segments.
    """
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}))
    print("##########################\n")

for col in cat_cols:
    if col not in "Churn":
        print("####", col, "####")
        target_summary_with_cat(df, "Churn", col)


def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    """
    Compute IQR-based lower and upper limits for outlier detection.
    """
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    """
    Return whether a numerical feature contains outliers.
    """
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

for col in num_cols:
    print(col, ":", check_outlier(df, col))


def missing_values_table(dataframe, na_name=False):
    """
    Summarize columns with missing values and optionally return their names.
    """
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns

missing_values_table(df)

na_cols = missing_values_table(df, True)

def missing_vs_target(dataframe, target, na_columns):
    """
    Check whether missingness itself is associated with the target.
    """
    temp_df = dataframe.copy()

    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)

    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns

    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "TARGET_COUNT": temp_df.groupby(col)[target].count()}), end="\n\n\n")

missing_vs_target(df, "Churn", na_cols)

[col for col in num_cols if (df[col] == 0).any()]
df[df.isnull().any(axis=1)]
pd.crosstab(df["TotalCharges"], df["tenure"], dropna=False)
df.isnull().sum().sort_values(ascending=False)


# =========================
# 2. FEATURE ENGINEERING
# =========================

# TotalCharges is missing mostly for very new customers.
# Filling with 0 preserves the business meaning of "no accumulated charges yet".
df.loc[df["TotalCharges"].isnull(), "TotalCharges"] = 0
df.isnull().sum().sort_values(ascending=False)

# Average monthly spending proxy adjusted by tenure.
df["AverageCharges"] = df["TotalCharges"] / (df["tenure"] + 1)
df["AverageCharges"].value_counts()

# Internet access indicator.
df["HasInternet"] = df["InternetService"].apply(lambda x: 0 if x == "No" else 1)
df["HasInternet"].value_counts()

# Fiber users are often behaviorally different from DSL / non-internet customers.
df["HasFiber"] = df["InternetService"].apply(lambda x: 1 if x == "Fiber optic" else 0)
df["HasFiber"].value_counts()

# Month-to-month contracts are usually a strong churn signal in telecom data.
df["IsMonthToMonth"] = df["Contract"].apply(lambda x: 1 if x == "Month-to-month" else 0)
df["IsMonthToMonth"].value_counts()

# Count how many optional services the customer has activated.
service_cols = ["PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
df["TotalServices"] = df[service_cols].sum(axis=1)
df["TotalServices"].value_counts()

df["tenure"].sort_values(ascending=False)

# Bucket tenure into coarse customer lifecycle groups.
df["tenure_year"] = pd.cut(df["tenure"], bins=[0, 12, 24, 48, 72], labels=["0-1 year", "1-2 year", "2-4 year", "4-6 year"], include_lowest=True)
df["tenure_year"].value_counts()

# Flag very recent customers.
df["IsNewCustomer"] = df["tenure"].apply(lambda x: 1 if x <= 1 else 0)
df["IsNewCustomer"].value_counts()

# Auto-payment often correlates with lower friction and lower churn.
df["IsAutoPayment"] = df["PaymentMethod"].apply(lambda x: 1 if "auto" in x else 0)
df["IsAutoPayment"].value_counts()

# Electronic check users often behave differently and may carry higher risk.
df["IsElectronicCheck"] = df["PaymentMethod"].apply(lambda x: 1 if "Electronic" in x else 0)
df["IsElectronicCheck"].value_counts()

# Ratio between monthly and cumulative charges can help separate new vs established customers.
df["MonthlyTotalRatio"] = df["MonthlyCharges"] / (df["TotalCharges"] + 1)
df["MonthlyTotalRatio"].value_counts()

cat_cols, num_cols, cat_but_car = grab_col_names(df)

def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}))
    print("##########################\n")

for col in cat_cols:
    if col not in "Churn":
        print("####", col, "####")
        target_summary_with_cat(df, "Churn", col)


# =========================
# 3. ENCODING
# =========================

cat_cols, num_cols, cat_but_car = grab_col_names(df)
cat_cols = [col for col in cat_cols if col not in "Churn"]

# Identify binary non-numeric features for label encoding.
binary_cols = [col for col in df.columns if df[col].dtypes not in ["int64", "float64"] and df[col].nunique() == 2]

def label_encoder(dataframe, binary_col):
    """
    Apply label encoding to binary categorical variables.
    """
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

for col in binary_cols:
    df = label_encoder(df, col)

df[binary_cols].value_counts()

cat_cols = [col for col in cat_cols if col not in binary_cols]

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    """
    Apply one-hot encoding to remaining multi-class categorical features.
    """
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df = one_hot_encoder(df, cat_cols, drop_first=True)

df.head()
df.info()

# Convert boolean dummies to integers for modeling consistency.
for col in df.columns:
    if df[col].dtypes == "bool":
        df[col] = df[col].astype(int)
        print(df.info())


# =========================
# 4. SCALING
# =========================

# Standardize numerical features for scale-sensitive algorithms such as Logistic Regression, KNN, and SVM.
num_cols
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

df[num_cols]
df.shape


# =========================
# 5. MODELING
# =========================

y = df["Churn"]
X = df.drop(["Churn", "customerID"], axis=1)

# Benchmark multiple classification algorithms using cross-validation
# and evaluate them on multiple business-relevant metrics.
models = [('LR', LogisticRegression(random_state=12)),
          ('KNN', KNeighborsClassifier()),
          ('CART', DecisionTreeClassifier(random_state=12)),
          ('RF', RandomForestClassifier(random_state=12)),
          ('SVM', SVC(gamma='auto', random_state=12)),
          ('XGB', XGBClassifier(random_state=12)),
          ("LightGBM", LGBMClassifier(random_state=12)),
          ("CatBoost", CatBoostClassifier(verbose=False, random_state=12))]

for name, model in models:
    cv_results = cross_validate(model, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc", "precision", "recall"])
    print(f"########## {name} ##########")
    print(f"Accuracy: {round(cv_results['test_accuracy'].mean(), 4)}")
    print(f"Auc: {round(cv_results['test_roc_auc'].mean(), 4)}")
    print(f"Recall: {round(cv_results['test_recall'].mean(), 4)}")
    print(f"Precision: {round(cv_results['test_precision'].mean(), 4)}")
    print(f"F1: {round(cv_results['test_f1'].mean(), 4)}")


# =========================
# 6. HYPERPARAMETER OPTIMIZATION
# =========================

# CatBoost is selected for tuning as a strong gradient boosting baseline.
catboost_model = CatBoostClassifier(random_state=17, verbose=False)

catboost_params = {"iterations": [200, 500],
                   "learning_rate": [0.01, 0.1],
                   "depth": [3, 6]}

catboost_best_grid = GridSearchCV(catboost_model, catboost_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

catboost_final = catboost_model.set_params(**catboost_best_grid.best_params_, random_state=17).fit(X, y)

# Re-evaluate the tuned model with cross-validation.
cv_results = cross_validate(catboost_final, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

# Score a sample customer record as a quick inference check.
sample = X.sample(1, random_state=17)
catboost_final.predict(sample)
