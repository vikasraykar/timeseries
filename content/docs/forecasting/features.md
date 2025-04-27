---
bookShowToc: true
weight: 3
title: "Interpretable features"
---

# Encoding time series as interpretable features

A common machine learning approach to time series forecasting is to reduce it  to a standard **supervised regression** problem. A regression task takes as input a {{< katex >}}d{{< /katex >}}-dimensional feature vector {{< katex >}}\mathbf{x}\in\mathbb{R}^d{{< /katex >}} and predicts a scalar {{< katex >}}y \in \mathbb{R}{{< /katex >}}. The regressor {{< katex >}}y = f(\mathbf{x}){{< /katex >}} is learnt based on a labelled training dataset {{< katex >}}\left(\mathbf{x}_i,y_i\right){{< /katex >}}, for {{< katex >}}i=1,..,n{{< /katex >}} samples. However there is do direct concept of input features ({{< katex >}}\mathbf{x}{{< /katex >}}) and output target ({{< katex >}}y{{< /katex >}}) for a time series.  Instead, we must choose the time series values to be forecasted as the variable to be predicted and use various feature engineering to construct the features that will be used to make predictions for future time steps. For each time point {{< katex >}}t{{< /katex >}} we generate a feature vector {{< katex >}}\mathbf{x}(t) \in\mathbb{R}^d{{< /katex >}} based on which we need to predict the observed time series value {{< katex >}}y(t) \in\mathbb{R}{{< /katex >}}. Here we describe some of the commonly used methods to transform a time series to feature matrix.

{{< hint warning >}}
The feature vector {{< katex >}}\mathbf{x}(t){{< /katex >}} needs to be constructed only based on the time step {{< katex >}}t{{< /katex >}} and the historical values of the time series {{< katex >}}y(1),...,y(t-1){{< /katex >}} and should not use the current time series value {{< katex >}}y(t){{< /katex >}}.
{{< /hint >}}

## Lag features
{{< button href="https://github.ibm.com/srom/aix360ts/tree/master/aix360ts/transformers/LagFeatures.py" >}}code{{< /button >}}
{{< button href="https://github.ibm.com/srom/aix360ts/tree/master/examples/transformers/LagFeatures.py" >}}example{{< /button >}}

The value of the time series at previous time steps. Lag features are the classical way that time series forecasting problems are transformed into supervised learning problems.

{{< tabs "Lag features" >}}

{{% tab "aix360ts" %}}
```python
from aix360ts.transformers import LagFeatures
transformer = LagFeatures(lags=3)
```
{{% /tab %}}

{{% tab "input" %}}
```
             sales
date
2017-01-01    21.0
2017-01-02    18.0
2017-01-03     9.0
2017-01-04    18.0
2017-01-05    15.0
...            ...
2019-12-27   923.0
2019-12-28  1194.0
2019-12-29  1341.0
2019-12-30   920.0
2019-12-31   748.0

[1095 rows x 1 columns]
```
{{% /tab %}}

{{% tab "ouput" %}}
```
            sales(t-3)  sales(t-2)  sales(t-1)
date
2017-01-01         NaN         NaN         NaN
2017-01-02         NaN         NaN        21.0
2017-01-03         NaN        21.0        18.0
2017-01-04        21.0        18.0         9.0
2017-01-05        18.0         9.0        18.0
...                ...         ...         ...
2019-12-27       796.0      1178.0       852.0
2019-12-28      1178.0       852.0       923.0
2019-12-29       852.0       923.0      1194.0
2019-12-30       923.0      1194.0      1341.0
2019-12-31      1194.0      1341.0       920.0

[1095 rows x 3 columns]
```
{{% /tab %}}

{{% tab features %}}
|            | description                                                           | type       |
|:-----------|:----------------------------------------------------------------------|:-----------|
| sales(t-3) | The value of the time series (sales) at the (t-3) previous time step. | continuous |
| sales(t-2) | The value of the time series (sales) at the (t-2) previous time step. | continuous |
| sales(t-1) | The value of the time series (sales) at the (t-1) previous time step. | continuous |
{{% /tab %}}

{{< /tabs >}}

## Seasonal lag features
{{< button href="https://github.ibm.com/srom/aix360ts/tree/master/aix360ts/transformers/SeasonalLagFeatures.py" >}}code{{< /button >}}
{{< button href="https://github.ibm.com/srom/aix360ts/tree/master/examples/transformers/SeasonalLagFeatures.py" >}}example{{< /button >}}

The value of the time series at time steps for the previous seasons. For example, with monthly data, the feature for February is equal to the last observed February value.

{{< tabs "Seasonal Lag features" >}}
{{% tab "aix360ts" %}}
```python
from aix360ts.transformers import SeasonalLagFeatures
transformer = SeasonalLagFeatures(lags=2, m=365)
```
{{% /tab %}}
{{% tab "input" %}}
```
             sales
date
2017-01-01    21.0
2017-01-02    18.0
2017-01-03     9.0
2017-01-04    18.0
2017-01-05    15.0
...            ...
2019-12-27   923.0
2019-12-28  1194.0
2019-12-29  1341.0
2019-12-30   920.0
2019-12-31   748.0

[1095 rows x 1 columns]
```
{{% /tab %}}
{{% tab "ouput" %}}
```
            sales(t-2*365)  sales(t-1*365)
date
2017-01-01             NaN             NaN
2017-01-02             NaN             NaN
2017-01-03             NaN             NaN
2017-01-04             NaN             NaN
2017-01-05             NaN             NaN
...                    ...             ...
2019-12-27           428.0           463.0
2019-12-28           440.0           607.0
2019-12-29           700.0           778.0
2019-12-30           894.0          1038.0
2019-12-31           828.0           531.0

[1095 rows x 2 columns]
```
{{% /tab %}}

{{% tab features %}}
|                | description                                                               | type       |
|:---------------|:--------------------------------------------------------------------------|:-----------|
| sales(t-2*365) | The value of the time series (sales) at the (t-2*365) previous time step. | continuous |
| sales(t-1*365) | The value of the time series (sales) at the (t-1*365) previous time step. | continuous |
{{% /tab %}}

{{< /tabs >}}

## Rolling window features

{{< button href="https://github.ibm.com/srom/aix360ts/tree/master/aix360ts/transformers/RollingWindowFeatures.py" >}}code{{< /button >}}
{{< button href="https://github.ibm.com/srom/aix360ts/tree/master/examples/transformers/RollingWindowFeatures.py" >}}example{{< /button >}}

Rolling window statistics (mean,max,min).

{{< tabs "Rolling window features" >}}
{{% tab "aix360ts" %}}
```python
from aix360ts.transformers import RollingWindowFeatures
transformer = RollingWindowFeatures(window=3)
```
{{% /tab %}}
{{% tab "input" %}}
```
             sales
date
2017-01-01    21.0
2017-01-02    18.0
2017-01-03     9.0
2017-01-04    18.0
2017-01-05    15.0
...            ...
2019-12-27   923.0
2019-12-28  1194.0
2019-12-29  1341.0
2019-12-30   920.0
2019-12-31   748.0

[1095 rows x 1 columns]
```
{{% /tab %}}
{{% tab "ouput" %}}
```
            sales_min(t-1,t-3)  sales_mean(t-1,t-3)  sales_max(t-1,t-3)
date
2017-01-01                 NaN                  NaN                 NaN
2017-01-02                 NaN                  NaN                 NaN
2017-01-03                 NaN                  NaN                 NaN
2017-01-04                 9.0            16.000000                21.0
2017-01-05                 9.0            15.000000                18.0
...                        ...                  ...                 ...
2019-12-27               796.0           942.000000              1178.0
2019-12-28               852.0           984.333333              1178.0
2019-12-29               852.0           989.666667              1194.0
2019-12-30               923.0          1152.666667              1341.0
2019-12-31               920.0          1151.666667              1341.0

[1095 rows x 3 columns]
```
{{% /tab %}}
{{% tab features %}}
|                     | description                                             | type       |
|:--------------------|:--------------------------------------------------------|:-----------|
| sales_min(t-1,t-3)  | The min of the past 3 values in the sales time series.  | continuous |
| sales_mean(t-1,t-3) | The mean of the past 3 values in the sales time series. | continuous |
| sales_max(t-1,t-3)  | The max of the past 3 values in the sales time series.  | continuous |
{{% /tab %}}
{{< /tabs >}}

## Expanding window features

{{< button href="https://github.ibm.com/srom/aix360ts/tree/master/aix360ts/transformers/ExpandingWindowFeatures.py" >}}code{{< /button >}}
{{< button href="https://github.ibm.com/srom/aix360ts/tree/master/examples/transformers/ExpandingWindowFeatures.py" >}}example{{< /button >}}

Expanding window statistics (mean,max,min).

{{< tabs "Expanding window features" >}}
{{% tab "aix360ts" %}}
```python
from aix360ts.transformers import ExpandingWindowFeatures
transformer = ExpandingWindowFeatures()
```
{{% /tab %}}
{{% tab "input" %}}
```
             sales
date
2017-01-01    21.0
2017-01-02    18.0
2017-01-03     9.0
2017-01-04    18.0
2017-01-05    15.0
...            ...
2019-12-27   923.0
2019-12-28  1194.0
2019-12-29  1341.0
2019-12-30   920.0
2019-12-31   748.0

[1095 rows x 1 columns]
```
{{% /tab %}}
{{% tab "ouput" %}}
```
            sales_min(0,t-1)  sales_mean(0,t-1)  sales_max(0,t-1)
date
2017-01-01              21.0          21.000000              21.0
2017-01-02              18.0          19.500000              21.0
2017-01-03               9.0          16.000000              21.0
2017-01-04               9.0          16.500000              21.0
2017-01-05               9.0          16.200000              21.0
...                      ...                ...               ...
2019-12-27               9.0         364.901008            1907.0
2019-12-28               9.0         365.660256            1907.0
2019-12-29               9.0         366.552608            1907.0
2019-12-30               9.0         367.058501            1907.0
2019-12-31               9.0         367.406393            1907.0

[1095 rows x 3 columns]
```
{{% /tab %}}
{{% tab features %}}
|                   | description                                                 | type       |
|:------------------|:------------------------------------------------------------|:-----------|
| sales_min(0,t-1)  | The min of all the values so far in the sales time series.  | continuous |
| sales_mean(0,t-1) | The mean of all the values so far in the sales time series. | continuous |
| sales_max(0,t-1)  | The max of all the values so far in the sales time series.  | continuous |
{{% /tab %}}
{{< /tabs >}}

## Date features

{{< button href="https://github.ibm.com/srom/aix360ts/tree/master/aix360ts/transformers/DateFeatures.py" >}}code{{< /button >}}
{{< button href="https://github.ibm.com/srom/aix360ts/tree/master/examples/transformers/DateFeatures.py" >}}example{{< /button >}}

Date related features.

{{< tabs "Date features" >}}
{{% tab "aix360ts" %}}
```python
from aix360ts.transformers import DateFeatures
transformer = DateFeatures(encode_cyclical_features=False)
```
{{% /tab %}}
{{% tab "input" %}}
```
             sales
date
2017-01-01    21.0
2017-01-02    18.0
2017-01-03     9.0
2017-01-04    18.0
2017-01-05    15.0
...            ...
2019-12-27   923.0
2019-12-28  1194.0
2019-12-29  1341.0
2019-12-30   920.0
2019-12-31   748.0

[1095 rows x 1 columns]
```
{{% /tab %}}
{{% tab "ouput" %}}
```
            year     month  day_of_year  day_of_month  week_of_year  week_of_month  ... is_month_end is_quarter_start  is_quarter_end is_year_start is_year_end is_leap_year
date                                                                                ...
2017-01-01  2017   January            1             1            52              1  ...           no              yes              no           yes          no           no
2017-01-02  2017   January            2             2             1              1  ...           no               no              no            no          no           no
2017-01-03  2017   January            3             3             1              1  ...           no               no              no            no          no           no
2017-01-04  2017   January            4             4             1              1  ...           no               no              no            no          no           no
2017-01-05  2017   January            5             5             1              1  ...           no               no              no            no          no           no
...          ...       ...          ...           ...           ...            ...  ...          ...              ...             ...           ...         ...          ...
2019-12-27  2019  December          361            27            52              4  ...           no               no              no            no          no           no
2019-12-28  2019  December          362            28            52              4  ...           no               no              no            no          no           no
2019-12-29  2019  December          363            29            52              5  ...           no               no              no            no          no           no
2019-12-30  2019  December          364            30             1              5  ...           no               no              no            no          no           no
2019-12-31  2019  December          365            31             1              5  ...          yes               no             yes            no         yes           no

[1095 rows x 18 columns]
```
{{% /tab %}}
{{% tab features %}}
|                  | description                                                                           | type        |
|:-----------------|:--------------------------------------------------------------------------------------|:------------|
| year             | The year.                                                                             | ordinal     |
| month            | The month name of the year from January to December.                                  | cyclical    |
| day_of_year      | The ordinal day of the year from 1 to 365.                                            | cyclical    |
| day_of_month     | The ordinal day of the month from 1 to 31.                                            | cyclical    |
| week_of_year     | The ordinal week of the year from 1 to 52.                                            | cyclical    |
| week_of_month    | The ordinal week of the month from 1 to 4.                                            | cyclical    |
| day_of_week      | The day of the week from Monday to Sunday.                                            | cyclical    |
| is_weekend       | Indicates whether the date is a weekend or not.                                       | binary      |
| quarter          | The ordinal quarter of the date from 1 to 4.                                          | cyclical    |
| season           | The season Spring/Summer/Fall/Winter.                                                 | categorical |
| fashion_season   | The fashion season Spring/Summer (January to June) or Fall/Winter (July to December). | categorical |
| is_month_start   | Indicates whether the date is the first day of the month.                             | binary      |
| is_month_end     | Indicates whether the date is the last day of the month.                              | binary      |
| is_quarter_start | Indicates whether the date is the first day of the quarter.                           | binary      |
| is_quarter_end   | Indicates whether the date is the last day of the quarter.                            | binary      |
| is_year_start    | Indicates whether the date is the first day of the year.                              | binary      |
| is_year_end      | Indicates whether the date is the last day of the year.                               | binary      |
| is_leap_year     | Indicates whether the date belongs to a leap year.                                    | binary      |
{{% /tab %}}
{{< /tabs >}}

## Time features

{{< button href="https://github.ibm.com/srom/aix360ts/tree/master/aix360ts/transformers/TimeFeatures.py" >}}code{{< /button >}}
{{< button href="https://github.ibm.com/srom/aix360ts/tree/master/examples/transformers/TimeFeatures.py" >}}example{{< /button >}}

Time related features.

{{< tabs "Time features" >}}
{{% tab "aix360ts" %}}
```python
from aix360ts.transformers import TimeFeatures
transformer = TimeFeatures(encode_cyclical_features=False)
```
{{% /tab %}}
{{% tab "input" %}}
```
             sales
date
2017-01-01    21.0
2017-01-02    18.0
2017-01-03     9.0
2017-01-04    18.0
2017-01-05    15.0
...            ...
2019-12-27   923.0
2019-12-28  1194.0
2019-12-29  1341.0
2019-12-30   920.0
2019-12-31   748.0

[1095 rows x 1 columns]
```
{{% /tab %}}
{{% tab "ouput" %}}
```
            hour  minute  second
date
2017-01-01     0       0       0
2017-01-02     0       0       0
2017-01-03     0       0       0
2017-01-04     0       0       0
2017-01-05     0       0       0
...          ...     ...     ...
2019-12-27     0       0       0
2019-12-28     0       0       0
2019-12-29     0       0       0
2019-12-30     0       0       0
2019-12-31     0       0       0

[1095 rows x 3 columns]
```
{{% /tab %}}
{{% tab features %}}
|        | description                | type     |
|:-------|:---------------------------|:---------|
| hour   | The hours of the day.      | cyclical |
| minute | The minutes of the hour.   | cyclical |
| second | The seconds of the minute. | cyclical |
{{% /tab %}}
{{< /tabs >}}

## Encoding Cyclical Features

May time attributes like `month`, `day_of_year`, `hour` etc. all occur in specific cycles and are refered to as **cyclical features**. One way to encode cyclical features is via an ordinal scale. For example, `month` is typically encoded via an *ordinal scale* from 1(January) to 12(December).

The main problem with ordinal scale is that the distance between two feature values does not reflect the true cyclical nature of the data. For example, November and January are equidistant to December, while in the ordinal scale the absolute distance between November and December is 1 while that between December and January if 11. While this may work reasonably well for certain algorithms sometime it is benefical to encode the cyclical feature to reflect the cyclical nature of the attribute.

One method commonly used for encoding a cyclical feature is to perform a sine and cosine transformation of the feature. For each feature {{< katex >}}x{{< /katex >}} which takes ordinal values from {{< katex >}}1,...,K{{< /katex >}} we use a pair of transformed features.
{{< katex display=True >}}
x_{sin} = \sin\left(\frac{2\pi (x-1)}{K}\right)\quad x_{cos} = \cos\left(\frac{2\pi (x-1)}{K}\right)\quad\text{for}\quad x=1,...,K
{{< /katex >}}
Note that is essentially maps the values around a circle. As an added benefit, it is also scaled to the range [-1, 1] which will also aid convergence for neural networks.

## Holiday features

{{< button href="https://github.ibm.com/srom/aix360ts/tree/master/aix360ts/transformers/HolidayFeatures.py" >}}code{{< /button >}}
{{< button href="https://github.ibm.com/srom/aix360ts/tree/master/examples/transformers/HolidayFeatures.py" >}}example{{< /button >}}

Encode country specific holidays as features. We use the python [holidays](https://github.com/dr-prodigy/python-holidays) package.

{{< hint info >}}
A buffer can also be specified before and after the holiday using a tapering triangular window.
{{< /hint >}}

{{< tabs "Holiday features" >}}
{{% tab "aix360ts" %}}
```python
from aix360ts.transformers import HolidayFeatures
transformer = HolidayFeatures(country="IN",
                              buffer=2,
                              include_holiday_name=True)

```
{{% /tab %}}
{{% tab "input" %}}
```
             sales
date
2017-01-01    21.0
2017-01-02    18.0
2017-01-03     9.0
2017-01-04    18.0
2017-01-05    15.0
...            ...
2019-12-27   923.0
2019-12-28  1194.0
2019-12-29  1341.0
2019-12-30   920.0
2019-12-31   748.0

[1095 rows x 1 columns]
```
{{% /tab %}}
{{% tab "ouput" %}}
```
            holiday-IN           holiday-IN-name
date
2017-01-01    0.000000                        no
2017-01-02    0.000000                        no
2017-01-03    0.000000                        no
2017-01-04    0.000000                        no
2017-01-05    0.000000                        no
2017-01-06    0.000000                        no
2017-01-07    0.000000                        no
2017-01-08    0.000000                        no
2017-01-09    0.000000                        no
2017-01-10    0.000000                        no
2017-01-11    0.000000                        no
2017-01-12    0.333333                        no
2017-01-13    0.666667                        no
2017-01-14    1.000000  Makar Sankranti / Pongal
2017-01-15    0.666667                        no
2017-01-16    0.333333                        no
2017-01-17    0.000000                        no
2017-01-18    0.000000                        no
2017-01-19    0.000000                        no
2017-01-20    0.000000                        no
2017-01-21    0.000000                        no
2017-01-22    0.000000                        no
2017-01-23    0.000000                        no
2017-01-24    0.333333                        no
2017-01-25    0.666667                        no
2017-01-26    1.000000              Republic Day
2017-01-27    0.666667                        no
2017-01-28    0.333333                        no
2017-01-29    0.000000                        no
2017-01-30    0.000000                        no
2017-01-31    0.000000                        no
2017-02-01    0.000000                        no
2017-02-02    0.000000                        no
2017-02-03    0.000000                        no
2017-02-04    0.000000                        no
2017-02-05    0.000000                        no
2017-02-06    0.000000                        no
2017-02-07    0.000000                        no
2017-02-08    0.000000                        no
2017-02-09    0.000000                        no
2017-02-10    0.000000                        no
2017-02-11    0.000000                        no
2017-02-12    0.000000                        no
2017-02-13    0.000000                        no
2017-02-14    0.000000                        no
2017-02-15    0.000000                        no
2017-02-16    0.000000                        no
2017-02-17    0.000000                        no
2017-02-18    0.000000                        no
2017-02-19    0.000000                        no
```
{{% /tab %}}
{{% tab features %}}
|                 | description                                        | type        |
|:----------------|:---------------------------------------------------|:------------|
| holiday-IN      | Indicates whether the date is a IN holiday or not. | continuous  |
| holiday-IN-name | The holiday name.                                  | categorical |
{{% /tab %}}
{{< /tabs >}}

## Trend features

{{< button href="https://github.ibm.com/srom/aix360ts/tree/master/aix360ts/transformers/TrendFeatures.py" >}}code{{< /button >}}
{{< button href="https://github.ibm.com/srom/aix360ts/tree/master/examples/transformers/TrendFeatures.py" >}}example{{< /button >}}

Features to model simple polynomial trend. Adds features of the form {{< katex >}}t,t^2,..{{< /katex >}}. High degrees can cause overfitting, do not go above two unless needed.

{{< tabs "Trend features" >}}
{{% tab "aix360ts" %}}
```python
from aix360ts.transformers import TrendFeatures
transformer = TrendFeatures(degree=3)
```
{{% /tab %}}
{{% tab "input" %}}
```
             sales
date
2017-01-01    21.0
2017-01-02    18.0
2017-01-03     9.0
2017-01-04    18.0
2017-01-05    15.0
...            ...
2019-12-27   923.0
2019-12-28  1194.0
2019-12-29  1341.0
2019-12-30   920.0
2019-12-31   748.0

[1095 rows x 1 columns]
```
{{% /tab %}}
{{% tab "ouput" %}}
```
            sales_trend_linear  sales_trend_quadratic  sales_trend_cubic
date
2017-01-01                   0                      0                  0
2017-01-02                   1                      1                  1
2017-01-03                   2                      4                  8
2017-01-04                   3                      9                 27
2017-01-05                   4                     16                 64
...                        ...                    ...                ...
2019-12-27                1090                1188100         1295029000
2019-12-28                1091                1190281         1298596571
2019-12-29                1092                1192464         1302170688
2019-12-30                1093                1194649         1305751357
2019-12-31                1094                1196836         1309338584

[1095 rows x 3 columns]
```
{{% /tab %}}
{{% tab features %}}
|                       | description                                                      | type       |
|:----------------------|:-----------------------------------------------------------------|:-----------|
| sales_trend_linear    | Feature to model simple polynomial (of degree 1) trend in sales. | continuous |
| sales_trend_quadratic | Feature to model simple polynomial (of degree 2) trend in sales. | continuous |
| sales_trend_cubic     | Feature to model simple polynomial (of degree 3) trend in sales. | continuous |
{{% /tab %}}
{{< /tabs >}}

## References

- https://www.avanwyk.com/encoding-cyclical-features-for-deep-learning/
