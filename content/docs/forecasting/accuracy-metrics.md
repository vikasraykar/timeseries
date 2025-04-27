---
bookShowToc: true
weight: 104
title: "Accuracy metrics"
---

# Forecast accuracy metrics

*A compilation of various metrics used to measure forecast accuracy*

{{< button href="https://github.ibm.com/retail-supply-chain/forecasting/blob/master/gists/gist_univariate_metrics.py" >}}gist{{< /button >}}
{{< button href="https://github.ibm.com/retail-supply-chain/forecasting/tree/master/forecasting/metrics" >}}code repo{{< /button >}}


Let $y(t)$ be the actual observation for time period $t$ and let $f(t)$ be the forecast for the same time period. Let $n$ the length of the training dataset (number of historical observations), and $h$ the forecasting horizon.

## Forecast errors

Based on how we measure the forecast error $e(t)$ at a time point $t$ several metrics are defined. A forecast error is the difference between an observed value and its forecast.

acronym | error type | equation | scale independent
:--- | :--- | :--- | :---
`*E` | [error](#scale-dependent-metrics) | {{< katex >}}e(t)=y(t)-f(t){{< /katex >}} | `no`
`*PE` | [percentage error](#percentage-error-metrics) | {{< katex >}}pe(t)=\frac{y(t)-f(t)}{y(t)}{{< /katex >}} | `yes`
`*RE` | [relative error](#relative-error-metrics) | {{< katex >}}re(t)=\frac{y(t)-f(t)}{y(t)-f^*(t)}{{< /katex >}} | `yes`
`*SE` | [scaled error](#scaled-error-metrics) | {{< katex >}}se(t)=\frac{y(t)-f(t)}{s}{{< /katex >}} | `yes`

We are evaluating the accuracy over $h$ forecasting time steps. The final forecast accuracy metric is then an aggregation (via mean, median etc.) over over $h$ time steps.

acronym | aggregation
:--- | :---
`M*` | Mean
`MA*` | Mean Absolute
`Md*` | Median
`MdA*` | Median Absolute
`GM*` | Geometric Mean
`GMA*` | Geometric Mean Absolute
`MS*` | Mean Squared
`RMS*` | Root Mean Squared

`{M|Md|GM}{-|A|S}{E|PE|RE|SE}`

## TLDR 

While there are several metrics we recommend the following 6 metrics to be definitely included in a forecasting toolkit.

acronym | name | comments
--- | --- | ---
[`MAE`](#mae) | Mean Absolute Error | For assessing accuracy on a a single time series use MAE because it is easiest to interpret. However, it cannot be compared across series because it is scale dependent.
[`RMSE`](#rmse) | Root Mean Square Error | A forecast method that minimises the MAE will lead to forecasts of the median, while minimising the RMSE will lead to forecasts of the mean. Hence RMSE is also widely used, despite being more difficult to interpret.
[`MAPE`](#mae) | Mean Absolute Percentage Error | MAPE has the advantage of being scale independent and hence can be used to compare forecast performance between different time series. However MAPE has the disadvantage of being infinite or undefined if there are zero values in the time series, as is frequent for intermittent demand data.
[`MASE`](#mase) | Mean Absolute Scaled Error | MASE is a scale free error metric and can be used to compare forecast methods on a single time series and also to compare forecast accuracy between series. It is also well suited for intermittent demand time series since is never gives infinite or undefined values.
[`wMAPE`](#wmape) | (volume) weighted Mean Absolute Percentage Error | The is MAPE which is weighted by the volume.
[`MSPL`](#mspl) | Mean Scaled Pinball Loss | The metric to use for quantile forecasts.

## Categorizing metrics

category | description
:--- | :---
`Accuracy` | How close are the forecasts to the true value?
`Bias` | Is there a systematic under- or over-forecasting?
`Error Variance` | The spread of the point forecast error around the mean.
`Accuracy Risk` | The variability in the accuracy metric over multiple forecasts.
`Stability` | The degree to twhich forecasts remain unchanged subject to minor variations in the underlying data.

## Scale dependent metrics

Let $e(t)$ be the *one-step forecast error*, which is the difference between the observation $y(t)$ and the forecast made using all observations up to but not including $y(t)$.

{{< katex >}}
e(t)=y(t)-f(t)
{{< /katex >}}

By **scale dependent** we mean that the value of the metric depends on the scale of the data.

### ME
*Mean Error*

{{< tabs "ME" >}}
{{< tab "ME" >}}
{{< katex >}}
\text{ME}=\text{mean}\left(e(t)\right)
{{< /katex >}}
{{< /tab >}}
{{< tab "notes" >}}
- ME is likely to be small since positive and negative errors tend to offset one another.
- ME will tell you if there is a systematic under- or over-forecasting, called the **forecast bias**.
- ME does not give much indication as the the size of the typical errors.
{{< /tab >}}
{{< /tabs >}}

### MAE
*Mean Absolute Error*

{{< tabs "MAE" >}}
{{< tab "MAE" >}}
{{< katex >}}
\text{MAE}=\text{mean}\left(|e(t)|\right)
{{< /katex >}}
{{< /tab >}}
{{< tab "notes" >}}
- MAE has the advantage of being more interpretable and is easier to explain to users.
- MAE is also sometimes referred to as MAD (Mean Absolute Deviation).
- A forecast method that minimises the MAE will lead to forecasts of the median, while minimising the RMSE will lead to forecasts of the mean.
{{< /tab >}}
{{< /tabs >}}

### MdAE
*Median Absolute Error*

{{< tabs "MdAE" >}}
{{< tab "MdAE" >}}
{{< katex >}}
\text{MdAE}=\text{median}\left(|e(t)|\right)
{{< /katex >}}
{{< /tab >}}
{{< tab "notes" >}}
- MdAE has the advantage of being more interpretable and is easier to explain to users.
{{< /tab >}}
{{< /tabs >}}

### GMAE
*Geometric Mean Absolute Error*

{{< tabs "GMAE" >}}
{{< tab "GMAE" >}}
{{< katex >}}
\text{GMAE}=\text{gmean}\left(|e(t)|\right)
{{< /katex >}}
{{< /tab >}}
{{< tab "notes" >}}
- GMAE has the flaw of being equal to zero when any of the error terms are zero.
- GMAE is same as GRMSE because the square root and the square cancel each other in a geometric mean.
{{< /tab >}}
{{< /tabs >}}

### MSE
*Mean Square Error*

{{< tabs "MSE" >}}
{{< tab "MSE" >}}
{{< katex >}}
\text{MSE}=\text{mean}\left(e(t)^2\right)
{{< /katex >}}
{{< /tab >}}
{{< tab "notes" >}}
- MSE has the advantage of being easier to handle mathematically and is commonly used in loss functions for optimization.
- MSE tends to penalize large errors over smaller ones.
{{< /tab >}}
{{< /tabs >}}

### RMSE
*Root Mean Square Error*

{{< tabs "RMSE" >}}
{{< tab "RMSE" >}}
{{< katex >}}
\text{RMSE}=\sqrt{\text{mean}\left(e(t)^2\right)}
{{< /katex >}}
{{< /tab >}}
{{< tab "notes" >}}
- RMSE has the advantage of being easier to handle mathematically and is commonly used in loss functions for optimization.
- RMSE tends to penalize large errors over smaller ones.
- A forecast method that minimises the MAE will lead to forecasts of the median, while minimising the RMSE will lead to forecasts of the mean.
{{< /tab >}}
{{< /tabs >}}

## Percentage error metrics

For scale dependent metrics that the value of the metric depends on the scale of the data. Therefore, they do not facilitate comparison across *different time series* and for *different time intervals*.

First we define a relative or percentage error as

{{< katex >}}
pe(t)=\frac{y(t)-f(t)}{y(t)}
{{< /katex >}}

Percentage errors have the advantage of being **scale independent** and hence useful to compare performance between different time series.

{{< hint warning >}}
When the time series values are very close to zero (especially for intermittent demand data), the relative or percentage error is meaningless and is actually not defined when the value is zero.
{{< /hint >}}

### MPE
*Mean Percentage Error*

{{< tabs "MPE" >}}
{{< tab "MPE" >}}
{{< katex >}}
\text{MPE}=\text{mean}\left(pe(t)\right)
{{< /katex >}}
{{< /tab >}}
{{< tab "notes" >}}
- MPE is likely to be small since positive and negative errors tend to offset one another.
- MPE will tell you if there is a systematic under- or over-forecasting, called the **forecast bias**.
- MPE does not give much indication as the the size of the typical errors.
{{< /tab >}}
{{< /tabs >}}

### MAPE
*Mean Absolute Percentage Error*

{{< tabs "MAPE" >}}
{{< tab "MAPE" >}}
{{< katex >}}
\text{MAPE}=\text{mean}\left(|pe(t)|\right)
{{< /katex >}}
{{< /tab >}}
{{< tab "notes" >}}
- MAPE puts heavier penalty on positive errors than on negative errors.
- This observation has led to the use of symmetric MAPE or [sMAPE](#smape).
{{< /tab >}}
{{< /tabs >}}

### MdAPE
*Median Absolute Percentage Error*

{{< tabs "MdAPE" >}}
{{< tab "MdAPE" >}}
{{< katex >}}
\text{MdAPE}=\text{median}\left(|pe(t)|\right)
{{< /katex >}}
{{< /tab >}}
{{< tab "notes" >}}
{{< /tab >}}
{{< /tabs >}}

### sMAPE
*Symmetric Mean Absolute Percentage Error*

{{< tabs "sMAPE" >}}
{{< tab "sMAPE" >}}
{{< katex >}}
\text{sMAPE}=\text{mean}\left(2\cdot\left|\frac{y(t)-f(t)}{y(t)+f(t)}\right|\right)
{{< /katex >}}
{{< /tab >}}
{{< tab "notes" >}}
- MAPE puts heavier penalty on positive errors than on negative errors.
- This observation has led to the use of symmetric MAPE or sMAPE.
- sMAPE which was used in the M3 forecasting competition.
{{< /tab >}}
{{< /tabs >}}

### sMdAPE
*Symmetric Median Absolute Percentage Error*

{{< tabs "sMdAPE" >}}
{{< tab "sMdAPE" >}}
{{< katex >}}
\text{sMdAPE}=\text{median}\left(2\cdot\left|\frac{y(t)-f(t)}{y(t)+f(t)}\right|\right)
{{< /katex >}}
{{< /tab >}}
{{< tab "notes" >}}
{{< /tab >}}
{{< /tabs >}}

### MSPE
*Mean Square Percentage Error*

{{< tabs "MSPE" >}}
{{< tab "MSPE" >}}
{{< katex >}}
\text{MSE}=\text{mean}\left(pe(t)^2\right)
{{< /katex >}}
{{< /tab >}}
{{< tab "notes" >}}
{{< /tab >}}
{{< /tabs >}}

### RMSPE
*Root Mean Square Percentage Error*

{{< tabs "RMSPE" >}}
{{< tab "RMSPE" >}}
{{< katex >}}
\text{MSE}=\sqrt{\text{mean}\left(pe(t)^2\right)}
{{< /katex >}}
{{< /tab >}}
{{< tab "notes" >}}
{{< /tab >}}
{{< /tabs >}}

### MAAPE
*Mean Absolute Arctangent Percent Error*

{{< tabs "MAAPE" >}}
{{< tab "MAAPE" >}}
{{< katex >}}
\text{MAAPE}= \text{mean} \left(\text{arctan}\left(|pe(t)|\right)\right) = \text{mean} \left(\text{tan}^{-1}\left(|pe(t)|\right)\right)
{{< /katex >}}
{{< /tab >}}
{{< tab "notes" >}}
- MAPE produces infinite or undefined values  when the  actual values are zero or close to zero, which is a common with intermittent data in retail. While other alternate measures have been proposed to deal with this disadvantage of MAPE, it still remains the preferred method of business forecasters following its intuitive interpretation as absolute percentage error. Mean Absolute Arctangent Percent Error is an alternative measure that has the same interpretation as an  absolute percentage error (APE) but can overcome MAPE’s disadvantage of generating infinite values.
- **Boundedness** MAAPE is bounded and it varies from 0 to $\pi/2$. MAAPE does not go to infinity even with close-to-zero actual values, which is a significant advantage of 
MAAPE  over MAPE.
- <img src="/img/arctan.png"/>
- **Robustness** Absolute Arctangent Percent Error given by $AAPE = \text{arctan}\left(|pe(t)|\right)$ converges to $\pi/2$ for large forecast errors,  thus limits the influence of outliers, which often distort the calculation of the overall forecast accuracy. Therefore, MAAPE can be particularly useful if there are extremely large forecast errors as a result of mistaken or incorrect  measurements.
- However, if the extremely large forecast errors are considered 
as genuine variations that might have some important business implications, rather than being due to mistaken or incorrect measurements, 
MAAPE would not be appropriate. 
- MAAPE is also asymmetric but has a more balanced penalty between positive and negative errors than MAPE. 
{{< /tab >}}
{{< tab "references" >}}
Kim, Sungil, and Heeyoung Kim. [A new metric of absolute percentage error for intermittent demand forecasts](https://core.ac.uk/reader/82178886). International Journal of Forecasting 32.3 (2016): 669-679.
{{< /tab >}}
{{< /tabs >}}


## Relative error metrics

An alternative to percentage error for the calculation of scale-independent metrics involves dividing each error by the error obtained using a baseline forecasting algorithm (typically the naive baseline $f^*(t)=y(t-1)$).

{{< katex >}}
re(t)=\frac{e(t)}{e^*(t)}=\frac{y(t)-f(t)}{y(t)-f^*(t)}
{{< /katex >}}

### MARE
*Mean Absolute Relative Error*

{{< tabs "MARE" >}}
{{< tab "MARE" >}}
{{< katex >}}
\text{MARE}=\text{mean}\left(|re(t)|\right)
{{< /katex >}}
{{< /tab >}}
{{< tab "notes" >}}
{{< /tab >}}
{{< /tabs >}}

### MdARE
*Median Absolute Relative Error*

{{< tabs "MdARE" >}}
{{< tab "MdARE" >}}
{{< katex >}}
\text{MdARE}=\text{median}\left(|re(t)|\right)
{{< /katex >}}
{{< /tab >}}
{{< tab "notes" >}}
{{< /tab >}}
{{< /tabs >}}

### GMARE
*Geometric Mean Absolute Relative Error*

{{< tabs "GMARE" >}}
{{< tab "GMARE" >}}
{{< katex >}}
\text{GMARE}=\text{gmean}\left(|re(t)|\right)
{{< /katex >}}
{{< /tab >}}
{{< tab "notes" >}}
{{< /tab >}}
{{< /tabs >}}

### U-statistic
*Theil's $U$-statistic*

{{< tabs "U-statistic" >}}
{{< tab "U-statistic" >}}
{{< katex >}}
\text{U}= \sqrt{\frac{A}{B}},\text{where},
{{< /katex >}}
{{< katex >}}
\quad\text{A}=\sum_{t=1}^{n-1}\left(\frac{y(t+1)-f(t+1)}{y(t)}\right)^2\quad\text{B}=\sum_{t=1}^{n-1}\left(\frac{y(t+1)-y(t)}{y(t)}\right)^2
{{< /katex >}}
{{< /tab >}}
{{< tab "notes" >}}
- Large errors given more weight then small errors and also provides a relative basis for comparison with naive (**NF1** forecast $f(t)=y(t-1)$) methods.
- The numerator is similar to the MAPE and the denominator to the MAPE of the naive forecast method.
- Smaller the better and should be less than 1.
- $U=1$ The naive method is as good as the forecasting technique being evaluated.
- $U<1$ The forecasting technique being evaluated is better than the naive method. The smaller the $U$-statistic, the better the forecasting method is relative to the naive method.
- $U>1$ There is no point in using the forecasting method, since using a naive method will produce better results.
{{< /tab >}}
{{< /tabs >}}


## Scaled error metrics

This generally involves scaling the error term by a suitable scale parameter.

{{< katex >}}
se(t)=\frac{e(t)}{s}=\frac{y(t)-f(t)}{s}
{{< /katex >}}

### MAD/Mean
*MAD/Mean ratio*

{{< tabs "MAD/Mean" >}}
{{< tab "MAD/Mean" >}}
{{< katex >}}
\text{MAD/Mean}=\text{mean}\left(|se(t)|\right),\text{where } s = \frac{1}{n}\sum_{i=1}^{n} y(t)
{{< /katex >}}
{{< /tab >}}
{{< tab "notes" >}}
- The scale parameter $s$ is the *in-sample* mean of the time series.
{{< katex >}}
s = \frac{1}{n}\sum_{i=1}^{n} y(t)
{{< /katex >}}
- Assumes stationarity, that is, the mean is stable over time, which is generally not true for data which shows trend, seasonality, or other patterns.
{{< /tab >}}
{{< tab "references" >}}
{{< /tab >}}
{{< /tabs >}}

### wMAPE
*(volume) weighted Mean Absolute Percentage Error*

{{< tabs "wMAPE" >}}
{{< tab "wMAPE" >}}
{{< katex >}}
\text{wMAPE}=\text{mean}\left(|se(t)|\right),\text{where } s = \sum_{i=n+1}^{n+h} |y(t)|
{{< /katex >}}
{{< /tab >}}
{{< tab "notes" >}}
- The scale parameter $s$ is the sum of the absolute value of the time series we are evaluation on (sometimes also called *volume*).
{{< katex >}}
s = \sum_{i=n+1}^{n+h} |y(t)|
{{< /katex >}}
-  So essentially wMAPE is the Sum of Absolute errors divided by the Sum of the Actuals. 
- This is equivalent to weighing the percentage error $pe(t)$ by a scale term, hence the term wMAPE.
{{< katex >}}
\frac{y(t)}{s} \cdot pe(t) = \frac{y(t)}{s} \cdot \frac{y(t)-f(t)}{y(t)}
{{< /katex >}}
- MAPE is sentitive to very small changes in low volume data. However, wMAPE is less sensitive to it.
- MAPE assumes that, absolute error on each item is equally important. Large error on a low-value item can unfairly skew the overall
error. In contrast, wMAPE penalizes more for errors in high volume data as compared to low volume data. 
- wMAPE do not have divide by zero error.
{{< /tab >}}
{{< tab "references" >}}
{{< /tab >}}
{{< /tabs >}}

### MASE
*Mean Absolute Scaled Error*

{{< tabs "MASE" >}}
{{< tab "MASE" >}}
{{< katex >}}
\text{MASE}=\text{mean}\left(|se(t)|\right),\text{where } s = \frac{1}{n-1}\sum_{i=2}^{n} |y(t)-y(t-1)|
{{< /katex >}}
{{< /tab >}}
{{< tab "notes" >}}
- MASE is independent of the scale of the data.
- The scale parameter $s$ is  the *in-sample* MAE from the naive forecast method.
{{< katex >}}
s = \frac{1}{n-1}\sum_{i=1}^{n} |y(t)-y(t-1)|
{{< /katex >}}
- A scaled error is less than one if it arises from a better forecast than the average one-step, naive forecast computed in-sample.
- The only scenario under which MASE would be infinite or undefined is when all historical observations are equal.
{{< /tab >}}
{{< tab "references" >}}
- Rob J. Hyndman and Anne B. Koehler, [Another look at measures of forecast accuracy](https://github.ibm.com/retail-supply-chain/planning/files/582845/Another.look.at.measures.of.forecast.accuracy.pdf), International Journal of Forecasting, Volume 22, Issue 4, 2006.
- Rob J Hyndman, [Another look at forecast-accuracy metrics for intermittent demand](https://github.ibm.com/retail-supply-chain/planning/files/582847/Metrics.for.intermitted.demand.pdf). Chapter 3.4, pages 204-211, in "Business Forecasting: Practical Problems and Solutions", John Wiley & Sons, 2015.
{{< /tab >}}
{{< /tabs >}}

### RMSSE
*Root Mean Squared Scaled Error*

{{< tabs "RMSSE" >}}
{{< tab "RMSSE" >}}
{{< katex >}}
\text{RMSSE}=\sqrt{\text{mean}\left(se(t)^2\right)},\text{where } s = \frac{1}{n-1}\sum_{i=2}^{n} |y(t)-y(t-1)|
{{< /katex >}}
{{< /tab >}}
{{< tab "notes" >}}
- This is the root mean squared variant of MASE.
- This is suited for time series characterized by intermittency, involving spardic unit sales with lots of zeros.
- Absolute errors used by MASE are optimized for the median and would assign lower scores to forecasting methods that derive the forecasts close to zero. RMSSE buils on squared errors, which are optimized for the mean.
- The measure is scale independent, meaning that it can be effectively used to compare forecasts across series with different scales.
- The measure penalizes positive and negative forecast errors, as well as large and small forecasts, equally, thus being symmetric.
- It can be safely computed as it does not rely on divisions with values that could be equal or close to zero (e.g. as done in percentage errors when $y(t)=0$ or relative errors when the error of the benchmark used for scaling is zero).
{{< /tab >}}
{{< tab "references" >}}
- [The M5 Competition](https://mofc.unic.ac.cy/m5-competition/)
- The M5 series are characterized by intermittency, involving sporadic unit sales with lots of zeros. This means that absolute errors, which are optimized for the median, would assign lower scores (better performance) to forecasting methods that derive forecasts close to zero. However, the objective of M5 is to accurately forecast the average demand and for this reason, the accuracy measure used builds on squared errors, which are optimized for the mean.
{{< /tab >}}
{{< /tabs >}}

### MSLE
*Mean Squared Log Error*

{{< tabs "MSLE" >}}
{{< tab "MSLE" >}}
{{< katex >}}
\text{MSLE}= \text{mean} \left(\log(y(t)+1)- \log(f(t)+1)\right) ^2
{{< /katex >}}
{{< /tab >}}
{{< tab "notes" >}}
- The introduction of the logarithm makes MSLE only care about the relative difference between the true and the predicted value. Thus making
this scale independent.

- Robustness: In the case of MSE, the presence of outliers can explode the error term to a very high value. 
But, in the case of MLSE the outliers are drastically scaled down therefore nullifying their effect. 
 
- Biased penalty: MSLE penalizes underestimates more than overestimates because of an asymmetry in the error curve. 
- The reason ‘1’ is added to both $y(t)$ and $f(t)$ is for mathematical convenience since $\log(0)$ is not defined but both $y(t)$
 or $f(t)$ can be 0. 
{{< /tab >}}
{{< tab "references" >}}
Kim, Sungil, and Heeyoung Kim. [A new metric of absolute percentage error for intermittent demand forecasts](https://core.ac.uk/reader/82178886). International Journal of Forecasting 32.3 (2016): 669-679.
{{< /tab >}}
{{< /tabs >}}

## Correlation based metrics

### PCC 
*Pearson correlation coefficient ($r$)*

{{< tabs "PCC" >}}
{{< tab "PCC" >}}
{{< katex >}}
r = \frac{\sum_{t = n+1}^{n+h} (y(t) - \text{mean} (y(t)))(f(t) - \text{mean} (f(t)))}{\sqrt{\sum_{t = n+1}^{n+h} (y(t) - \text{mean} (y(t)))^2}\sqrt{\sum_{t = n+1}^{n+h} (f(t) - \text{mean} (f(t)))^2}}
{{< /katex >}}
{{< /tab >}}
{{< tab "notes" >}}
- Pearson correlation measures how two continuous signals co-vary over time and indicate the linear relationship as a number between -1 (negatively correlated) to 0 (not correlated) to 1 (perfectly correlated).
- Sensitive to outliers.
- Based on the assumption of  homoscedasticity of the data (variance of your data is homogenous across the data range). 
- It is a scale independent and offset independent metric.
- Caution is necessary when using Pearson's correlation coefficient for model selection (see references).

{{< /tab >}}
{{< tab "references" >}}
- Waldmann, Patrik. [On the use of the Pearson correlation coefficient for model evaluation in genome-wide prediction](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6781837/) Frontiers in Genetics 10 (2019): 899.
- Armstrong, Richard A. [Should Pearson's correlation coefficient be avoided?.](https://onlinelibrary.wiley.com/doi/full/10.1111/opo.12636) Ophthalmic and Physiological Optics 39.5 (2019): 316-327. 
{{< /tab >}}
{{< /tabs >}}

### KRCC 
*Kendall rank correlation coefficient ($\tau $)*

A rank correlation coefficient measures the degree of similarity between two rankings, and can be used to assess the significance of the relation between them. 

For $t_i < t_j$, two sequences $y(t)$ and $f(t)$ are said to be *concordant* if the ranks for both elements agree: that is, if both 
$f(t_i)>f(t_j)$ and $y(t_i)>y(t_j)$; or if both $f(t_i)<f(t_j)$ and $y(t_i)<y(t_j)$. 

They are said to be *discordant*, if $fx(t_i)>f(t_j)$ and $y(t_i)\<y(t_j)$ or if $f(t_i) \< f(t_j)$ and $y(t_i)>y(t_j) $. 

If $x(t_i)=x(t_j)$ and $y(t_i)=y(t_j) $, the pair is neither concordant nor discordant.

{{< tabs "KRCC" >}}
{{< tab "KRCC" >}}
{{< katex >}}
\tau = \frac{(\text{number of concordant pairs}) - (\text{number of discordant pairs})}{ 
 {n \choose 2} }
 {{< /katex >}}
{{< /tab >}}
{{< tab "notes" >}}
- The denominator ${n \choose 2}$ is the total number of pair combinations, so the coefficient must be in the range $−1 \leq \tau \leq 1$.
- If the agreement between the two rankings is perfect (i.e., the two rankings are the same) the coefficient has value 1.
- If the disagreement between the two rankings is perfect (i.e., one ranking is the reverse of the other) the coefficient has value −1.
- If $X$ and $Y$ are independent, then we would expect the coefficient to be approximately zero.
- An explicit expression for Kendall's rank coefficient is $\tau= \frac{2}{n(n-1)}\sum_{i<j} \text{sign}(x_i-x_j)\text{sign}(y_i-y_j)$.
{{< /tab >}}
{{< /tabs >}}



### NGini
*Normalized Gini Coefficient*

Measures how  far away the true  values sorted by predictions are from  a random  sorting, in terms  of number  of swaps. 
If $y(t)$ represents "True Values" and $f(t)$ represents the corresponding predictions. Now we let $y_f(t)$ represent "Sorted True" values,
 where $y(t)$ is sorted based of values of $f(t)$. 
 
 
Let $\text{swaps}_{\text{sorted}}$ represent the  number of swaps of adjacent digits
 (like in bubble sort) it would take to get from the "Sorted True" 
state to the "True Values" state. 

And $\text{swaps}_{\text{random}}$ represents the number of swaps it would take on average to get from a random state to the "True Values" state.


{{< tabs "NGC" >}}
{{< tab "NGC" >}}
{{< katex >}}
\text{NGini} = \frac{(\text{swaps}_{\text{random}}) - (\text{swaps}_{\text{sorted}})}{ 
 \text{swaps}_{\text{random}} }
 {{< /katex >}}
{{< /tab >}}
{{< /tabs >}}

## Poisson metrics

These are mainly suitable for sparse intermiited sales data or count data in general and are essentailly based on the Poisson goodness-of-fit metrics.

### PNLL
*Poisson Negative Log Likelihood*

{{< tabs "PNLL" >}}
{{< tab "PNLL" >}}
{{< katex >}}
\text{PNLL} = \text{mean}(f(t)-y(t) \cdot\log f(t))
{{< /katex >}}
{{< /tab >}}
{{< tab "notes" >}}

{{< /tab >}}
{{< tab "references" >}}
- https://ibm.ent.box.com/file/686833894993
- https://data.princeton.edu/wws509/notes/c4.pdf
- https://github.ibm.com/srom/forecasting/blob/master/fc_app_pak/metrics/metrics.py
{{< /tab >}}
{{< /tabs >}}

### P-DEV
*Poisson Deviance*


{{< tabs "P-DEV" >}}
{{< tab "P-DEV" >}}
{{< katex >}}
D = 2 \cdot \text{mean} \left(y(t)\cdot \log\left(\frac{y(t)}{f(t)}\right)- \left(y(t) - f(t)\right)\right)
{{< /katex >}}
{{< /tab >}}
{{< tab "notes" >}}
- A measure of discrepancy between observed and fitted values is the deviance.
- The deviance indicates the extent to which the likelihood of the saturated model exceeds the likelihood of the proposed model. 
- If the proposed model has a good fit, the deviance will be small. If the proposed model has a bad fit, the deviance will be high.


{{< /tab >}}
{{< tab "references" >}}
- https://ibm.ent.box.com/file/686833894993
- https://data.princeton.edu/wws509/notes/c4.pdf
- https://github.ibm.com/srom/forecasting/blob/master/fc_app_pak/metrics/metrics.py
{{< /tab >}}
{{< /tabs >}}

### CHI^2
*Pearson’s chi-squared statistic*

{{< tabs "CHI^2" >}}
{{< tab "CHI^2" >}}
{{< katex >}}
\chi^2 = \sum \frac{(y(t)-f(t))^2}{f(t)} 
{{< /katex >}}
{{< /tab >}}
{{< tab "notes" >}}
The chi-squared statistic can be used  to test the hypothesis that observed data follow a particular
distribution (Poisson).
{{< /tab >}}
{{< tab "references" >}}
- https://ibm.ent.box.com/file/686833894993
- https://data.princeton.edu/wws509/notes/c4.pdf
- https://github.ibm.com/srom/forecasting/blob/master/fc_app_pak/metrics/metrics.py
{{< /tab >}}
{{< /tabs >}}


### D^2
*Percent of deviance explained*

{{< tabs "D^2" >}}
{{< tab "D^2" >}}
{{< katex >}}
D^2 = \frac{\text{Null deviance}- \text{Residual deviance}}{\text{Null deviance}}
{{< /katex >}}
{{< /tab >}}
{{< tab "notes" >}}
In the defintion, the null deviance is the deviance of the model with the intercept only, 
and the residual deviance is the deviance that remains unexplained by the model after all final variables have been included.

- Null deviance refers to the null model (i.e. an intercept-only model).  A low null deviance implies that the data can be 
modeled well merely using the intercept. If the null deviance is low, you should consider using few features for modeling the data.
- For our purposes null deviance can be defined using the mean of observations and null deviance can be defined as
{{< katex >}}
D_{0} = 2 \cdot \text{mean} \left(y(t)\cdot \log\left(\frac{y(t)}{\text{mean}(y(t))}\right)- \left(y(t) - \text{mean}(y(t))\right)\right)
{{< /katex >}}

- Residual deviance refers to the trained model and a low residual deviance implies that the model you have trained is appropriate.
{{< /tab >}}
{{< tab "references" >}}
- https://ibm.ent.box.com/file/686833894993
- https://data.princeton.edu/wws509/notes/c4.pdf
- https://github.ibm.com/srom/forecasting/blob/master/fc_app_pak/metrics/metrics.py
{{< /tab >}}
{{< /tabs >}}



## Metrics for Quantile Forecasts

Point forecast $f(t)$ and the observed/real value $y(t)$  at time $t$ can be expressed as
{{< katex >}}
y(t) = f(t) + \epsilon_t
{{< /katex >}}
where $\epsilon_t$ is the corresponding error. The most common extension from point to probabilistic forecasts is
to construct prediction intervals. A number of methods can be
used for this purpose, the most popular take into account both the
point forecast and the corresponding error: the center of the prediction interval
at the $(1-\alpha)$  confidence level is set equal to $f(t)$ and its bounds are defined by 
the $\alpha/2$ and $(1-\alpha/2)$ quantiles of the cumulative distribution function (CDF) of 
$\epsilon_t$ as  $[f_L(t),f_U(t)]$. 

Another popular alternative  to this approach is to find the distribution of the point forecast $f(t)$ itself, as $F_t$, where
$F_t$ is random variable with  an explicit density distribution, that is 

{{< katex >}}
F_{n+h}(x) = P (f(n+h) \leq x \,\,|\,\, \{y(t)\}_{t=1, \dots, n})
{{< /katex >}}

From a practical perspective, a probabilistic forecast $F_{n+h}$ is usually represented as a histogram 
where each bin represents a range of future demand, and where the bin height represents the estimated probability that 
future demand will fall within the specific range associated to a bucket.

When evaluating a probabilistic forecast, the main challenge is that
we never observe the true distribution of the underlying process. In
other words, we cannot compare the predictive distribution, $F_{t}$, or the
prediction interval, $[f_L(t),f_U(t)]$, with the actual distribution of the $y(t)$, but only with observed past values $y(t)$, $t< n$.

Scoring rules provide summary measures for the evaluation of probabilistic forecasts, by assigning a numerical score,
$S(F\_{t}, y(t))$, based on the predictive distribution, $F_{t}$ and on the actual
observed value $y(t)$. A scoring rule is called proper if given the true distribution of $y(t)$ as $Y\_t$, then
{{< katex >}}
S(F_{t}, Y_t) \leq S(F_{t}, y(t))
{{< /katex >}}
Now we present two proper scoring rules: Pinball loss and Continuous Ranked Probability Score
 
### Pinball Loss

In general $y(t)$ is a random variable with it own distribution and what is typically predicted as the **point forecast**, $f(t)$, is the **mean** $\mathbb{E}[y(t)]$ of this distribution. A **quantile forecast** $f_p(t)$ is an estimation of the $p^{th}$ quantile of the distribution. The $p^{th}$ quantile of $y(t)$ is defined as
{{< katex >}}
\mathbb{Q}_p[y(t)] = \left\{x : \text{Pr}[y(t) \leq x] = p \right\}
{{< /katex >}}

The **pinball loss function** is a metric used to assess the accuracy of a quantile forecast.
{{< katex >}}
pl(t,p) = \begin{cases}
 \ p\left(y(t)-f_p(t)\right)\quad\text{if}\quad  f_p(t) \leq y(t)\\
  (1-p)\left(f_p(t)-y(t)\right)\quad\text{if}\quad f_p(t) > y(t)
\end{cases}
{{< /katex >}}

<img src="/img/pinball-loss.jpg"/>
*The pinball loss function has been named after its shape that looks like the trajectory of a ball on a [pinball](https://en.wikipedia.org/wiki/Pinball).*

- The pinball function is always non-negative and farther the value of forecast from real value, larger the value of the loss function.
- The slope is used to reflect the desired imbalance in the quantile forecast.
- You want the truth $y(t)$ to be less than your prediction  100$p$% of the time. For example, your prediction for the 0.25 quantile should be such that the true value is less than your prediction 25% of the time. The loss function enforces this by penalizing cases when the truth is on the more unusual side of your prediction more heavily than when it is on the expected side.
- The metric further encourages you to predict as narrow a range of possibilities as possible by weighting the penalty by the absolute difference between the truth and your quantile.
- Stockout and inventory costs of the quantile.

### MSPL

*Mean Scaled Pinball Loss*

{{< tabs "MSPL" >}}
{{< tab "MSPL" >}}
{{< katex >}}
\text{MSPL}(p)=\text{mean}\left(\frac{pl(t,p)}{s}\right),\text{where } s = \frac{1}{n-1}\sum_{i=2}^{n} |y(t)-y(t-1)|
{{< /katex >}}
{{< /tab >}}
{{< tab "notes" >}}
- Like MASE MSPL is independent of the scale of the data. The scale parameter $s$ is  the *in-sample* MAE from the naive forecast method.
{{< katex >}}
s = \frac{1}{n-1}\sum_{i=1}^{n} |y(t)-y(t-1)|
{{< /katex >}}
- MASE = 2 MSPL(0.5)
- *Best quantile model has the lowest pinball loss*. The most important result
associated with the pinball loss function is that the lower the pinball loss,
the more accurate the quantile forecast.
{{< /tab >}}
{{< tab "references" >}}
- https://www.lokad.com/pinball-loss-function-definition
- https://www.kaggle.com/c/m5-forecasting-uncertainty/discussion/137098
{{< /tab >}}
{{< /tabs >}}

## Metrics for Prediction Intervals

Prediction intervals express the uncertainty in the forecasts. This is useful because it provides the user of the forecasts with *worst* and *best* case estimates and a sense of how depedenable the forecast it.

While we would like to estimate the actual distribution based on training as a surrogate we typically estimate a 95% prediction interval $[f_L(t),f_U(t)]$ such that $\text{Pr}[f_L(t)
\leq y(t) \leq f_U(t)]=0.95$.

MSPL values can be aggregated over different values of $p$. For example, for a a 95% prediction interval $p$ can be set to is set to $p_1=0.025, p_2=0.5, p_3=0.975$. The the aggregate SPL can be determined as below to measure the accuracy of the **prediction interval**.

### AgMSPL
*Aggregate Mean Scaled Pinball Loss*

{{< tabs "AgMSPL" >}}
{{< tab "AgMSPL" >}}
{{< katex >}}
\text{AgMSPL}=\frac{1}{3} \sum_{i=1}^3 \text{MSPL}(p_i)
{{< /katex >}}
{{< /tab >}}
{{< tab "notes" >}}
{{< /tab >}}
{{< tab "references" >}}
- https://www.kaggle.com/c/m5-forecasting-uncertainty
{{< /tab >}}
{{< /tabs >}}

### CRPS
*Continuous Ranked Probability Score*

{{< tabs "CRPS" >}}
{{< tab "CRPS" >}}
{{< katex >}}
\text{CRPS}\left(F_t, y(t)\right)= \int_{-\infty}^{\infty} \left( F_t(x)- \unicode{x1D7D9}_{ (x-y(t))} \right) \text{d}x
\quad \text{where } \unicode{x1D7D9} \text{ is the indicator function}.
{{< /katex >}}
{{< /tab >}}
{{< tab "notes" >}}
- The CRPS generalizes the MAE; in fact, it reduces to the MAE if the forecast is deterministic.

- CRPS can be defined equivalently as follows:
{{< katex >}}
\text{CRPS}\left(F_t, y(t)\right)= \int_{0}^{1} pl(t,p) \text{d}p
{{< /katex >}}
Discretization of this integral, e.g., replacing the integral by a sum over quantiles
$q = 0.01, \dots, 0.99$, enables us to avoid the complications with original formula.
{{< /tab >}}
{{< tab "references" >}}
https://pypi.org/project/properscoring/
{{< /tab >}}
{{< /tabs >}}

## Metrics for multiple time series

When forecasting multiple time series an agggregate metric can be computed via weighted combinaton of the corresponding metric ($\text{metric}_j$) for each time series $y_j(t)$.
{{< katex >}}
\text{W-metric}= \frac{\sum_{j=1}^K w_j \text{metric}_j}{\sum_{j=1}^K w_j}
{{< /katex >}}
For example,
{{< katex >}}
\text{W-MASE}= \frac{\sum_{j=1}^K w_j \text{MASE}_j}{\sum_{j=1}^K w_j}
{{< /katex >}}
The weights $w_j$ represent a certain relative importance of the $j^{th}$ time series $y_j(t)$.

- The simplest weights are $w_j=1/K$ which gives a uniform weight to all the time series.
- For example the M5 competition used the sum of units sold in the last 28 observations multiplied by their respective price (*cumulative actual
dollar sales*). It is an objective proxy of value of an unit for the
company in monetary terms, hence it can be used as weight.


## Metrics for hierarchical forecasting

<img src="/img/product_location_hierarchy.jpg" alt="product_location_hierarchy"/>

Node level metrics The forecast accuracy metric can be computed at any node in the specified hierarchy. However there are differnt ways to generate an accuracy metric at a node.

1. **aggregate-metrics** For any node compute the metric for all the children and then aggregate the metric via suitable weighted aggregation method as earlier.
1. **aggregate-forecasts** Forecast for all the children at a node, aggregate the forecasts and then evaluate the metric.
1. **aggregate-data** Aggregate the data fro all the children at a node,  forecast and then evaluate the metric.

Aggregate across all nodes at a level

Aggregate across all levels

## Metrics for new products

- Cold start evaluation
- Forecast adjustments after $k$ time periods

## References

https://otexts.com/fpp2/accuracy.html

https://www.relexsolutions.com/resources/measuring-forecast-accuracy/

Rob J. Hyndman and Anne B. Koehler, [Another look at measures of forecast accuracy](https://github.ibm.com/retail-supply-chain/planning/files/582845/Another.look.at.measures.of.forecast.accuracy.pdf), International Journal of Forecasting, Volume 22, Issue 4, 2006.

Rob J Hyndman, [Another look at forecast-accuracy metrics for intermittent demand](https://github.ibm.com/retail-supply-chain/planning/files/582847/Metrics.for.intermitted.demand.pdf). Chapter 3.4, pages 204-211, in "Business Forecasting: Practical Problems and Solutions", John Wiley & Sons, 2015.

Sungil Kim and Heeyoung Kim, [A new metric of absolute percentage error for intermittent demand forecasts](https://github.ibm.com/retail-supply-chain/planning/files/582846/A.new.metric.of.absolute.percentage.error.for.intermittent.demand.forecasts.pdf), International Journal of Forecasting, Volume 32, Issue 3, 2016.

Makridakis, S., Spiliotis, E. and Assimakopoulos, V, The M4 Competition: 100,000 time series and 61 forecasting methods. International Journal of Forecasting, 36, 54–74, 2000.

Spyros Makridakis and Michèle Hibon, [The M3-Competition: results, conclusions and implications](https://github.ibm.com/retail-supply-chain/planning/files/582844/M3_competition_results_conclusions_implications.pdf), International Journal of Forecasting, Volume 16, Issue 4, 2000.

Fotios Petropoulos, Xun Wang and Stephen M. Disney. [The inventory performance of forecasting methods: Evidence from the M3 competition data](https://github.ibm.com/retail-supply-chain/planning/files/582842/The.inventory.performance.of.forecasting.methods-.Evidence.from.the.M3.competition.data.pdf),International Journal of Forecasting, Volume 35, Issue 1, 2019.

Gneiting, Tilmann, Fadoua Balabdaoui, and Adrian E. Raftery. [Probabilistic forecasts, calibration and sharpness](https://github.ibm.com/retail-supply-chain/planning/files/593835/Recent.advances.in.electricity.price.forecasting-.A.review.of.probabilistic.forecasting.pdf) Journal of the Royal Statistical Society: Series B (Statistical Methodology) 69.2 (2007): 243-268.