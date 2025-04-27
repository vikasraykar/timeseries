---
bookShowToc: true
weight: 2
title: "TimeSHAP"
---

# TimeSHAP

{{< button href="https://github.ibm.com/srom/aix360ts/tree/master/aix360ts/forecasting/explainers/_TimeSHAP" >}}code{{< /button >}}
{{< button href="https://pages.github.ibm.com/srom/aix360ts/aix360ts.forecasting.explainers.html" >}}API docs{{< /button >}}
{{< button href="https://github.ibm.com/srom/aix360ts/blob/master/examples/forecasting/explainers/TimeSHAP.py" >}}example{{< /button >}}
{{< button href="https://pages.github.ibm.com/srom/aix360ts/notebooks/TimeSHAP.html" >}}notebook{{< /button >}}

{{< hint info >}}
TimeSHAP is a feature based post-hoc black box explainer to explain the forecast
of any univariate time series forecaster using tree-based regressors to build the
surrogate model and SHAP(SHapley Additive exPlanations) values for the explanations.
{{< /hint >}}

For any given univariate time series we first generate a sequence of backtested
historical forecasts using an (expanding window) splitter. Using the splitter we
split the time series into a sequence of train an test splits. The expanding window
splitter uses more and more training data, while keeping the test window size fixed
to the forecast horizon.  The method is model agnostic in the sense that it needs
access to only the ``fit`` and ``predict`` methods of the forecaster. The forecaster
is trained on the train split and evaluated on the test split and all the test split
predictions are concatenated to get the backtested historical forecasts for each
step of the forecast horizon. We the construct a surrogate time series forecasting
task by reducing it to a standard supervised regression problem. For each time point
we generate a set of interpretable features (lag features, seasonal features,
date time encodings etc) based on which we need to predict the backtested forecasted
time series values. The surrogate model is fitted using tree-based regressors like
XGBoost, CatBoost,LightGBM etc. Explanation is in now in terms of features that
encode the time series. We mainly rely on the TreeSHAP
(SHapley Additive exPlanations) algorithm to explain the output of
ensemble tree models. In order to improve the sensitivity we extend the above
approach by aggregating multiple explanations from bootstrapped versions of
the time series.

## Time series forecasting

A univariate time series is a series with a single time-dependent variable. Let {{< katex >}}f(t):\mathbb{Z}\to\mathbb{R}^1{{< /katex >}} represent a latent **univariate time series** for any discrete time index {{< katex >}}t \in \mathbb{Z}{{< /katex >}}. We observe a sequence of historical *noisy* (and potentially missing) values {{< katex >}}y(t){{< /katex >}} for {{< katex >}}t \in [1,\dots,T]{{< /katex >}}  such that in expectation {{< katex >}}\mathbb{E}[y(t)]=f(t){{< /katex >}}. For example, in the retail domain {{< katex >}}y(t){{< /katex >}} could represent the daily sales of a product and {{< katex >}}f(t){{< /katex >}} the true latent demand for the product.

The task of **time series forecasting** is to estimate {{< katex >}}f(t){{< /katex >}} for all {{< katex >}}t >T{{< /katex >}} based on the observed historical time series {{< katex >}}y(t){{< /katex >}} for {{< katex >}}t \in [1,\dots,T]{{< /katex >}}. When we talk about the *forecast*, we usually mean the average value of the forecast distribution also known as the **point forecast**, {{< katex >}}f(t){{< /katex >}}, which is the **mean** ({{< katex >}}\mathbb{E}[y(t)]{{< /katex >}}) of the forecast distribution. The time series forecast is typically done for a fixed number or periods in the future, refered to as the **forecast horizon**, {{< katex >}}h{{< /katex >}}. Let {{< katex >}}\hat{f}(T+h){{< /katex >}} for {{< katex >}}h \in [1,\dots,H]{{< /katex >}} be the forecasted time series for the forecast horizon {{< katex >}}h{{< /katex >}} based on the historical observed time series {{< katex >}}y(1),...,y(T){{< /katex >}}.
{{< katex display >}}
\textbf{forecaster model}\:\:\hat{f}(T+h|y(1),...,y(T))\:\text{for}\:h=1,...,H
{{< /katex >}}
> Using the language of supervised learning, {{< katex >}}y(1),...,y(T){{< /katex >}} is the training data of {{< katex >}}T{{< /katex >}} samples based on which we learn/train a forecaster model {{< katex >}}\hat{f}{{< /katex >}}. The trained model {{< katex >}}\hat{f}{{< /katex >}} is then used to predict/forecast on the test set of {{< katex >}}H{{< /katex >}} time points in the future. Unlike supervised learing where the model is usually fit once in time series forecasting for many algorithms we will have to continously fit the model before forecasting as more recent data arrives.

![dataset](/img/timeshap_dataset.jpg)

## Explainability for forecasting

**Explainability** is the degree to which a human can understand the cause of a decision (or prediction) made by a prediction model [^molnar_2018][^miller_2017][^kim_2017].Various notions of explainability (local and global explanations) and explainer algorithms has been studied in classical supervised learning paradigms like classification and regression.

### Scope of explanations

An explanation is the answer to either a **why**-question or a **what if**-scenario. We define the following three notions of **explanations** in the context of time series forecasting.
1. A **local explantion** explains the forecast made by a forecaster at a certain point in time.
    - *Why is the forecasted sales on July 22, 2019 much higher than the average sales?*
    - *Will the forecast increase if I increase the offered discount?*
1. A **global explanation**  explains the forecaster trained on the historical time series.
    - *What are the most important attributes the forecaster relies on to make the forecast?*
    - *What is the impact of diccount on the sales forecast?*
1. A **semi-local explanation** explains the overall forecast made by a forecaster in a certain time interval. In general this returns **one** (semi-local) explanation aggregated over all the multiple time steps in the forecast horizon.
    - *Why is the forecasted sales over the next 4 weeks much higher?*

### Type of explanations

In the context of time series the explanations can boradly of the following two types.

1. **Features based** Explanation is in terms of features that encode the time series (lag features, date encodings etc.) and external regressors.
1. **Instance based** Explanation is in terms of the importance or certain time points in the historical time series.

### Type of explainers

An **explainer** is an algorithm the generates local, semi-local and global explanations for a forecasting algorithm. We can boradly categorize explainers into the following 3 types.

1. **Directly interpretable explainers**  Some time series forecasting algorithms are inherently interpretable by desgin. For example, for an autoregressive model of order {{< katex >}}p{{< /katex >}} ({{< katex >}}AR(p){{< /katex >}}) the coefficients correponds to the importance associated with the past {{< katex >}}p{{< /katex >}} values of the time series. Another example is, Prophet which uses a decomposable additive time series model with three main model components: trend, seasonality, and holidays. The explanation is directly in terms of these 3 components.
1. **White-box explainers** Though not directly interpretable, some forecasting algorithms can be explained if we have access to the internals of the corresponding forecasting algorithm. For example, for deep neural forecasting algorithms if we have access to the model we can compute saliency maps and activations to explain the inner working of the model. Tree based regression ensembles like XGBoost, CatBoost and LightGBM gprovide global explanations in terms of the feature importance scores.
1. **Black-box explainers**  Such explainers are model agnostic and generally require access to model's `predict` (and sometimes `fit`) functions. Given a source black box model, such explainers generally train a surrogate model that is explainable.

## TimeSHAP

{{< hint info >}}
TimeSHAP is a feature based post-hoc black box explainer to explain the forecast
of any univariate time series forecaster using tree-based regressors to build the
surrogate model and SHAP(SHapley Additive exPlanations) values for the explanations.
{{< /hint >}}

### Surrogate model

The method is model agnostic in the sense that it needs access to only the `fit` and `predict` methods of the forecaster.
At any time {{< katex >}}t{{< /katex >}} using all the historical data available so far (that its, {{< katex >}}y(1),...,y(t){{< /katex >}})
we can train the forecaster model {{< katex >}}f{{< /katex >}} using the `fit` method. Once the forecaster is trained we can
then generate the corresponding {{< katex >}}H{{< /katex >}}(forecast horizon) forecasts
{{< katex display >}}
f(t+h|t)=f(t+h|y(1),...,y(t))\:\text{for}\:h=1,...,H,
{{< /katex >}}
time steps ahead using the `predict` method. Our goal now is to learn a surrogate model(s) {{< katex >}}g{{< /katex >}} to **predict the forecasts from the forecaster**.
{{< katex display >}}
g(t+h|t)=g(t+h|y(1),...,y(t))\:\text{for}\:h=1,...,H,
{{< /katex >}}
While the original forecaster learns to predict {{< katex >}}y(t+h){{< /katex >}} based on {{< katex >}}y(1),...,y(t){{< /katex >}} the surrogate model is trained to predict the forecasts {{< katex >}}f(t+h){{< /katex >}} made by the forecaster. Essentially we want to mimic the forecaster using a surrogate model. We choose the surrogate model that can be easily interpreted.

![recursive](/img/timeshap_surrogate.jpg)

### Backtested historical forecasts

In order to generate data to train the surrogate model we use backtesting. For any given univariate time series we first generate a sequence of backtested historical forecasts using an expanding window splitter. Using the splitter we
split the time series into a sequence of train and test splits. The expanding window
splitter uses more and more training data, while keeping the test window size fixed
to the forecast horizon. The forecaster
is trained on the train split and evaluated on the test split and all the test split
predictions are concatenated to get the backtested historical forecasts for each
step of the forecast horizon.

![backtest](/img/timeshap_backtest.jpg)

{{< hint warning >}}
This is one of the most computationally expensive steps since for a time series of length {{< katex >}}n{{< /katex >}} we will have to potentially invoke `fit` and `predict` roughly {{< katex >}}O(n){{< /katex >}} times to generate the backtested forecasts.
- This can be effficiently parallelized since each task can be executed independently. In our implementation we use [ray](https://github.com/ray-project/ray) to parallelize this.
-  For computationally expensive `fit` models it may be cheaper to forecast without full refitting. However we may need to pass the context/historical time series so far to do the forecast. Some forecasters have a light weight `update` method has the same signature as train but does not do refitting and only does minimal context updates to make the prediction.
- Certain forecasting algorithms train one large model based on large number of multiple univariate time series. In such scenarios we completely avoid refitting the model and rely only on the `predict` method to generate the in-sample forecasts and build a surrogate model. While this avoids backtesting and can potentially overfit, since the model is trained or large number of time series this may be fine.
{{< /hint >}}

### Regressor reduction

We now have an original time series and the corresponding backtested forecast time series
for each step of the forecast horizon. The goal of the surrogate model is to learn to
predict the backtested forecast time series based on on the original time series. We
construct a surrogate time series forecasting task by reducing it to a standard
supervised regression problem.

A common machine learning approach to time series forecasting is to reduce it to a standard **supervised regression** problem. A standard supervised regression task takes as input a {{< katex >}}d{{< /katex >}}-dimensional feature vector {{< katex >}}\mathbf{x}\in\mathbb{R}^d{{< /katex >}} and predicts a scalar {{< katex >}}y \in \mathbb{R}{{< /katex >}}. The regressor {{< katex >}}y = f(\mathbf{x}){{< /katex >}} is learnt based on a labelled training dataset {{< katex >}}\left(\mathbf{x}_i,y_i\right){{< /katex >}}, for {{< katex >}}i=1,..,n{{< /katex >}} samples. However there is no direct concept of input features ({{< katex >}}\mathbf{x}{{< /katex >}}) and output target ({{< katex >}}y{{< /katex >}}) for a time series. Instead, we must choose the backtested time series forecast values as the variable to be predicted and use various time series feature engineering techniques (like lag features, date time encodings etc.) to construct the features from the original time series.

### Interpretable features

Essentially, for each time point {{< katex >}}t{{< /katex >}} we generate a feature vector {{< katex >}}\mathbf{x}(t) \in\mathbb{R}^d{{< /katex >}} based on the original time series values  observed so far based on which we need to predict the backtested forecast time series for each step in the forecast horizon {{< katex >}}y(t) \in\mathbb{R}{{< /katex >}}. The table below is a list of features we use. See [here]({{< relref "features.md" >}}) for more details. We generally like the features to be interpretable in the sense that the end user of the explanation should be able to comprehend the meaning of these features.

{{< hint warning >}}
The feature vector {{< katex >}}\mathbf{x}(t){{< /katex >}} needs to be constructed only based on the time step {{< katex >}}t{{< /katex >}} and the historical values of the time series {{< katex >}}y(1),...,y(t-1){{< /katex >}} and should not use the current time series value {{< katex >}}y(t){{< /katex >}}.
{{< /hint >}}

| feature name                     | description                                                                           |
|:--------------------|:--------------------------------------------------------------------------------------|
| sales(t-3)          | The value of the time series (sales) at the (t-3) previous time step.                 |
| sales(t-2)          | The value of the time series (sales) at the (t-2) previous time step.                 |
| sales(t-1)          | The value of the time series (sales) at the (t-1) previous time step.                 |
| sales(t-2*365)      | The value of the time series (sales) at the (t-2*365) previous time step.             |
| sales(t-1*365)      | The value of the time series (sales) at the (t-1*365) previous time step.             |
| sales_min(t-1,t-3)  | The min of the past 3 values in the sales time series.                                |
| sales_mean(t-1,t-3) | The mean of the past 3 values in the sales time series.                               |
| sales_max(t-1,t-3)  | The max of the past 3 values in the sales time series.                                |
| sales_min(0,t-1)    | The min of all the values so far in the sales time series.                            |
| sales_mean(0,t-1)   | The mean of all the values so far in the sales time series.                           |
| sales_max(0,t-1)    | The max of all the values so far in the sales time series.                            |
| year                | The year.                                                                             |
| month               | The month name of the year from January to December.                                  |
| day_of_year         | The ordinal day of the year from 1 to 365.                                            |
| day_of_month        | The ordinal day of the month from 1 to 31.                                            |
| week_of_year        | The ordinal week of the year from 1 to 52.                                            |
| week_of_month       | The ordinal week of the month from 1 to 4.                                            |
| day_of_week         | The day of the week from Monday to Sunday.                                            |
| is_weekend          | Indicates whether the date is a weekend or not.                                       |
| quarter             | The ordinal quarter of the date from 1 to 4.                                          |
| season              | The season Spring/Summer/Fall/Winter.                                                 |
| fashion_season      | The fashion season Spring/Summer (January to June) or Fall/Winter (July to December). |
| is_month_start      | Indicates whether the date is the first day of the month.                             |
| is_month_end        | Indicates whether the date is the last day of the month.                              |
| is_quarter_start    | Indicates whether the date is the first day of the quarter.                           |
| is_quarter_end      | Indicates whether the date is the last day of the quarter.                            |
| is_year_start       | Indicates whether the date is the first day of the year.                              |
| is_year_end         | Indicates whether the date is the last day of the year.                               |
| is_leap_year        | Indicates whether the date belongs to a leap year.                                    |
| hour                | The hours of the day.                                                                 |
| minute              | The minutes of the hour.                                                              |
| second              | The seconds of the minute.                                                            |
| holiday-IN          | Indicates whether the date is a IN holiday or not.                                    |
| t                   | Feature to model simple polynomial (of degree 1) trend in sales.                      |
| t^2                 | Feature to model simple polynomial (of degree 2) trend in sales.                      |


{{< hint info >}}
**External regressors** Classical time series forecasting algorithms esentially learn a model to forecast based on the historical values of the time series. In many domains, the value of the time series depends on several external time series which we refer to as related external regressors. For example in the retail domain, the sales is potentially influence by discount, promotion, events, weather etc. Each external regressor in itself is a time series and some methods allow to explicity include external regressors to improve forecasting. If external regressors are avaiable we can also encode them as interpretable features using similar features as above.

Typically some exeternal regressors can be **forwad looking**. For example, the discount that will be used iin the future is typically planned by the retailer in advance.
{{< /hint >}}

| feature name                     | description                                                                           |
|:--------------------|:--------------------------------------------------------------------------------------|
| discount(t-3)          | The value of the time series (discount) at the (t-3) previous time step.                 |
| discount(t-2)          | The value of the time series (discount) at the (t-2) previous time step.                 |
| discount(t-1)          | The value of the time series (discount) at the (t-1) previous time step.                 |
| discount(t)          | The value of the time series (discount) at the current time step.                 |

### Multi-step forecasting

In the last section we described some of the commonly used methods to transform the original time series to a set of interpretable features.
Recall that we have {{< katex >}}H{{< /katex >}} backtested time series forecats which are to be used as targets to learn the surrogate regressor model. Here we will desxribe common strategies[^reduction-paper] that can be used used for multi-step forecasting to regression reduction.

#### Recursive

A **single regressor model** is fit for one-step-ahead forecast horizon and then called recursively to predict multiple steps ahead. Let {{< katex >}}\mathcal{G}(y(1),...,y(t)){{< /katex >}} be the one-step ahead surrogate forecaster that has been learnt based on the training data, where the forecaster predicts the one-step ahead forecast using features based on the time series values till {{< katex >}}t{{< /katex >}}, that is, {{< katex >}}y(1),...,y(t){{< /katex >}}. The forecasts from the surrogate models are made recusively as follows,
{{< katex display>}}
g(t+1|t) = \mathcal{G}(y(1),...,y(t)).
{{< /katex >}}
For {{< katex >}}h=2,3,...,H{{< /katex >}}
{{< katex display>}}
g(t+h|t) = \mathcal{G}(y(1),...,y(t),g(t+1|t),...,g(t+h-1|t)).
{{< /katex >}}

For example,

horizon | forecast | strategy
:--- | :--- | :---
1 | {{< katex >}}
g(t+1|t)
{{< /katex >}} | {{< katex >}} \mathcal{G}(y(1),...,y(t))
{{< /katex >}}
2 | {{< katex >}}
g(t+2|t)
{{< /katex >}} | {{< katex >}} \mathcal{G}(y(1),...,y(t),g(t+1|t))
{{< /katex >}}
h | {{< katex >}}
g(t+h|t)
{{< /katex >}} | {{< katex >}} \mathcal{G}(y(1),...,y(t),g(t+1|t),...,g(t+h-1|t))
{{< /katex >}}

{{< hint warning >}}
A well-known drawback of the recursive method is its sensitivity to the estimation error, since estimated values, instead of actual ones, are more and more used when we get further in the future forecasts.
{{< /hint >}}

![recursive](/img/timeshap_recursive.jpg)

#### Direct

A **separate regressor model** is fit for each step ahead in the forecast horizon and then independently applied to predict multiple steps ahead.  Let {{< katex >}}\mathcal{G}_h(y(1),...,y(t)){{< /katex >}} be the h-step ahead surrogate forecaster that has been learnt based on the training data, where the forecaster predicts the h-step ahead forecast using features based on the time series values till {{< katex >}}t{{< /katex >}}, that is, {{< katex >}}y(1),...,y(t){{< /katex >}}. The forecasts are made directly as follows,
{{< katex display>}}
g(t+h|t) = \mathcal{G}_h(y(1),...,y(t))\text{ for }h=1,2,...,H
{{< /katex >}}

For example,

horizon | forecast | strategy
:--- | :--- | :---
1 | {{< katex >}}
g(t+1|t)
{{< /katex >}} | {{< katex >}} \mathcal{G}_1(y(1),...,y(t))
{{< /katex >}}
2 | {{< katex >}}
g(t+2|t)
{{< /katex >}} | {{< katex >}} \mathcal{G}_2(y(1),...,y(t))
{{< /katex >}}
h | {{< katex >}}
g(t+h|t)
{{< /katex >}} | {{< katex >}} \mathcal{G}_h(y(1),...,y(t))
{{< /katex >}}

{{< hint warning >}}
Since the Direct strategy does not use any approximated values to compute the forecasts, it is not prone to any accumulation of errors. However since the models are learned independently no statistical dependencies between the predictions is considered. This strategy also demands a large computational time since the number of models to learn is equal to the size of the forecast horizon.
{{< /hint >}}

![direct](/img/timeshap_direct.jpg)

#### DirRec

The DirRec strategy  combines the architectures and the principles underlying the Direct and the Recursive strategies. DirRec computes the forecasts with different models for every horizon (like the Direct strategy) and, at each time step, it enlarges the set of inputs by adding variables corresponding to the forecasts of the previous step (like the Recursive strategy).

{{< katex display>}}
g(t+1|t) = \mathcal{G}_1(y(1),...,y(t))
{{< /katex >}}

For {{< katex >}}h=2,3,...,H{{< /katex >}}
{{< katex display>}}
g(t+h|t) = \mathcal{G}_h(y(1),...,y(t),g(t+1|t),...,g(t+h-1|t))
{{< /katex >}}

For example,

horizon | forecast | strategy
:--- | :--- | :---
1 | {{< katex >}}
g(t+1|t)
{{< /katex >}} | {{< katex >}} \mathcal{G}_1(y(1),...,y(t))
{{< /katex >}}
2 | {{< katex >}}
g(t+2|t)
{{< /katex >}} | {{< katex >}} \mathcal{G}_2(y(1),...,y(t),g(t+1|t))
{{< /katex >}}
h | {{< katex >}}
g(t+h|t)
{{< /katex >}} | {{< katex >}} \mathcal{G}_h(y(1),...,y(t),g(t+1|t),...,g(t+h-1|t))
{{< /katex >}}

### Tree ensemble regressors

So far, for each time point we generate a set of interpretable features (lag features, seasonal features,
date time encodings etc) based on which we need to predict the backtested forecasted
time series values. The surrogate model is then fitted using tree-based regressors like
XGBoost, CatBoost,LightGBM etc.

### SHapley Additive exPlanations

While in general we can use any regressor we prefer tree-based ensembles like [XGBoost](https://github.com/dmlc/xgboost), [CatBoost](https://github.com/catboost/catboost) and [LightGBM](https://github.com/microsoft/LightGBM) since they are resonably accurate and more importantly support fast explaninablity algorithms like [TreeSHAP](https://github.com/slundberg/shap) which we reply on for the various explanations. Explanation is in now in terms of features that
encode the time series. We mainly rely on the TreeSHAP
(SHapley Additive exPlanations) algorithm to explain the output of
ensemble tree models.  SHAP (SHapley Additive exPlanations) is a game theoretic approach to explain the
output of any machine learning model. It connects optimal credit allocation with
local explanations using the classic Shapley values from game theory and their
related extensions. [^shapgit][^treeshap]

## Global explanations

A **global explanation**  explains the forecaster trained on the historical time series. The explanation type can be one of following four types.

{{< hint info >}}
Note that the recurisve strategy has only one model, while the direct and the dirrec strategies have {{< katex >}}H{{< /katex >}} models corresponding to each step of the forecasting horizon. Hence there will be a possible seprate global explanation for each model based on the forecast horizon.
{{< /hint >}}

### SHAP feature importance

The importance score for each features based on the shap values. Specifically this is the mean absolute value of the shap values for each feature across the entire dataset. To get an overview of which features are most important for a model we can compute the SHAP values of every feature for every sample. We can then take the mean absolute value of the SHAP values for each feature to get a feature importance score for each feature.

![timeshap_global_shap_feature_importance](/img/timeshap_global_shap_feature_importance.jpg)

### Feature importance

The relative contribution of each feature to the model. A higher value of this score when compared to another feature implies it is more important for generating a forecast. Feature importance provides a score that indicates how useful or valuable each feature was in the construction of the boosted decision trees within the model. The more an attribute is used to make key decisions with decision trees, the higher its **relative importance**. Feature importance is calculated for a single decision tree by the amount that each attribute split point improves the performance measure, weighted by the number of observations the node is responsible for. The feature importances are then averaged across all of the the decision trees within the model. The **gain** is the most relevant attribute to interpret the relative importance of each feature. The **gain** implies the relative contribution of the corresponding feature to the model calculated by taking each feature’s contribution for each tree in the model. A higher value of this metric when compared to another feature implies it is more important for generating a prediction.

![timeshap_global_feature_importance](/img/timeshap_global_feature_importance.jpg)

### Partial dependence plot

The [partial dependence plot](https://christophm.github.io/interpretable-ml-book/pdp.html) (PDP) for a feature shows the marginal effect the feature has on the forecast (from the surrogate model). The PDP shows how the average prediction in your dataset changes when a particular feature is changed. The partial dependence function at a particular feature value represents the average prediction if we force all data points to assume that feature value.

{{< hint info >}}
The calculation for the partial dependence plots has a causal interpretation. One way to think about PDP is that it is an **intervention query**. We intervene on a feature and measure the changes in the predictions. In doing so, we analyze the causal relationship between the feature and the prediction.
{{< /hint >}}

![timeshap_global_pdp](/img/timeshap_global_pdp.jpg)

{{< hint warning >}}
The assumption of independence is the biggest issue with PDP plots. It is assumed that the feature(s) for which the partial dependence is computed are not correlated with other features.
{{< /hint >}}

### SHAP dependence plot

The shap dependence plot (SDP) for each feature shows the mean shap value for a particular features across the entire dataset. This shows how the model depends on the given feature, and is like a richer extenstion of the classical partial dependence plots.

![timeshap_global_sdp](/img/timeshap_global_sdp.jpg)

## Local explanations

A **local explantion** explains the forecast made by a forecaster at a certain point in time.

### SHAP explanation

[SHAP](https://github.com/slundberg/shap) (SHapley Additive exPlanations) is a unified approach to explain the output of any machine learning model. SHAP connects game theory with local explanations. While SHAP values can explain the output of any machine learning model, high-speed exact algorithms are available for tree ensemble methods

The SHAP explanation shows features contributing to push the forecasted sales from the base value (the average sales) to the forecaster model output. Features pushing the forecast higher are shown in blue and those pushing the forecast lower are in red.

![timeshap_local_shap](/img/timeshap_local_shap.jpg)

> This time instance (Mon Jul  1 00:00:00 2019) has a forecasted sales (5028.76), 2579.59 units higher than the average (2449.17) mainly because of the discount(t) (40.50) and sales(t-1) (4242.54) while sales(t-2*52) (4375.00) was trying to push it lower.

### Local partial dependence plot

The (local) PDP for a given feature shows how the forecast (from the surrogate model) varies as the feature value changes.

![timeshap_local_pdp](/img/timeshap_local_pdp.jpg)

### Local SHAP dependence plot

The (local) SDP for a given feature shows how the shap value varies as the feature value changes.

![timeshap_local_sdp](/img/timeshap_local_sdp.jpg)

## Semi-local explanations

A **semi-local explanation** explains the overall forecast made by a forecaster in a certain time interval. In general this returns **one** (semi-local) explanation aggregated over all the multiple time steps in the forecast horizon.

### SHAP feature importance

The importance score for each feature which is the corresponding shap value for that feature.

![timeshap_semilocal_shap](/img/timeshap_semilocal_shap.jpg)

### Partial dependence plot

The PDP for a given feature shows how the forecast (from the surrogate model) varies as the feature value changes.

![timeshap_semilocal_pdp](/img/timeshap_semilocal_pdp.jpg)

### SHAP dependence plot

The SDP for a given feature shows how the shap value varies as the feature value changes.

![timeshap_semilocal_sdp](/img/timeshap_semilocal_sdp.jpg)

## Explaining prediction intervals

The same setup can also be used to explain the width of the prediction interval. Instead of regressing on the mean forecast from the forecaster we can regress on the width of the prediction interval.

## Bootstrapped ensemble

In order to improve the sensitivity we extend the above
approach by aggregating multiple explanations from bootstrapped versions of
the time series.

## Examples

### Naive

The forecast is the value of the last observation.
{{< katex display >}}
f(t+h|t) = y(t),\text{ for }h=1,2,...
{{< /katex >}}


{{< tabs "Naive" >}}

{{% tab "aix360ts" %}}
```python
from aix360ts.forecasting.forecasters import Naive
from aix360ts.forecasting.explainers import TimeSHAP

forecaster = Naive()
explainer = TimeSHAP(forecaster=forecaster)
```
{{% /tab %}}

{{% tab "forecaster" %}}
![timeshap_naive_forecast](/img/timeshap_naive_forecast.jpg)
{{% /tab %}}

{{% tab "global - SHAP" %}}
![timeshap_naive_global_shap](/img/timeshap_naive_global_shap.jpg)
{{% /tab %}}

{{% tab "global - PDP" %}}
![timeshap_naive_global_pdp](/img/timeshap_naive_global_pdp.jpg)
{{% /tab %}}

{{% tab "local - SHAP" %}}
![timeshap_naive_local_shap](/img/timeshap_naive_local_shap.jpg)
{{% /tab %}}

{{% tab "local - PDP" %}}
![timeshap_naive_local_pdp](/img/timeshap_naive_local_pdp.jpg)
{{% /tab %}}

{{< /tabs >}}

### SeasonalNaive

The forecast is the value of the last observation from the same season of the year. For example, with monthly data, the forecast for all future February values is equal to the last observed February value.

{{< tabs "SeasonalNaive" >}}

{{% tab "aix360ts" %}}
```python
from aix360ts.forecasting.forecasters import SeasonalNaive
from aix360ts.forecasting.explainers import TimeSHAP

forecaster = SeasonalNaive(m=52)
explainer = TimeSHAP(forecaster=forecaster)
```
{{% /tab %}}

{{% tab "forecaster" %}}
![timeshap_seasonalnaive_forecast](/img/timeshap_seasonalnaive_forecast.jpg)
{{% /tab %}}

{{% tab "global - SHAP" %}}
![timeshap_seasonalnaive_global_shap](/img/timeshap_seasonalnaive_global_shap.jpg)
{{% /tab %}}

{{% tab "global - PDP" %}}
![timeshap_seasonalnaive_global_pdp](/img/timeshap_seasonalnaive_global_pdp.jpg)
{{% /tab %}}

{{% tab "local - SHAP" %}}
![timeshap_seasonalnaive_local_shap](/img/timeshap_seasonalnaive_local_shap.jpg)
{{% /tab %}}

{{% tab "local - PDP" %}}
![timeshap_seasonalnaive_local_pdp](/img/timeshap_seasonalnaive_local_pdp.jpg)
{{% /tab %}}

{{< /tabs >}}


### MovingAverage

A moving average forecast of order {{< katex >}}k{{< /katex >}}, or, {{< katex >}}MA(k){{< /katex >}}, is the mean of the last {{< katex >}}k{{< /katex >}} observations of the time series.

{{< tabs "MovingAverage" >}}
{{% tab "aix360ts" %}}
```python
from aix360ts.forecasting.forecasters import MovingAverage
from aix360ts.forecasting.explainers import TimeSHAP

forecaster = MovingAverage(k=6)
explainer = TimeSHAP(forecaster=forecaster)
```
{{% /tab %}}

{{% tab "forecaster" %}}
![timeshap_movingaverage_forecast](/img/timeshap_movingaverage_forecast.jpg)
{{% /tab %}}

{{% tab "global - SHAP" %}}
![timeshap_movingaverage_global_shap](/img/timeshap_movingaverage_global_shap.jpg)
{{% /tab %}}

{{% tab "global - PDP" %}}
![timeshap_movingaverage_global_pdp](/img/timeshap_movingaverage_global_pdp.jpg)
{{% /tab %}}

{{% tab "local - SHAP" %}}
![timeshap_movingaverage_local_shap](/img/timeshap_movingaverage_local_shap.jpg)
{{% /tab %}}

{{% tab "local - PDP" %}}
![timeshap_movingaverage_local_pdp](/img/timeshap_movingaverage_local_pdp.jpg)
{{% /tab %}}

{{< /tabs >}}

### Simple Exponential Smoothing

The forecast is the exponentially weighted average of its past values. The forecast can also be interpreted as a weighted average between the most recent observation and the previous forecast.

{{< tabs "SES" >}}
{{% tab "aix360ts" %}}
```python
from aix360ts.forecasting.forecasters import SES
from aix360ts.forecasting.explainers import TimeSHAP

forecaster = SES(alpha=0.5)
explainer = TimeSHAP(forecaster=forecaster)
```
{{% /tab %}}

{{% tab "forecaster" %}}
![timeshap_ses_forecast](/img/timeshap_ses_forecast.jpg)
{{% /tab %}}

{{% tab "global - SHAP" %}}
![timeshap_ses_global_shap](/img/timeshap_ses_global_shap.jpg)
{{% /tab %}}

{{% tab "global - PDP" %}}
![timeshap_ses_global_pdp](/img/timeshap_ses_global_pdp.jpg)
{{% /tab %}}

{{% tab "local - SHAP" %}}
![timeshap_ses_local_shap](/img/timeshap_ses_local_shap.jpg)
{{% /tab %}}

{{% tab "local - PDP" %}}
![timeshap_ses_local_pdp](/img/timeshap_ses_local_pdp.jpg)
{{% /tab %}}

{{< /tabs >}}

### Prophet

[Prophet](https://github.com/facebook/prophet) is a procedure for forecasting time series data based on an additive model where non-linear trends are fit with yearly, weekly, and daily seasonality, plus holiday effects.

{{< tabs "Prophet" >}}
{{% tab "aix360ts" %}}
```python
from aix360ts.forecasting.forecasters import Prophet
from aix360ts.forecasting.explainers import TimeSHAP

forecaster = Prophet()
explainer = TimeSHAP(forecaster=forecaster)
```
{{% /tab %}}

{{% tab "forecaster" %}}
![timeshap_prophet_forecast](/img/timeshap_prophet_forecast.jpg)
{{% /tab %}}

{{% tab "global - SHAP" %}}
![timeshap_prophet_global_shap](/img/timeshap_prophet_global_shap.jpg)
{{% /tab %}}

{{% tab "global - PDP" %}}
![timeshap_prophet_global_pdp](/img/timeshap_prophet_global_pdp.jpg)
{{% /tab %}}

{{% tab "local - SHAP" %}}
![timeshap_prophet_local_shap](/img/timeshap_prophet_local_shap.jpg)
{{% /tab %}}

{{% tab "local - PDP" %}}
![timeshap_prophet_local_pdp](/img/timeshap_prophet_local_pdp.jpg)
{{% /tab %}}

{{< /tabs >}}

### XGBoost

Forecasting to Regression reduction using XGBoost.

{{< tabs "XGBoost" >}}
{{% tab "aix360ts" %}}
```python
from aix360ts.forecasting.forecasters import RegressorReduction
from aix360ts.forecasting.explainers import TimeSHAP

forecaster = RegressorReduction()
explainer = TimeSHAP(forecaster=forecaster)
```
{{% /tab %}}

{{% tab "forecaster" %}}
![timeshap_naive_forecast](/img/timeshap_xgboost_forecast.jpg)
{{% /tab %}}

{{% tab "global - SHAP" %}}
![timeshap_xgboost_global_shap](/img/timeshap_xgboost_global_shap.jpg)
{{% /tab %}}

{{% tab "global - PDP" %}}
![timeshap_xgboost_global_pdp](/img/timeshap_xgboost_global_pdp.jpg)
{{% /tab %}}

{{% tab "local - SHAP" %}}
![timeshap_xgboost_local_shap](/img/timeshap_xgboost_local_shap.jpg)
{{% /tab %}}

{{% tab "local - PDP" %}}
![timeshap_xgboost_local_pdp](/img/timeshap_xgboost_local_pdp.jpg)
{{% /tab %}}

{{< /tabs >}}

## References

[^shapgit]: https://github.com/slundberg/shap

[^ray]: https://github.com/ray-project/ray

[^treeshap]: Lundberg, S.M., Erion, G., Chen, H. et al. [From local explanations to global understanding with explainable AI for trees]( https://doi.org/10.1038/s42256-019-0138-9). Nat Mach Intell 2, 56–67 (2020).

[^molnar_2018]: [Interpretable Machine Learning: A Guide for Making Black Box Models Explainable](https://christophm.github.io/interpretable-ml-book/), Christoph Molnar

[^reduction-paper]: Bontempi G., Ben Taieb S., Le Borgne YA. (2013) [Machine Learning Strategies for Time Series Forecasting](https://doi.org/10.1007/978-3-642-36318-4_3). In: Aufaure MA., Zimányi E. (eds) Business Intelligence. eBISS 2012. Lecture Notes in Business Information Processing, vol 138. Springer, Berlin, Heidelberg.

[^miller_2017]: [Explanation in Artificial Intelligence: Insights from the Social Sciences](https://arxiv.org/abs/1706.07269), Tim Miller

[^kim_2017]: [Towards A Rigorous Science of Interpretable Machine Learning](https://arxiv.org/abs/1702.08608), Finale Doshi-Velez, Been Kim
