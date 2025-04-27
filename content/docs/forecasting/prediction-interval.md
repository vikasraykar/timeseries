---
bookShowToc: true
weight: 104
title: "Prediction interval"
---

# Prediction interval

*Prediction intervals express the uncertainty in the forecasts.*

Prediction intervals express the uncertainty in the forecasts. This is useful because it provides the user of the forecasts with *worst* and *best* case estimates and a sense of how depedenable the forecast it. A forecast should be accompanied by a prediction interval giving a range of values the random variable could take with relatively high probability.  The value of prediction intervals is that they express the uncertainty in the forecasts. If we only produce point forecasts, there is no way of telling how accurate the forecasts are. However, if we also produce prediction intervals, then it is clear how much uncertainty is associated with each forecast. For this reason, point forecasts can be of almost no value without the accompanying prediction intervals.

## Forecast distribution

In general the value of a time series at time $t$, $y(t)$, is a random variable with it own distribution. In time series forecasting, we call this the **forecast distribution**. This is essentially the set of values that this random variable $y(t)$ could take, along with their relative probabilities.

{{< katex >}}
y(t)\quad\text{is a random variable with its own forecast distribution.}
{{< /katex >}}

## Point forecast

When we talk about the *forecast*, we usually mean the average value of the forecast distribution also known as the **point forecast**, $f(t)$, which is the **mean** $\mathbb{E}[y(t)]$ of the forecast distribution.

{{< katex >}}
f(t) = \mathbb{E}[y(t)]
{{< /katex >}}

## Quantile forecast

A **quantile forecast** $f_p(t)$ is an estimation of the $p^{th}$ **quantile** ($p \in [0,1]$) of the distribution. The $p^{th}$ quantile of $y(t)$ is defined as
{{< katex >}}
f_p(t) = \mathbb{Q}_p[y(t)] = \left\{x : \text{Pr}[y(t) \leq x] = p \right\}\quad p \in [0,1]
{{< /katex >}}

## Prediction interval

A central **prediction interval** of confidence level $\alpha \in [0,1]$ is specified as range of values $[f_L(t),f_U(t)]$ where $f_L(t)$ and $f_U(t)$ are the lower and upper quantiles repectively, such, that
{{< katex >}}
\text{Pr}[f_L(t) \leq y(t) \leq f_U(t)] = 1-\alpha
{{< /katex >}}
For example, a 95% prediction interval contains a range of values which should include the actual future value with probability 95%.


{{< hint warning >}}
Prediction and confidence intervals are often confused with each other. However, they are not quite the same thing. A confidence interval is a range of values associated with a population parameter. For example, the mean of a population. A prediction interval is where you expect a future value to fall.
{{< /hint >}}

## Normally distributed uncorrelated residuals

If we assume that forecast errors (**residuals**) are normally distributed, an $\alpha$ confidence level prediction interval for the $h$-step forecast $f(t+h|t)$ given historical time series till $t$, is given by,
{{< katex >}}
\left[f(t+h|t) - z \sigma_h,f(t+h|t) + z \sigma_h\right],
{{< /katex >}}
where $\sigma_h$ is an estimate of the standard deviation of the $h$-step forecast distribution and $z$ is the $1-(\alpha/2)$ quantile of the standard normal distribution. For example, the 95% prediction interval ($\alpha=0.05$) is given by
{{< katex >}}
f(t+h|t) \pm 1.96 \sigma_h.
{{< /katex >}}

### One-step prediction intervals

When forecasting one step ahead, the standard deviation of the forecast distribution is almost the same as the standard deviation of the residuals. The standard deviation of the forecast distribution is slightly larger than the residual standard deviation, although this difference is often ignored.

###  Multi-step prediction intervals

A common feature of prediction intervals is that they increase in length as the forecast horizon increases. The further ahead we forecast, the more uncertainty is associated with the forecast, and thus the wider the prediction intervals. That is, $\sigma_h$ usually increases with $h$ (although there are some non-linear forecasting methods that do not have this property).

For one-step forecasts ($h=1$), the residual standard deviation provides a good estimate of the forecast standard deviation $\sigma_1$. For multi-step forecasts, a more complicated method of calculation is required. These calculations assume that the residuals are uncorrelated.

- For some simple baselines, it is possible to mathematically derive the forecast standard deviation under the assumption of uncorrelated residuals.
- Stadard implementations of SARIMA (for example [pmdarima](http://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.ARIMA.html) generally provide prediction intervals.
- Prophet also provides its own prediction interval.

## Backtested empirical prediction interval

A model agnostic way of computing the prediction interval for any forecaster is by computing the empirical error distribution from a sequence of backtests in the training data[^williams_1971].

- For example we can use the expanding window approach to generate a sequence of (train,test) splits. The size of the test split is the *forecast horizon*.
- For each split we fit the model/forecaster on the train split and generate the point forecast $f(t)$ on the test split $y(t)$.
- We then measure the forecast error $e(t)=y(t)-f(t)$ on each of the test split.
- For each time step till the forecast horizon we build the empirical distribution of the forecasting errors $e(t)$ based on all the backtest splits.
- This distribution can then be used to set the prediction intervals for subsequent forecasts. The prediction interval is given by the lower and the upper percentiles of this empirical error distribution.
- Sometimes a smooth cone is fit to the percentiles across the forecasting horizons.

{{< hint warning >}}
This method assumes that the *future forecast errors* come form the same distribution as the *past forecast errors*. Also we would not want to back a lot in the time series during backtesting in order to have a stable distribution of forecasting errors. It also implicitly assumes that the prediction interval is independent of the time and only depends on the forecast horizon.
{{< /hint >}}

{{< hint info >}}
This method can be [effectively parallelized](https://www.youtube.com/watch?v=FoUX-muLlB4) to a great extent when used in combination with backtesting for accuracy evaluation in the outer loop.
{{< /hint >}}


## Bootstrapped prediction interval

When a normal distribution for the forecast errors is an unreasonable assumption, one alternative is to use bootstrapping, which only assumes that the **forecast errors are uncorrelated**.

The forecast error $e(t)$ is defined as
{{< katex >}}
e(t)=y(t)-f(t|t-1).
{{< /katex >}}
We can re-write this as
{{< katex >}}
y(t)=f(t|t-1)+e(t).
{{< /katex >}}
So we can **simulate** the next observation of a time series using
{{< katex >}}
y(t+1)=f(t+1|t)+e(t+1).
{{< /katex >}}
where $f(t+1|t)$ is the one-step forecast and $e(t+1)$ is the unknown future error. Assuming future errors will be similar to past errors, we can replace $e(t+1)$ by sampling from the collection of errors we have seen in the past (i.e., the residuals). Adding the new simulated observation to our data set, we can repeat the process to obtain
{{< katex >}}
y(t+2)=f(t+2|t+1)+e(t+2),
{{< /katex >}}
where $e(t+2)$ is another draw from the collection of residuals. Continuing in this way, we can simulate an entire set of future values for our time series.

Doing this repeatedly, we obtain many possible futures. Then we can compute prediction intervals by calculating percentiles for each forecast horizon. The result is called a **bootstrapped prediction interval**. The name *bootstrap* is a reference to pulling ourselves up by our bootstraps, because the process allows us to measure future uncertainty by only using the historical data.

## Quantile regression

For forecasting algorithms which are a reduction of supervised regression problems prediction intervals intervals are generally built with [quantile regression](https://en.wikipedia.org/wiki/Quantile_regression) which aims at estimating the conditional quantiles based on the quantile loss.  By combining two quantile regressors, it is possible to build an interval that is surrounded by the two sets of predictions produced by these two models.

This [blog](https://medium.com/@qucit/a-simple-technique-to-estimate-prediction-intervals-for-any-regression-model-2dd73f630bcb) claims quantile regression has some drawbacks in that the intervals are wider then necessary and each quantile needs its own regressor. Quantile regression is alos not available for all types of regression models. In scikit-learn, the only model that implements it is the Gradient Boosted Regressor.

{{< hint warning >}}
Quantile regression needs 3 models.
{{< /hint >}}

For tree based models these are the options available.

algorithm | prediction interval  | framework
--- | --- | ---
[GradientBoostingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html#sklearn.ensemble.GradientBoostingRegressor) | [quantile loss](https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_quantile.html) |  scikit-learn
[XGBoost](https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn) | |  scikit-learn
[XGBoost](https://xgboost.readthedocs.io/en/latest/python/python_intro.html) | [custom loss](https://towardsdatascience.com/regression-prediction-intervals-with-xgboost-428e0a018b) | python
[CatBoost](https://catboost.ai/) | [quantile loss](https://catboost.ai/docs/concepts/loss-functions-regression.html) |  python

## Deep Quantile Regression [^pinball_loss_video] [^pinball]


The below table summarizes the loss functions generally used to predict mean, median and quantiles.

Prediction | Loss Function
--- | --- |
Mean | Squared Error loss |
Median | Absolute Error loss |
Quantile | Pinball loss |

<!-- <img src="/img/quantile-loss.jpg"/ width="400" height="150">
 -->

Pinball loss function is defined as:

{{< katex >}}
pl(t,p) = \begin{cases}
 \ p\left(y(t)-f_p(t)\right)\quad\text{if}\quad  f_p(t) \leq y(t)\\
  (1-p)\left(f_p(t)-y(t)\right)\quad\text{if}\quad f_p(t) > y(t)
\end{cases}
{{< /katex >}}


<img src="/img/pinball_graph.jpg"/ width="400" height="250">

In the graph, three different quantiles were plotted, take quantile 0.8 as an example, when the error is positive (i.e. predicted value is higher than the actual value), the loss is less than that when error is negative. In another world, higher error is less punished, this makes sense in that for high quantile prediction, the loss function encourages higher prediction value, and vice versa for low quantile prediction.

Deep Quantile Regression works by directly predicting the quantile as output and minimizing its pinball loss function.
As depicted in the below figure, we can also predict multiple quantiles from a same model by estimating multiple quantiles and minimizing the aggregated pinball loss (via sum or weighted sum) of all quantiles.

<img src="/img/mul_pinball.jpg"/ width="400" height="250">


Or as an alternative, we can also enable multiple quantile predictions by training different models, where each model outputs a particular quantile prediction. This process can be parallized at a pipleline or task level.



## Distribution estimators

A [distribution estimator](https://medium.com/@qucit/a-simple-technique-to-estimate-prediction-intervals-for-any-regression-model-2dd73f630bcb) is a trained model that can compute quantile regression for any given probability without the need to do any re-training or recalibration. A distribution estimator works by producing a prediction and an estimate error for that prediction.

This approach assumes that $y|\mathbf{x}$ is a normal distribution. The base model) predicts the mean of the gaussian distribution, whereas the estimated error give us the standard deviation of the distribution.

The data $(\mathbf{X},\mathbf{y})$ is split into $(\mathbf{Xb},\mathbf{yb})$ to train the base model and $(\mathbf{Xe},\mathbf{ye})$ to train the error model. A base model $f_b$ is first trained on the $(\mathbf{Xb},\mathbf{yb})$. The squared prediction error $(\mathbf{ye}-f_b(\mathbf{Xe}))^2$ is computed on the validation set $(\mathbf{Xe},\mathbf{ye})$. The error model $f_e$ is then trained on $(\mathbf{Xe},(\mathbf{ye}-f_b(\mathbf{Xe}))^2)$ to regress on the squared error.

For any new instance $\mathbf{x}$ the mean prediction is given by $f_b(\mathbf{x})$ and the 90% prediction interval is given by $[f_b(\mathbf{x})-1.64\sqrt{f_e(\mathbf{x})},f_b(\mathbf{x})+1.64\sqrt{f_e(\mathbf{x})}]$.

{{< hint warning >}}
Distribution estimator needs 2 models.
{{< /hint >}}

See an example implementation with xgboost [here](https://github.ibm.com/retail-supply-chain/salesanalysis/blob/master/salesanalysis/RegressorXGBoost.py#L293).

## Metrics

**Prediction Interval Coverage Probability** (PICP) is the fraction of points for which the actual value is observed to be inside the prediction interval.

**Mean Prediction Interval Width** (MPIW) is the average width of the prediction interval across all instances.

See quantile loss.


## Some literature to read up

[^pearce_2018] [^tagasovska_2019]

Bootstrap methods for time series[^kreiss_2012][^sani_2015][^stine_wharton]

## Some open problems


- Consistent quantile regression
- Distribution estimator as a generalization of the backtest approach
- Parametirc forms for the error distirbution

## References

- https://en.wikipedia.org/wiki/Prediction_interval
- https://otexts.com/fpp2/prediction-intervals.html
- https://otexts.com/fpp2/arima-forecasting.html
- https://otexts.com/fpp2/bootstrap.html
- https://www.youtube.com/watch?v=FoUX-muLlB4


[^williams_1971]: A simple method for the construction of empirical confidence limits for economic forecasts, W. H. Williams and M. L. Goodman, Journal of the American Statistical Association, 1971, Volume 66, Number 336, Apprications Section.

[^pearce_2018]: Pearce, T., Brintrup, A., Zaki, M. & Neely, A.. (2018). [High-Quality Prediction Intervals for Deep Learning: A Distribution-Free, Ensembled Approach](http://proceedings.mlr.press/v80/pearce18a.html). Proceedings of the 35th International Conference on Machine Learning, in PMLR 80:4075-4084.

[^tagasovska_2019]: Tagasovska, Natasa and Lopez-Paz, David (2019). [Single-Model Uncertainties for Deep Learning](http://papers.neurips.cc/paper/8870-single-model-uncertainties-for-deep-learning), Advances in Neural Information Processing Systems 32.

[^kreiss_2012]: Jens-Peter Kreiss and Soumendra Nath Lahiri, [Bootstrap Methods for Time Series](https://doi.org/10.1016/B978-0-444-53858-1.00001-6), In Time Series Analysis: Methods and Applications, 2012. [pdf](citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.589.2692&rep=rep1&type=pdf#page=22)

[^sani_2015]: Amir Sani, Alessandro Lazaric, Daniil Ryabko. [The Replacement Bootstrap for Dependent Data](https://hal.inria.fr/hal-01144547/document). Proceedings of the IEEE International Symposium on Information Theory, Jun 2015.

[^stine_wharton]: [Resampling Methods for Time Series](www-stat.wharton.upenn.edu/~stine/stat910/lectures/13_bootstrap.pdf), Time Series Analysis, Wharton course.

[^pinball_loss_video]: https://www.youtube.com/watch?v=GpRuhE04lLs

[^pinball]: https://towardsdatascience.com/lightgbm-for-quantile-regression-4288d0bb23fd
