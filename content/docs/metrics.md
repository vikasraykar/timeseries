---
bookShowToc: true
weight: 2
title: "Metrics"
---

# Metrics

## Evaluating feature based local explanations

Let {{< katex >}}f{{< /katex >}} be a **black box predictor** that maps an input {{< katex >}}\mathbf{x} \in \mathbb{R}^d{{< /katex >}} to an output {{< katex >}}f(\mathbf{x}) \in \mathbb{R}{{< /katex >}}.

An **explanation function** {{< katex >}}g{{< /katex >}} takes in a predictor {{< katex >}}f{{< /katex >}} and an instances {{< katex >}}\mathbf{x}{{< /katex >}} and returns the feature importance scores {{< katex >}}g(f,\mathbf{x}) \in \mathbb{R}^d{{< /katex >}}.

Let {{< katex >}}\rho: \mathbb{R}^d \times \mathbb{R}^d \rightarrow \mathbb{R}^{+}{{< /katex >}} be a **distance metric** over input instances.

Let {{< katex >}}D: \mathbb{R}^d \times \mathbb{R}^d \rightarrow \mathbb{R}^{+}{{< /katex >}} be a **distance metric** over explanations.

An **evaluation metric** {{< katex >}}\mu{{< /katex >}} takes in as input a predictor {{< katex >}}f{{< /katex >}},an explanation fucntion {{< katex >}}g{{< /katex >}}, and input {{< katex >}}\mathbf{x}{{< /katex >}}, and outputs a scalar {{< katex >}}\mu(f,g;\mathbf{x}){{< /katex >}}.

We wil mainly focus on these threes evaluation metrics which can be evaluated without access to ground-truth explanations[^bhatt_2020].

### Faithfulness

{{< hint info >}}
(high) faithfulness,relevance,fidelity.
{{< /hint >}}

The feature importance scores from {{< katex >}}g{{< /katex >}} should correspond to the importance features of {{< katex >}}\mathbf{x}{{< /katex >}} for {{< katex >}}f{{< /katex >}}, such that, when we set a particular set of features {{< katex >}}\mathbf{x}_s{{< /katex >}} to a baseline value {{< katex >}}\overline{\mathbf{x}}_s{{< /katex >}}, the change in predictor's output should be proportional (measured via correlation) to the sum of the attribution scores of features in {{< katex >}}\mathbf{x}_s{{< /katex >}}.

For a subset of indices {{< katex >}}S \subset {1,2,...,d}{{< /katex >}}, let {{< katex >}}\mathbf{x}_s = ( \mathbf{x}_i,i \in S ){{< /katex >}} a sub-vector of input features. For a given subset size {{< katex >}}|S|{{< /katex >}}, we define faithfullness as

{{< katex display >}}
\mu_{F}(f,g,|S|;\mathbf{x}) = \text{corr}_{S \in \binom {d}{|S|}}\left( \sum_{i \in S}g(f,\mathbf{x})_{i},f(\mathbf{x})-f(\mathbf{x}|\mathbf{x}_s=\overline{\mathbf{x}}_s)\right)
{{< /katex >}}

The baseline can be the mean of the training data.

### Sensitivity

{{< hint info >}}
(low) sensitivity, stability, reliability, explanation continuity.
{{< /hint >}}

If inputs are near each other and their model outputs are similar, then their explanations should be close to each other.

Let {{< katex >}}\mathcal{N}_r(\mathbf{x}){{< /katex >}} be a neighborhood of datapoints within a radius {{< katex >}}r{{< /katex >}} of {{< katex >}}\mathbf{x}{{< /katex >}}.

{{< katex display >}}
\mathcal{N}_r(\mathbf{x}) = \left\{ \mathbf{z} \in \mathcal{D}_x | \rho(\mathbf{x},\mathbf{z}) \leq r, f(\mathbf{x}) = f(\mathbf{z}) \right\}
{{< /katex >}}

Max Sensitivity

{{< katex display >}}
\mu_{M}(f,g,r;\mathbf{x}) = \max_{z\in\mathcal{N}_r(\mathbf{x})} D(g(f,\mathbf{x}),g(f,\mathbf{z}))
{{< /katex >}}

Average Sensitivity

{{< katex display >}}
\mu_{A}(f,g,r;\mathbf{x}) = \int_{\mathcal{N}_r(\mathbf{x})} D(g(f,\mathbf{x}),g(f,\mathbf{z})) \mathbb{P}_{\mathbf{x}}(\mathbf{z}) d\mathbf{z}
{{< /katex >}}

### Complexity

{{< hint info >}}
(low) complexity,information gain,sparsity.
{{< /hint >}}

A complex explantion is one that uses all the {{< katex >}}d{{< /katex >}} features in its explanation. The simplest explanation would be concentrated on one feature.

We define complexity as the entropy of the fractional contribution distribution.

{{< katex display >}}
\mu_{C}(f,g;\mathbf{x}) = \mathbb{E}_{i}\left[ -\ln(\mathbb{P}_{g})\right] = - \sum_{i=1}^{d} \mathbb{P}_{g}(i) \ln(\mathbb{P}_{g}(i))
{{< /katex >}}

where {{< katex >}}\mathbb{P}_{g}{{< /katex >}} is the fractional contribution distribution

{{< katex display >}}
\mathbb{P}_{g}(i) = \frac{|g(f,\mathbf{x})_i|}{\sum_{j=1}^{d}|g(f,\mathbf{x})_j|}.
{{< /katex >}}


## References

[^bhatt_2020]: [Evaluating and Aggregating Feature-based Model Explanations](https://www.ijcai.org/Proceedings/2020/417), Bhatt, Umang and Weller, Adrian and Moura, José M. F., Proceedings of the Twenty-Ninth International Joint Conference on Artificial Intelligence (IJCAI-20), 2020.





