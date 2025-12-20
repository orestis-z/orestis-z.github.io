---
layout: post
title: "Backpropagation in Practice"
date: 2025-12-20 12:00:00 +0100
categories: [AI, Machine Learning]
tags: [backpropagation, theory, mathematics]
author: Orestis Zambounis
---

The "magic" behind the modern deep learning revolution isn't just the architectures themselves, but our ability to train them efficiently. At the heart of this efficiency lies backpropagation, an algorithm that is often misunderstood as a "black box." In reality, it is an elegant application of the Multivariate Chain Rule, implemented via Dynamic Programming. This post aims to demystify the algorithm by bridging the gap between classical textbook notations and the high-performance implementation strategies used in frameworks like PyTorch and TensorFlow.

## Notations

In the following, we introduce the notations representing the computations within a neural network. As we progress from simple neurons to complex graphs, the notation evolves to remain intuitive.

### Textbook Notation

At its most fundamental level, a neuron consists of a linear transformation followed by a non-linear activation function:

$$
f: \mathbb{R}^n \to \mathbb{R}, \quad f(\mathbf{x}) = \sigma(\mathbf{w}^\top \mathbf{x} + b) \tag{1}
$$

where $\sigma: \mathbb{R} \to \mathbb{R}$ is the activation function, $\mathbf{w} \in \mathbb{R}^n$ is the weight vector, and $b \in \mathbb{R}$ is the scalar bias. For multiple neurons operating on the same input, we use matrix notation:

$$
f: \mathbb{R}^n \to \mathbb{R}^m, \quad f(\mathbf{x}) = \sigma(\mathbf{W}\mathbf{x} + \mathbf{b}) \tag{2} \label{eq:multi-neurons}
$$

where $\mathbf{W} \in \mathbb{R}^{m \times n}$ is the weight matrix and $\mathbf{b} \in \mathbb{R}^m$ is the bias vector. We define this operation as a layer $l$:

$$
\mathbf{a}^l = \sigma(\mathbf{W}^l \mathbf{a}^{l-1} + \mathbf{b}^l), \quad l \in \{1, \dots, L\} \tag{3}
$$

In a *Fully Connected Neural Network* (FCNN), we chain these layers together:

$$
\mathbf{o} = f_L(f_{L-1}(\dots f_1(\mathbf{x}) \dots)) \tag{4}
$$



### Row-Vector Notation

In contrast to textbooks, libraries like PyTorch or TensorFlow use right-multiplication:

$$
f: \mathbb{R}^{B \times n} \to \mathbb{R}^{B \times m}, \quad f(\mathbf{X}) = \sigma(\mathbf{X}\mathbf{W}^\top + \mathbf{b}) \tag{5}
$$

Here, $B$ represents the batch size. Why the difference?

1.  **Batching Convention:** In data science, rows represent observations and columns represent features. This matches standard dataset structures (CSVs, SQL).
2.  **Hardware Efficiency:** Modern GPUs use row-major storage. If data were stored as columns, the GPU would have to "jump" across memory addresses to fetch a single sample. Row-wise processing allows for *memory coalescing*, maximizing throughput.

## Training and Optimization

The objective is to find parameters $\mathbf{W}$ and $\mathbf{b}$ that minimize the distance between the network output $\mathbf{o}$ and labels $\mathbf{y}$ across a dataset of size $N$:

$$
\hat{\mathbf{W}}, \hat{\mathbf{b}} = \arg\min_{\mathbf{W}, \mathbf{b}} \frac{1}{N} \sum_{i=1}^N C(\mathbf{W}, \mathbf{b}; \mathbf{x}_i, \mathbf{y}_i) \tag{6}
$$

We use *Stochastic Gradient Descent* (SGD) to update parameters. Since calculating the gradient over $N$ samples is expensive, we use mini-batches of size $M < N$. This provides a "noisy" gradient that helps the network escape local minima:

$$
\mathbf{W}_t = \mathbf{W}_{t-1} - \eta \nabla_{\mathbf{W}} C, \quad \mathbf{b}_t = \mathbf{b}_{t-1} - \eta \nabla_{\mathbf{b}} C \tag{7}
$$



## Gradient Derivation

To calculate these gradients, we use the *Multivariate Chain Rule*. If $f$ depends on $x$ through intermediate variables $g_i$:

$$
\frac{\partial f}{\partial x} = \sum_{i} \frac{\partial f}{\partial g_i} \frac{\partial g_i}{\partial x} \tag{8}
$$

### 1. The Output Layer Error

Let $\mathbf{z}^l = \mathbf{W}^l \mathbf{a}^{l-1} + \mathbf{b}^l$. For the last layer $L$, the partial derivatives for a single weight and bias are:

$$
\frac{\partial C}{\partial b_i^L} = \frac{\partial C}{\partial z_i^L} \frac{\partial z_i^L}{\partial b_i^L} = \frac{\partial C}{\partial z_i^L} \tag{9}
$$

$$
\frac{\partial C}{\partial W_{ij}^L} = \frac{\partial C}{\partial z_i^L} \frac{\partial z_i^L}{\partial W_{ij}^L} = \frac{\partial C}{\partial z_i^L} a_j^{L-1} \tag{10}
$$

We define the error term $\delta_i^L := \frac{\partial C}{\partial z_i^L}$. In matrix form:

$$
\boldsymbol{\delta}^L = \nabla_{\mathbf{a}^L} C \odot \sigma'(\mathbf{z}^L) \tag{BP1}
$$

### 2. Propagating Backward

For a hidden layer $l$, a change in $z_i^l$ affects the cost through all neurons $k$ in the next layer $l+1$. Summing these paths:

$$
\delta_i^l = \sum_k \left( \frac{\partial C}{\partial z_k^{l+1}} \frac{\partial z_k^{l+1}}{\partial a_i^l} \right) \frac{\partial a_i^l}{\partial z_i^l} = \left( \sum_k \delta_k^{l+1} W_{ki}^{l+1} \right) \sigma'(z_i^l) \tag{11}
$$

Which gives the recursive matrix form:

$$
\boldsymbol{\delta}^l = ((\mathbf{W}^{l+1})^\top \boldsymbol{\delta}^{l+1}) \odot \sigma'(\mathbf{z}^l) \tag{BP2}
$$

### 3. Parameter Gradients

Using the error terms, the final gradients for layer $l$ are:

$$
\nabla_{\mathbf{b}^l} C = \boldsymbol{\delta}^l \tag{BP3}
$$

$$
\nabla_{\mathbf{W}^l} C = \boldsymbol{\delta}^l (\mathbf{a}^{l-1})^\top \tag{BP4}
$$

*Note on Batching:* For a batch size $B$, the total cost is $C_{tot} = \frac{1}{B}\sum C_i$. Consequently, the final parameter updates use the *averaged gradients* across the batch.

## Generalization: From Layers to Graphs

Modern architectures like ResNets are *Directed Acyclic Graphs* (DAGs). This is where *Reverse-Mode Automatic Differentiation* comes in:

$$
\frac{\partial C}{\partial v_i} = \sum_{j \in \text{Children}(i)} \frac{\partial C}{\partial v_j} \frac{\partial v_j}{\partial v_i} \tag{12}
$$



| | FCNN | General DAG |
| :--- | :--- | :--- |
| **Perspective** | Layers as blocks | Every operation is a node |
| **Error Signal** | $\delta_i^l = \frac{\partial C}{\partial z_i^l}$ | $\bar{v}_i = \frac{\partial C}{\partial v_i}$ |
| **Gradient Flow** | Weighted sum of next layer | Sum over all successor nodes |

## Conclusion

Backpropagation is a recursive application of the chain rule optimized through *Dynamic Programming*. It achieves a temporal complexity of $\mathcal{O}(P)$, where $P$ is the number of parameters, in a single backward pass. This linear efficiency is what allows us to train the massive models we see today.

---

### References

* Nielsen, Michael A. [Neural networks and deep learning](http://neuralnetworksanddeeplearning.com/). 2015.
* Goodfellow, Ian, et al. [Deep learning](https://www.deeplearningbook.org/). MIT press, 2016.
