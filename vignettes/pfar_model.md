---
title: "Paired factor analysis for tree reconstruction"
author: "Kushal K Dey, Gao Wang"
date: "July 19, 2016"
output: pdf_document
---

Let $D_{nj}$ be the data corresponding to $n$-th sample and $j$-th gene. 
We assume for now that the data is Gaussian in its distribution. 
We assume there are $K$ factors or nodes of the tree. We assume the model 

$$ E \left [ D_{nj} | Z_{n} = (k_1, k_2), \lambda_{n}=q, F \right] = q F_{k1,j} + (1-q) F_{k2,j} $$

We assume a prior on $\lambda$,

$$ Pr \left [ \lambda_{n} = q \right ] = \pi_{q}  $$

Then we can write 

$$ Pr \left [ D_{n} | Z_{n}=(k_1,k_2), F, s^2_{j=1,2,\cdots,J} \right ] = \sum_{q} \pi_{q} Pr \left [D_{n} | Z_{n}=(k_1,k_2), \lambda_{n}=q, F, s^2_{j=1,2,\cdots,J} \right ] $$

where $s^2_{j}$ is the variance of the $j$th feature.

We also assume the prior 

$$ Pr \left [ Z_{n} = (k_1, k_2) \right ] = \pi_{k_1,k_2} \hspace{1 cm} k_1 < k_2$$

Then we can write 

$$ Pr \left [ D_{n} | \pi, F \right ] = \sum_{k_1 < k_2} \pi_{k_1, k_2} Pr \left [ D_{n} | Z_{n}=(k_1,k_2), F, s^2_{j=1,2,\cdots,J} \right ] $$

We define the joint prior over the edges and the fraction of the edge represented as 

$$ \pi_{k_1,k_2, q} = \pi_{k_1, k_2} \pi_{q} \hspace{1 cm} k_1 < k_2 $$

The overall likelihood 

$$ L(\pi, F) = \prod_{n=1}^{N} Pr \left [ D_{n} | \pi, F, s^2_{j=1,2,\cdots,J} \right ] $$

or we can write it as 

$$ L(\pi, F) = \prod_{n=1}^{N} \sum_{k_1 < k_2} \sum_{q} \left [ \pi_{k_1,k_2, q} \times \prod_{j=1}^{G} N \left (D_{nj}; q F_{k_1,g} + (1-q) F_{k_2, g}, s^2_{j} \right) \right ]  $$

$$ log L (\pi, F) = \sum_{n=1}^{N} log \left (\sum_{k_1 < k_2} \sum_{q} \left [ \pi_{k_1,k_2, q} \times \prod_{j=1}^{G} N \left (D_{nj}; q F_{k_1,g} + (1-q) F_{k_2, g}, s^2_{j} \right) \right ] \right ) $$

This is the log likelihood we want to maximize and we need to return this log-likelihood. 

We assume that $q$ can take a finite set of values between $0$ and $1$, 
say $1/100, 2/100, \cdots, 90/100, 1$.

Suppose we have run upto $m$ iterations. For the $(m+1)$th iteration, we have 
$$\delta^{(m+1)}_{n, k_1, k_2, q} = Pr \left [ Z_{n} = (k_1, k_2), \lambda_{n} = q | \pi^{(m)}, F^{(m)}, s^{(m)}_{j=1,2,\cdots,J}, D_{n} \right ] $$

$$\delta^{(m+1)}_{n, k_1, k_2, q} \propto Pr \left [ Z_{n} = (k_1, k_2) \right] Pr \left [ \lambda_{n} = q \right] Pr \left [ D_{n} |  \pi^{(m)}, F^{(m)}, s^{(m)}_{j=1,2,\cdots,J}, Z_{n}= (k_1, k_2), \lambda_{n}=q \right] $$

$$ \delta^{(m+1)}_{n, k_1, k_2, q} \propto \pi^{(m)}_{k_1,k_2, q} \prod_{j} N \left (D_{nj} | qF^{(m)}_{k_1,j} + (1-q)F^{(m)}_{k_2,j}, {s_j^{(m)}}^2 \right) $$

where ${s_j^{(m)}}^2$ is the residual variance of gene $j$.

We normalize $\delta$ so that 

$$ \sum_{k_1 < k_2} \sum_{q} \delta^{(m+1)}_{n, k_1, k_2, q} = 1 \hspace{1 cm} \forall n $$

We define 

$$ \pi^{(m+1)}_{k_1, k_2, q} = \frac{1}{N}\sum_{n=1}^{N} \delta^{(m+1)}_{n, k_1, k_2, q} $$


We have therefore updated $\pi^{(m)}_{k_1, k_2, q}$ to $\pi^{(m+1)}_{k_1, k_2, q}$.

We define the parameter 

$$ \theta : = \left (\pi_{k_1,k_2, q}, F, s_{j=1,2,\cdots,J} \right ) $$

We define the complete loglikelihood 

$$ log L_{c} \left (\theta; D, Z, \lambda \right ) = log \pi_{k_1,k_2, q} + log L (D | Z, \lambda, q, F) $$

We take the expectation of this quantity with respect to $\left [ Z, \lambda | D, \theta^{(m)} \right ]$.

$$ Q (\theta | \theta^{(m)}) \propto - \sum_{n=1}^{N} \sum_{k_1 < k_2} \sum_{q} \delta^{(m+1)}_{n, k_1, k_2, q}  \sum_{j} \left [ log s_{j} + \frac{(D_{nj} - q F_{k_1,j} - (1-q) F_{k_2,j})^2}{2s^2_{j}} \right]$$

We try to maximize this quantity with respect to $F$, So, we can take derivative with respect to $F$ and try to solve the resulting normal equation.

This equation, conditional on $\left [ Z, \lambda | D, \theta^{(m)} \right ]$, can be written as 

$$ D_{N \times J} = L_{N \times K} F_{K \times J} + E_{N \times J} $$

where 

$$ e_{nj} \sim N(0, s^2_{j}) $$

We define 

$$ D^{'}_{nj} : = \frac{D_{nj}}{s_{j}} $$

If we consider finding the factors on a gene by gene basis, we do not need to worry about $s_j$.

\begin{align*}
L_{nk} =
\begin{cases}
    q~\text{or}~(1-q) & \lambda_{n}=q \\
    0 & \text{o.w.}
\end{cases}
\end{align*}

We have 

$$ E_{ Z, \lambda | D, \theta^{(m)}} \left [ L_{nk} \right ] = \sum_{q}  \sum_{k_2 > k} q \delta^{(m+1)}_{n,k,k_2, q}  + \sum_{q}  \sum_{k_1 < k} (1-q) \delta^{(m+1)}_{n,k1,k,q}$$

$$ E_{ Z, \lambda | D, \theta^{(m)}} \left [ L^2_{nk} \right ] = \sum_{q}  \sum_{k_2 > k} q^2 \delta^{(m+1)}_{n,k,k_2, q}  + \sum_{q}  \sum_{k_1 < k} (1-q)^2 \delta^{(m+1)}_{n,k1,k,q} $$

Also for any $k \neq l$,

$$ E_{ Z, \lambda | D, \theta^{(m)}} \left [ L_{nk}L_{nl} \right ] =
\sum_{q} q(1-q) \delta^{(m+1)}_{n,k,l,q} $$

We use these to solve for the equation

$$ \left [ E_{ Z, \lambda | D, \theta^{(m)}} \left( L^{T}L \right ) \right ] F \approx \left [ E_{ Z, \lambda | D, \theta^{(m)}} (L) \right] ^{T} D $$

The solution therefore is 

$$ F \approx \left [ E_{ Z, \lambda | D, \theta^{(m)}} \left( L^{T}L \right ) \right]^{-1} \left [ E_{ Z, \lambda | D, \theta^{(m)}} (L) \right]^{T} D $$

For $W = L^{T}L$

$$ W_{kl} = \sum_{n} L_{kn}L_{nl} $$

$$ E_{ Z, \lambda | D, \theta^{(m)}} \left ( W_{kl} \right ) = \sum_{n}  E_{ Z, \lambda | D, \theta^{(m)}} \left ( L_{nk}L_{nl} \right) $$

We use the definition of $E_{ Z, \lambda | D, \theta^{(m)}} \left [ L^2_{nk} \right ]$ 
and $E_{ Z, \lambda | D, \theta^{(m)}} \left [ L_{nk}L_{nl} \right ]$ 
from above to solve this linear system. 

In the same way as we computed $F$ by solving for the normal equation obtained from taking derivative of the function $Q (\theta | \theta^{(m)})$, we take derivative of the latter with respect to $s^2_{j}$ to obtain EM updates of the genes variance terms. O  taking derivative, we obtain the estimates as 

$$ \hat{s}^2_{j} = \sum_{n=1}^{N} \sum_{k_1 < k_2} \sum_{q} \delta^{(m+1)}_{n, k_1, k_2, q} (D_{nj} - q F_{k_1,j} - (1-q) F_{k_2,j})^2 $$

where the $F$ are the estimated values of the factors from the previous step.

We then continue this procedure described above for multiple iterations.
