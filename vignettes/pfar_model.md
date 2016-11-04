---
title: "Paired factor analysis for tree reconstruction"
author: "Kushal K Dey, Gao Wang"
date: "July 19, 2016"
output: pdf_document
---

## The model

Let $D_{nj}$ be the data corresponding to $n$-th sample and $j$-th feature, where $n$ runs from $1$ to $N$ and $j$ runs from $1$ to $G. We assume as a first pass that the data is Gaussian in its distribution. 

Let us consider a graph model with $K$ nodes (factors) and $E$ edges (in our model right now, $E=K(K-1)/2$).

Let us define latent variables $Z$ and $\Lambda$ such that $z_{n}$ is a $E \times 1$ vector which is one-hot, where $E$ is the total number of edges in the graph. Again $\lambda_{n}$ is a $Q \times 1$ one-hot vector, where $Q$ is the cardinality of the set of values that $q$ can take. 

Then one can write 

$$  Pr (Z |  \pi )  = \prod_{n=1}^{N} \prod_{k_1 < k_2} \pi_{k1,k2}^{z_{n,k1,k2}} $$

We assume that $q$ can take a finite set of values between $0$ and $1$, 
say $1/100, 2/100, \cdots, 90/100, 1$.

Also

$$  Pr (\Lambda | \delta ) = \prod_{n=1}^{N} \prod_{q=1}^{Q} \delta_{q}^{\lambda_{nq}} $$

here $Q$ is $100$, if there are $100$ values that $q$ can take.


We assume the model 

\begin{eqnarray}
 E \left [ D_{nj} | Z_{n, k_1, k_2} = 1, \lambda_{n,q}=1, F \right] = q F_{k1,j} + (1-q) F_{k2,j}
\end{eqnarray}

The prior we define over $\Lambda$ is equivalent to writing 

$$ Pr \left [ \Lambda_{n, q} = 1 \right ] = \pi_{q}  $$

Then we can write 

$$ Pr \left [ D_{n} | Z_{n, k_1, k_2}=1, F, s^2_{j=1,2,\cdots,J} \right ] = \sum_{q} \pi_{q} Pr \left [D_{n} | Z_{n, k_1, k_2}=1, \Lambda_{n, q}=1, F, s^2_{j=1,2,\cdots,J} \right ] $$

where $s^2_{j}$ is the variance of the $j$th feature.

We also get from the prir on $Z_{n,k1,k2}$,

$$ Pr \left [ Z_{n, k_1, k_2} = 1 \right ] = \pi_{k_1,k_2} \hspace{1 cm} k_1 < k_2$$

Then we can write 

\begin{eqnarray}
 Pr \left [ D_{n} | \pi, F \right ] = \sum_{k_1 < k_2} \pi_{k_1, k_2} Pr \left [ D_{n} | Z_{n, k_1, k_2}=1, F, s^2_{j=1,2,\cdots,J} \right ] 
\end{eqnarray}

We define the joint prior over the edges and the fraction of the edge represented as 

$$ \pi_{k_1,k_2, q} = \pi_{k_1, k_2} \pi_{q} \hspace{1 cm} k_1 < k_2 $$

The overall likelihood 

$$ L(\pi, F) = \prod_{n=1}^{N} Pr \left [ D_{n} | \pi, F, s^2_{j=1,2,\cdots,J} \right ] $$

or we can write it as 

$$L(\pi, F) = \prod_{n=1}^{N} \sum_{k_1 < k_2} \sum_{q} \left [ \pi_{k_1,k_2, q} \times \prod_{j=1}^{G} N \left (D_{nj}; q F_{k_1,g} + (1-q) F_{k_2, g}, s^2_{j} \right) \right ]  $$

And the log likelihood

\begin{eqnarray}
log L (\pi, F) = \sum_{n=1}^{N} log \left (\sum_{k_1 < k_2} \sum_{q} \left [ \pi_{k_1,k_2, q} \times \prod_{j=1}^{G} N \left (D_{nj}; q F_{k_1,g} + (1-q) F_{k_2, g}, s^2_{j} \right) \right ] \right )
\end{eqnarray}

This is the quantity we want to maximize. 

\subsection{E step (EM updates)}

Suppose we have run upto $m$ iterations. For the $(m+1)$th iteration, we have 

\begin{eqnarray}
\delta^{(m+1)}_{n, k_1, k_2, q} &=& Pr \left [ Z_{n, k1, k2} = 1, \Lambda_{n,q} = 1 | \pi^{(m)}, F^{(m)}, s^{(m)}_{j=1,2,\cdots,J}, D_{n} \right ] \\
 &\propto& Pr \left [ Z_{n, k1, k2} = 1 \right] Pr \left [ \lambda_{n,q} = 1 \right] Pr \left [ D_{n} | \pi^{(m)}, F^{(m)}, s^{(m)}_{j=1,2,\cdots,J}, Z_{n, k1, k2}= 1, \lambda_{n, q}=1 \right] \\
 &\propto& \pi^{(m)}_{k_1,k_2, q} \prod_{j} N \left (D_{nj} | qF^{(m)}_{k_1,j} + (1-q)F^{(m)}_{k_2,j}, {s_j^{(m)}}^2 \right)
\end{eqnarray}

where ${s_j^{(m)}}^2$ is the residual variance of feature $j$.

We normalize $\delta$ so that 

$$ \sum_{k_1 < k_2} \sum_{q} \delta^{(m+1)}_{n, k_1, k_2, q} = 1 \hspace{1 cm} \forall n $$

We define 

$$ \pi^{(m+1)}_{k_1, k_2, q} = \frac{1}{N}\sum_{n=1}^{N} \delta^{(m+1)}_{n, k_1, k_2, q} $$


We have therefore updated $\pi^{(m)}_{k_1, k_2, q}$ to $\pi^{(m+1)}_{k_1, k_2, q}$.


\subsection{Variational EM updates (Model 1)}

In model 1 Variational EM, we assume that the variational distributions of the two latent variables $Z$ and $\Lambda$ are independent. 

In this set up, we assume prior distributions of $\pi$ and $\delta$ as follows

$$ Pr (\pi | \alpha_{0}) = C (\alpha_0) \prod_{k_1 < k_2} \pi_{k1, k2}^{\alpha_0 -1} $$

Similarly the prior distribution for $\delta$ is 

$$ Pr (\delta | \beta_0) = C (\beta_0) \prod_{q=1}^{Q} \delta_{q}^{\beta_0 -1 }  $$

The likelihood above can be written as 

$$ p (D | Z, \Lambda, F, s_{j=1,2,\cdots,J}) = \prod_{n=1}^{N} \prod_{k_1 < k_2} \prod_{q=1}^{Q} [ \prod_{j=1}^{G} N (D_{ng} | qF_{k1,g} + (1-q)F_{k2,g}, s^2_{g})]^{\Lambda_{nq}Z_{n,k1,k2}} $$

The joint probablity distribution distribution is given by 

$$ p (D, Z, \Lambda, \pi, \delta | F, s_{j=1,2,\cdots,J}, \alpha_{0}, \beta_{0}) = p (\pi | \alpha_0) p (\delta | \beta_0)  p (\Lambda | \delta) p (Z | \pi) p (D | Z, \Lambda, F, s_{j=1,2,\cdots,J})  $$


We assume the following mean field variational distribution.

$$ q(Z, \Lambda, \pi, \delta) = q(Z) q(\Lambda) q(\pi) q(\delta) $$

The variational distribution for $Z$

\begin{align}
\ln {q^{\star} (Z)}  = E_{\pi, \delta, \Lambda} \left [ ln p(\pi|\alpha_0)
  + ln p(\delta | \beta_0) + ln p(\Lambda | \delta) + ln p(Z | \pi) 
  + ln p(D | Z, \Lambda, F, s_{j=1,2,\cdots,J}) \right ] & \\ 
  \hspace{-2 cm} = E_{\pi, \delta, \Lambda} \left [ ln p(Z | \pi) + 
 ln p(D | Z, \Lambda, F, s_{j=1,2,\cdots,J}) \right] + constant & \\ 
  = \sum_{n=1}^{N} \sum_{k_1 < k_2} z_{n, k1, k2} E_{\pi} \left [ ln (\pi_{k1,k2}) \right ] + \sum_{n=1}^{N} \sum_{k_1 < k_2}  z_{n, k1, k2} \sum_{q} E_{\Lambda}(\lambda_{nq}) \left [ - \sum_{j=1}^{G} ln (s_j) \right . & \\
\qquad \left.  - \frac{G}{2} ln (2 \pi) - \sum_{j=1}^{G} \frac{(D_{nj} - qF_{k1,j} - (1-q)F_{k2,j})^2}{2s^2_j} \right] 
\end{flalign}

\begin{align}
ln q^{\star} (\Lambda)  = E_{\pi, \delta, Z} \left [ ln p(\pi|\alpha_0)
  + ln p(\delta | \beta_0) + ln p(\Lambda | \delta) + ln p(Z | \pi) 
  + ln p(D | Z, \Lambda, F, s_{j=1,2,\cdots,J}) \right ] & \\ 
   \hspace{-2 cm} = E_{\pi, \delta, Z} \left [ ln p(\Lambda | \delta) + 
 ln p(D | Z, \Lambda, F, s_{j=1,2,\cdots,J}) \right] + constant & \\ 
= \sum_{n=1}^{N} \sum_{q=1}^{Q} \lambda_{n,q} E_{\delta} \left [ ln (\delta_{q}) \right ] + \sum_{n=1}^{N} \sum_{q} \lambda_{nq} \sum_{k_1 < k_2} E_{Z}(z_{n, k1, k2}) \left [ - \sum_{j=1}^{G} ln (s_j) \right . & \\
\qquad \left.  - \frac{G}{2} ln (2 \pi) - \sum_{j=1}^{G} \frac{(D_{nj} - qF_{k1,j} - (1-q)F_{k2,j})^2}{2s^2_j} \right] 
\end{align}

So we get 

$$ q*(Z) \propto \prod_{n=1}^{N} \prod_{k_1 < k_2} \rho_{n, k1, k2}^{Z_{n,k1,k2}} $$

where  we define

$$ \rho_{n, k1, k2} \propto exp \left (E_{\pi} \left [ ln (\pi_{k1,k2}) \right ]   + \sum_{q} E_{\Lambda}(\lambda_{nq}) \left [ - \sum_{j=1}^{G} ln (s_j)  - \frac{G}{2} ln (2 \pi) - \sum_{j=1}^{G} \frac{(D_{nj} - qF_{k1,j} - (1-q)F_{k2,j})^2}{2s^2_j} \right] \right) $$

$$ \rho_{n, k1, k2} \propto exp \left (E_{\pi} \left [ ln (\pi_{k1,k2}) \right ]   + \sum_{q} \nu_{nq} \left [ - \sum_{j=1}^{G} ln (s_j)  - \frac{G}{2} ln (2 \pi) - \sum_{j=1}^{G} \frac{(D_{nj} - qF_{k1,j} - (1-q)F_{k2,j})^2}{2s^2_j} \right] \right) $$

$$ \rho_{n, k1, k2} \propto exp \left ( \psi_{a_{k1,k2}} - \psi(\sum_{k1 < k2} a_{k1,k2})   +  \left [ - \sum_{j=1}^{G} ln (s_j)  - \frac{G}{2} ln (2 \pi) - \sum_{j=1}^{G} \frac{(D_{nj} - qF_{k1,j} - (1-q)F_{k2,j})^2}{2s^2_j} \right] \right) $$

$\rho_{n,k1,k2}$ is normalized to sum to 1 for each $n$ over $k1$ and $k2$.

We also get

$$ q*(\Lambda) \propto \prod_{n=1}^{N} \prod_{q=1}^{Q} \nu_{nq}^{\Lambda_{nq}} $$

where 

$$ \nu_{nq} : \propto exp \left (  E_{\delta} \left [ ln (\delta_{q}) \right ] + \sum_{k_1 < k_2} E_{Z}(z_{n, k1, k2}) \left [ - \sum_{j=1}^{G} ln (s_j) - \frac{G}{2} ln (2 \pi) - \sum_{j=1}^{G} \frac{(D_{nj} - qF_{k1,j} - (1-q)F_{k2,j})^2}{2s^2_j} \right ] \right ) $$

$$ \nu_{nq} : \propto exp \left (  E_{\delta} \left [ ln (\delta_{q}) \right ] + \sum_{k_1 < k_2} \rho_{n,k1,k2} \left [ - \sum_{j=1}^{G} ln (s_j) - \frac{G}{2} ln (2 \pi) - \sum_{j=1}^{G} \frac{(D_{nj} - qF_{k1,j} - (1-q)F_{k2,j})^2}{2s^2_j} \right ] \right ) $$


$$ \nu_{nq} : \propto exp \left (  \psi(b_{q}) - \psi(\sum_{q=1}^{Q} b_{q}) +  \left [ - \sum_{j=1}^{G} ln (s_j) - \frac{G}{2} ln (2 \pi) - \sum_{j=1}^{G} \frac{(D_{nj} - qF_{k1,j} - (1-q)F_{k2,j})^2}{2s^2_j} \right ] \right ) $$

$\nu_{nq}$ are normalized to sum to 1.

We can also derive variational distributions similarly for $\pi$ and $\delta$.

\begin{align}
ln q^{\star} (\pi) = E_{\Lambda, Z, \delta} \left [ ln p(\pi|\alpha_0)
  + ln p(\delta | \beta_0) + ln p(\Lambda | \delta) + ln p(Z | \pi) 
  + ln p(D | Z, \Lambda, F, s_{j=1,2,\cdots,J}) \right ] & \\ 
  = E_{Z} \left [ ln p(Z | \pi) \right] + ln p(\pi | \alpha_0) + constant & \\
  = \sum_{n=1}^{N}\sum_{k1 < k2} E(z_{n,k1,k2}) ln \pi_{k1,k2} + (\alpha_0 -1) \sum_{k1 < k2} ln \pi_{k1,k2} & \\
  = \sum_{k1 < k2} \left [ \sum_{n=1}^{N} \rho_{n,k1,k2} + (\alpha_0 -1) \right] ln \pi_{k} \\
\end{align}

We define 

$$ a_{k1,k2} = \alpha_0 + \sum_{n=1}^{N} \rho_{n,k1,k2} $$

$$ q^{\star} (\pi) = Dir(\pi | a)  $$


\begin{align}
ln q^{\star} (\delta) = E_{\Lambda, Z, \pi} \left [ ln p(\pi|\alpha_0)
  + ln p(\delta | \beta_0) + ln p(\Lambda | \delta) + ln p(Z | \pi) 
  + ln p(D | Z, \Lambda, F, s_{j=1,2,\cdots,J}) \right ] & \\ 
  = E_{\Lambda} \left [ ln p(\Lambda | \delta) \right] + ln p(\delta | \beta_0) + constant & \\
  = \sum_{n=1}^{N}\sum_{q=1}^{Q} E(\lambda_{n,q}) ln \delta_{q} + (\beta_0 -1) \sum_{q=1}^{Q} ln \delta_{q} & \\
  = \sum_{q=1}^{Q} \left [ \sum_{n=1}^{N} \nu_{n,q} + (\beta_0 -1) \right] ln \delta_{q} \\
\end{align}

We define 

$$ b_{q} = \beta_0 + \sum_{n=1}^{N} \nu_{n,q} $$

$$ q^{\star} (\delta) = Dir(\delta | b)  $$


We alternate between the Variational E and M steps, $E$ steps being the ones where we compute the responsibilities $\rho_{n,k1,k2}$ and $\nu_{n,q}$ and the M step is where we update the variational distribution of the parameters $\pi$ and $\delta$. 

We can start with $a= \alpha_0$ and $b=\beta_0$. We can then estimate $\rho_{n,k1,k2}$ from the equation above and also $\nu_{nq}$ and then then product of these two terms to get new responsibility

$$ \delta_{n, k1, k2, q} = \rho_{n, k1, k2} \nu_{nq} $$

and we replace the $\delta_{n, k1, k2, q}$ by $r_{n, k1, k2, q}$. As an alternative Variational EM version 2, we assume the following variational model.

\subsection{Variational EM (model 2)}


We first update the paramaeters $a_{k1, k2}$. 


$$ q(Z, \Lambda, \pi, \delta) = q(Z, \Lambda) q(\pi) q(\delta) $$


\begin{align}
ln q^{\star} (Z, \Lambda)  = E_{\pi, \delta} \left [ ln p(\pi|\alpha_0)
  + ln p(\delta | \beta_0) + ln p(\Lambda | \delta) + ln p(Z | \pi)
  + ln p(D | Z, \Lambda, F, s_{j=1,2,\cdots,J}) \right ] & \\
  \hspace{-2 cm} = E_{\pi, \delta, \Lambda} \left [ ln p(Z | \pi) + ln p(\Lambda | \delta) + ln p(D | Z, \Lambda, F, s_{j=1,2,\cdots,J}) \right] + constant & \\
  = \sum_{n=1}^{N} \sum_{k_1 < k_2}\sum_{q=1}^{Q} z_{n, k1, k2} \lambda_{nq} E_{\pi} \left [ ln (\pi_{k1,k2}) \right ] + \sum_{n=1}^{N} \sum_{q=1}^{Q} \sum_{k_1 < k_2} \lambda_{n,q} z_{n, k1, k2} E_{\delta} \left [ ln (\delta_{q}) \right ] + \sum_{n=1}^{N} \sum_{k_1 < k_2}  \sum_{q} z_{n, k1, k2} \lambda_{nq} \left [ - \sum_{j=1}^{G} ln (s_j) \right . & \\
\qquad \left. - \frac{G}{2} ln (2 \pi) - \sum_{j=1}^{G} \frac{(D_{nj} - qF_{k1,j} - (1-q)F_{k2,j})^2}{2s^2_j} \right]
\end{align}

From here one can get 

$$ q*(Z, \Lambda) \propto \prod_{n=1}^{N} \prod_{k_1 < k_2} \prod_{q=1}^{Q} \delta_{n, k1, k2, q}^{Z_{n,k1,k2} \Lambda_{n,q}} $$

then 

 $$ \delta_{n, k1, k2, q} \propto exp \left (  E_{\pi} \left [ ln (\pi_{k1,k2}) \right] +  E_{\delta} \left [ ln (\delta_{q}) \right ] +  \left [ - \sum_{j=1}^{G} ln (s_j)  - \frac{G}{2} ln (2 \pi) - \sum_{j=1}^{G} \frac{(D_{nj} - qF_{k1,j} - (1-q)F_{k2,j})^2}{2s^2_j} \right] \right )  $$

 $$ \delta_{n, k1, k2, q} \propto exp \left ( \psi_{a_{k1,k2}} - \psi(\sum_{k1 < k2} a_{k1,k2}) +   \psi(b_{q}) - \psi(\sum_{q=1}^{Q} b_{q}) + \left [ - \sum_{j=1}^{G} ln (s_j)  - \frac{G}{2} ln (2 \pi) - \sum_{j=1}^{G} \frac{(D_{nj} - qF_{k1,j} - (1-q)F_{k2,j})^2}{2s^2_j} \right] \right ) $$

The updates for $\pi$ and $\delta$ are same as before. Here also we have the same way of initializing $\pi$ and $\delta$ first, then use $a_{k1,k2}=\alpha_0$ and $b_{q}=\beta_0$ to begin with and estimate $\delta_{n,k1,k2,q}$. In this case, we do not assume independence of the $\Lambda$ and $Z$ variational distributions, so this model is more generalized.



\subsection{M step EM}


We define the parameter 

$$ \theta : = \left (\pi_{k_1,k_2, q}, F, s_{j=1,2,\cdots,J} \right ) $$

We define the complete loglikelihood 

$$ log L_{c} \left (\theta; D, Z, \lambda \right ) = log \pi_{k_1,k_2, q} + log L (D | Z, \lambda, q, F) $$

We take the expectation of this quantity with respect to $\left [ Z, \lambda | D, \theta^{(m)} \right ]$.

\begin{eqnarray}
 Q (\theta | \theta^{(m)}) \propto - \sum_{n=1}^{N} \sum_{k_1 < k_2} \sum_{q} \delta^{(m+1)}_{n, k_1, k_2, q}  \sum_{j} \left [ log s^{(m+1)}_{j} + \frac{(D_{nj} - q F_{k_1,j} - (1-q) F_{k_2,j})^2}{2{s_j^{(m+1)}}^2} \right]
\end{eqnarray}

We try to maximize this quantity with respect to $F$, So, we can take derivative with respect to $F$ and try to solve the resulting normal equation.

This equation, conditional on $\left [ Z, \lambda | D, \theta^{(m)} \right ]$, can be written as 

\begin{eqnarray}
 D_{N \times J} = L_{N \times K} F_{K \times J} + E_{N \times J}
\end{eqnarray}

where 

$$ e_{nj} \sim N(0, s^2_{j}) $$

We define 

$$ D^{'}_{nj} : = \frac{D_{nj}}{s_{j}} $$

If we consider finding the factors on a feature-by-feature basis, we do not need to worry about $s_j$.

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
from above to solve $F$. 

In the same way as we computed $F$ by solving for the normal equation obtained from taking derivative of the function $Q (\theta | \theta^{(m)})$, we take derivative of the latter with respect to $s^2_{j}$ to obtain EM updates of the residual variance terms. Taking the derivative, we obtain the estimate as 

\begin{eqnarray}
\widehat{s_{j}^{(m+1)}}^2 = \frac{1}{N}\sum_{n=1}^{N} \sum_{k_1 < k_2} \sum_{q} \delta^{(m+1)}_{n, k_1, k_2, q} (D_{nj} - q F_{k_1,j} - (1-q) F_{k_2,j})^2
\end{eqnarray}

where the $F$ are the estimated values of the factors from the previous step.

We then continue this procedure described above for multiple iterations.
