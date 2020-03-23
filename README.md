# Monte Carlo univariate distribution sampler

This repository contains several continuous and discrete univariate distributions. 
It uses MRG32k3a generator to create uniform samples. These uniform samples are then 
transformed to yield a sample from any supported distribution. A space system is used 
to represent the domain of the samples.

## Supported distributions

The following distributions are supported. 

### Continous

- Beta
- Cauchy
- Exponential
- Fisher-Snedecor
- Gamma
- Gumbel
- Laplace
- Log normal
- Logistic
- Normal
- Pareto
- Students
- Uniform
- Wald
- Weibull

### Discrete

- Bernoulli
- Binomial
- DPhaseType
- DUniform
- Geometric
- Hypergeometric
- Negative binomial
- Poisson

## References

1. [Handbook of Monte Carlo](https://people.smp.uq.edu.au/DirkKroese/montecarlohandbook/)
2. https://www.hongliangjie.com/2012/12/19/how-to-generate-gamma-random-variables/
3. https://pubsonline.informs.org/doi/pdf/10.1287/opre.47.1.159