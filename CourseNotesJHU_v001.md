---
title: "Caffo Stats and Regression Summary"
author: "davegoblue"
date: "April 5, 2016"
output: html_document
---



## Background  
This document is to summarize some of the most salient themes from a few e-books written by Brian Caffo from the Biostats department of JHU.  
  
* Statistical Inference for Data Science  
* Regression Models for Data Science in R  
* Advanced Linear Models for Data Science (book only partially written)  
  
The main objective of these notes is to shore up my background in these areas and to create a reference document for future use.  
  
## Statistical Inference for Data Science  
### Chaper 1: Introduction  
Key concerns associated with statistical inferences include:  
  
* Is the sample representative of the population we want to draw inferences about?  
* Are there any of "known and observed", "known and unobserved", or "unknown and unobserved" variables that may contaminate our conclusions?  
* Is there systematic bias due to missing data or study design?  
* What randomness exists in the data, and what adjustments are merited?  
    - Randomness may be dur to experiment design or implicit based on aggregations from an unknown process  
* Are we trying to estimate an underlying mechanistic model?  
  
Statistical inference is the process of navigating the assumptions, tools, and concerns to consider how to draw conclusions from the data.  Typical goals of inference include:  
  
* Estimate a population quantity and the associated uncertainty (e.g., election polls)  
* Determine whether a population quantity hits a benchmark (e.g., treatment effectiveness in a clinical trial)  
* Infer a mechanistic relationship when quantities are measured with noise (e.g., slope for Hooke's Law)  
* Predict the impact of a policy (e.g., how would reducing pollution impact asthma)  
* Discuss the probability that something occurs  
  
There are many tools of the trade, including but not limited to:  
  
1. Randomization: balancing out potentially confounding variables  
2. Random Sampling: obtain data that is representative of the full population  
3. Sampling Models: creating a model for sampling (often iid - identical and independently distributed)  
4. Hypothesis Testing: decision making in the presence of uncertainty  
5. Confidence Intervals: quantifying uncertainty in estimates  
6. Probability Models: formal connection between data and the population of interest (often assumed or approximated)  
7. Study Design: Deisgning an experiment to minimize biases and variability  
8. Non-parametric Bootstrapping: Creating inferences from the data with minimal Probability Model assumptions  
9. Permutation: Using permutations (randomization, exchanges) to perform inferences  
  
Lastly, there are several different styles of thinking about probability.  While the frequentist approach is the most common (and covered first), success often comes from leveraging each style, and hybrids, as needed:  
  
* Frequency probability: long run proportion of times events occur with repeated iid  
* Frequency inference: control for error rates using frequency interpretations of probabilities (keep overall level of mistakes to a "tolerable" level)  
* Bayesian probability: probability calculus of beliefs given that beliefs follow certain rules  
* Bayesian inference: using Bayesian probability to perform inference ("given my subjective beliefs and the objective data, what should I believe now")  
  
### Chaper 2: Probability  
