# Zollman Bandit Model (2010 Variation)

A bandit model by [Kevin Zollman](https://www.kevinzollman.com/) examining how network connectivity affects the reliability and speed of scientific consensus. This is the **2010 variation** emphasizing the trade-off between community reliability and convergence speed. View the full paper <a href="https://www.kevinzollman.com/uploads/5/0/3/6/50361245/zollman_-_transient_diversity.pdf" target="_blank" rel="noopener noreferrer">here</a>.

## Abstract

> Increasingly, epistemologists are becoming interested in social structures and their effect on epistemic enterprises, but little attention has been paid to the proper distribution of experimental results among scientists. This paper will analyze a model first suggested by two economists, which nicely captures one type of learning situation faced by scientists. The results of a computer simulation study of this model provide two interesting conclusions. First, in some contexts, a community of scientists is, as a whole, more reliable when its members are less aware of their colleagues' experimental results. Second, there is a robust trade-off between the reliability of a community and the speed with which it reaches a correct conclusion.

## Model Implementation (2010)

This variation implements an enhanced bandit model where scientists:

- Use Bayesian updating with Beta distributions for two competing methodologies
- Make decisions based on expected values of success probabilities
- Learn from both their own experiments and neighbors' results
- Demonstrate the reliability-speed trade-off in different network structures

### Key Features

- **Learning Mechanism**: Bayesian updating with Beta-Binomial conjugate priors
- **Decision Rule**: Choose methodology with higher expected value
- **Information Sharing**: Agents observe neighbors' experimental outcomes
- **Network Effects**: Different structures (complete, cycle, wheel) affect information flow

## Model Parameters

- `Num Nodes`: Size of network (default: 10)
- `A Objective`: True success probability of methodology A (default: 0.5)
- `B Objective`: True success probability of methodology B (default: 0.499)
- `Num Trials Per Step`: Number of trials per experiment (default: 10), determines how much evidence is gathered each timestep.
- `Max Prior Value`: Maximum value for initial Beta distribution parameters (default: 4.0), affects the interaction between priors and evidence.
- `Graph Type`: Network structure - "complete", "cycle", or "wheel"

## Agent Attributes

Each scientist maintains:

- `a_expectation`, `b_expectation`: Expected success rates for each methodology. Scientists choose the methodology they currently think has a higher success rate.

## Key Differences from 2007 Variation

In the 2007 version, the success rate of A was known. In this version, agents don't know the success rate of either methodology and gather evidence on both.

**Click the 'Visualizations' tab to get started.**
