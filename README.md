# Zollman Bandit Model (2010 Variation)
A bandit model by [Kevin Zollman](https://www.kevinzollman.com/) examining how network connectivity affects the reliability and speed of scientific consensus. This is the **2010 variation** emphasizing the trade-off between community reliability and convergence speed. View the full paper [here](https://www.kevinzollman.com/uploads/5/0/3/6/50361245/zollman_-_communication_structure.pdf).

## Abstract
> Increasingly, epistemologists are becoming interested in social structures and their effect on epistemic enterprises, but little attention has been paid to the proper distribution of experimental results among scientists. This paper will analyze a model first suggested by two economists, which nicely captures one type of learning situation faced by scientists. The results of a computer simulation study of this model provide two interesting conclusions. First, in some contexts, a community of scientists is, as a whole, more reliable when its members are less aware of their colleagues' experimental results. Second, there is a robust trade-off between the reliability of a community and the speed with which it reaches a correct conclusion.

## Model Implementation (2010)
This variation implements an enhanced bandit model where scientists:
- Use Bayesian updating with Beta distributions for two competing methodologies
- Make decisions based on expected values of success probabilities
- Learn from both their own experiments and neighbors' results
- Demonstrate the reliability-speed trade-off in different network structures

### Key Features
- **Enhanced Learning**: More sophisticated Bayesian belief updating
- **Dual Methodology Tracking**: Separate Beta distributions for each research method
- **Network Learning**: Agents update beliefs based on neighbors' experimental outcomes
- **Reliability Focus**: Emphasis on long-term accuracy vs. quick consensus

## Model Parameters
* `num_nodes`: Size of network (default: 15)
* `true_probs_2010`: True success probabilities [methodology1, methodology2] (default: [0.3, 0.7])
* `num_trials_per_step_2010`: Number of trials per experimental round (default: 10)
* `max_prior_value_2010`: Maximum value for initial Beta distribution parameters (default: 4.0)
* `graph_type`: Network structure - "complete", "cycle", or "wheel"

## Agent Attributes
Each scientist maintains:
- `s_alpha1`, `s_beta1`: Beta distribution parameters for methodology 1
- `s_alpha2`, `s_beta2`: Beta distribution parameters for methodology 2

## Key Differences from 2008 Variation
- **Methodology Focus**: Two distinct research methodologies rather than A/B options
- **Learning Scope**: Agents learn from all neighbors' experiments, not just their own
- **Parameter Structure**: Streamlined belief representation with separate methodology tracking
- **Experimental Design**: Higher trial counts and different default probability settings