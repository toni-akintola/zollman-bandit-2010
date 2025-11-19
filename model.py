import numpy as np
import networkx as nx
import random
from emergent.main import AgentModel

def generateInitialData(model: AgentModel):
    """
    Initializes a scientist (node) with random priors.
    """
    # Retrieve max_prior_value from model parameters (default 4.0)
    max_prior = model["max_prior_value"] if "max_prior_value" in model.list_parameters() else 4.0
    
    return {
        # Beta distribution parameters for Method A (0) and Method B (1)
        "alpha1": np.random.uniform(0, max_prior),
        "beta1": np.random.uniform(0, max_prior),
        "alpha2": np.random.uniform(0, max_prior),
        "beta2": np.random.uniform(0, max_prior),
        # Tracking variables
        "current_action": -1, 
        "last_successes": 0,
        "last_trials": 0
    }


def generateTimestepData(model: AgentModel):
    """
    Runs one iteration of the Zollman Bandit model.
    """
    graph = model.get_graph()
    
    # Retrieve simulation parameters
    trials_per_experiment = model["trials_per_experiment"]
    a_objective = model["a_objective"]
    b_objective = model["b_objective"]
    
    # --- Phase 1: Actions & Experiments ---
    # We calculate results for all agents first to ensure synchronous updating.
    # This prevents an agent from updating beliefs based on a neighbor's *new* result
    # from the current timestep before the neighbor has actually finished.
    step_results = {} 
    
    for node_id, node_data in graph.nodes(data=True):
        # 1. Calculate Expected Value (Mean of Beta Distribution: alpha / (alpha + beta))
        ev1 = node_data["alpha1"] / (node_data["alpha1"] + node_data["beta1"])
        ev2 = node_data["alpha2"] / (node_data["alpha2"] + node_data["beta2"])
        
        # 2. Greedy Action Selection: Choose method with higher EV
        if ev1 > ev2:
            action = 0 # Choose Method A
        else:
            action = 1 # Choose Method B
        
        # 3. Determine objective probability based on chosen action
        current_objective_prob = a_objective if action == 0 else b_objective

        # 4. Run Experiment (Sample from Binomial Distribution)
        successes = np.random.binomial(trials_per_experiment, current_objective_prob)
        
        # 5. Store data for the update phase
        step_results[node_id] = (action, successes, trials_per_experiment)
        
        # Update node state for external tracking/visualization
        node_data["current_action"] = action
        node_data["last_successes"] = successes
        node_data["last_trials"] = trials_per_experiment

    # --- Phase 2: Bayesian Belief Updating ---
    for node_id, node_data in graph.nodes(data=True):
        
        # Update beliefs based on OWN results
        own_action, own_successes, own_trials = step_results[node_id]
        if own_action == 0:
            node_data["alpha1"] += own_successes
            node_data["beta1"] += (own_trials - own_successes)
        else:
            node_data["alpha2"] += own_successes
            node_data["beta2"] += (own_trials - own_successes)
            
        # Update beliefs based on NEIGHBORS' results (Social Learning)
        for neighbor_id in graph.neighbors(node_id):
            neigh_action, neigh_successes, neigh_trials = step_results[neighbor_id]
            
            if neigh_action == 0:
                node_data["alpha1"] += neigh_successes
                node_data["beta1"] += (neigh_trials - neigh_successes)
            else:
                node_data["alpha2"] += neigh_successes
                node_data["beta2"] += (neigh_trials - neigh_successes)

    # Commit changes to the model
    model.set_graph(graph)


def constructModel() -> AgentModel:
    """
    Constructs the Zollman Bandit Model instance with default parameters.
    """
    model = AgentModel()
    
    # Set the logic hooks
    model.set_initial_data_function(generateInitialData)
    model.set_timestep_function(generateTimestepData)
    
    # Set default parameters
    model.update_parameters({
        "num_nodes": 10,
        "graph_type": "cycle",            # Can be 'cycle', 'wheel', 'complete'
        "a_objective": 0.5,               # Probability of success for Method A (Action 0)
        "b_objective": 0.499,             # Probability of success for Method B (Action 1)
        "trials_per_experiment": 1000,    # Number of trials per time step
        "max_prior_value": 4.0            # Initial strength of beliefs (Beta params)
    })
    
    return model