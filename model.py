import numpy as np
import networkx as nx
import random
from emergent.main import AgentModel


def generateInitialData(model: AgentModel):
    """
    Initializes a scientist (node) with random priors.
    """
    # Retrieve max_prior_value from model parameters (default 4.0)
    max_prior = (
        model["max_prior_value"]
        if "max_prior_value" in model.list_parameters()
        else 4.0
    )

    # Initialize Beta distribution parameters
    alpha1 = np.random.uniform(0, max_prior)
    beta1 = np.random.uniform(0, max_prior)
    alpha2 = np.random.uniform(0, max_prior)
    beta2 = np.random.uniform(0, max_prior)

    # Calculate initial expectations
    ev1 = alpha1 / (alpha1 + beta1) if (alpha1 + beta1) > 0 else 0
    ev2 = alpha2 / (alpha2 + beta2) if (alpha2 + beta2) > 0 else 0

    return {
        # Beta distribution parameters for Method A (0) and Method B (1)
        "alpha1": alpha1,
        "beta1": beta1,
        "alpha2": alpha2,
        "beta2": beta2,
        # Expected values (expectations)
        "a_expectation": ev1,
        "b_expectation": ev2,
        # Tracking variables
        "current_action": "none",
        "last_successes": 0,
        "last_trials": 0,
    }


def generateTimestepData(model: AgentModel):
    """
    Runs one iteration of the Zollman Bandit model.
    """
    graph = model.get_graph()

    # Retrieve simulation parameters
    a_objective = model["a_objective"]
    b_objective = model["b_objective"]

    # --- Phase 1: Actions & Experiments ---
    # We calculate results for all agents first to ensure synchronous updating.
    # This prevents an agent from updating beliefs based on a neighbor's *new* result
    # from the current timestep before the neighbor has actually finished.
    step_results = {}

    for node_id, node_data in graph.nodes(data=True):
        # 1. Get Expected Values (use stored expectations for consistency)
        ev1 = node_data["a_expectation"]
        ev2 = node_data["b_expectation"]

        # 2. Greedy Action Selection: Choose method with higher EV
        if ev1 > ev2:
            action = "a"  # Choose Method A
        else:
            action = "b"  # Choose Method B

        # 3. Determine objective probability based on chosen action
        current_objective_prob = a_objective if action == "a" else b_objective

        # 4. Run Experiment (Sample from Binomial Distribution)
        successes = np.random.binomial(1, current_objective_prob)

        # 5. Store data for the update phase
        step_results[node_id] = (action, successes, 1)

        # Update node state for external tracking/visualization
        node_data["current_action"] = action
        node_data["last_successes"] = successes
        node_data["last_trials"] = 1

    # --- Phase 2: Bayesian Belief Updating ---
    for node_id, node_data in graph.nodes(data=True):

        # Update beliefs based on OWN results
        own_action, own_successes, own_trials = step_results[node_id]
        if own_action == "a":
            node_data["alpha1"] += own_successes
            node_data["beta1"] += own_trials - own_successes
        else:
            node_data["alpha2"] += own_successes
            node_data["beta2"] += own_trials - own_successes

        # Update beliefs based on NEIGHBORS' results (Social Learning)
        for neighbor_id in graph.neighbors(node_id):
            neigh_action, neigh_successes, neigh_trials = step_results[neighbor_id]

            if neigh_action == "a":
                node_data["alpha1"] += neigh_successes
                node_data["beta1"] += neigh_trials - neigh_successes
            else:
                node_data["alpha2"] += neigh_successes
                node_data["beta2"] += neigh_trials - neigh_successes

        # Recalculate expectations after updating beliefs
        alpha1_sum = node_data["alpha1"] + node_data["beta1"]
        alpha2_sum = node_data["alpha2"] + node_data["beta2"]
        node_data["a_expectation"] = (
            node_data["alpha1"] / alpha1_sum if alpha1_sum > 0 else 0
        )
        node_data["b_expectation"] = (
            node_data["alpha2"] / alpha2_sum if alpha2_sum > 0 else 0
        )

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
    model.update_parameters(
        {
            "num_nodes": 10,
            "graph_type": "cycle",  # Can be 'cycle', 'wheel', 'complete'
            "a_objective": 0.5,  # Probability of success for Method A (Action 0)
            "b_objective": 0.499,  # Probability of success for Method B (Action 1)
            "max_prior_value": 4.0,  # Initial strength of beliefs (Beta params)
        }
    )

    return model
