import random
import numpy as np
from emergent.main import AgentModel
from typing import Tuple, List
import networkx as nx
import matplotlib.pyplot as plt


# --- Unified Initial Data Generation Function ---
def generateInitialData(model: AgentModel):

    max_prior = model["max_prior_value"]
    # Ensure max_prior is positive, default to a small value if not.
    if max_prior <= 0:
        max_prior = 1.0

    return {
        "s_alpha1": random.uniform(
            1e-5, max_prior
        ),  # Use small positive minimum to avoid 0
        "s_beta1": random.uniform(1e-5, max_prior),
        "s_alpha2": random.uniform(1e-5, max_prior),
        "s_beta2": random.uniform(1e-5, max_prior),
    }



# --- Unified Timestep Data Generation Function ---
def generateTimestepData(model: AgentModel):

    graph = model.get_graph()

    num_trials = model["num_trials_per_step"]
    true_probs = model["true_probs"]

    for _node, node_data in graph.nodes(data=True):
        if node_data["a_expectation"] > node_data["b_expectation"]:
            successes = int(np.random.binomial(num_trials, true_probs[0], size=None))
            node_data["a_alpha"] += successes
            node_data["a_beta"] += max(0, num_trials - successes)
            a_denom = node_data["a_alpha"] + node_data["a_beta"]
            node_data["a_expectation"] = (
                node_data["a_alpha"] / a_denom if a_denom > 0 else 0
            )
        else:
            successes = int(np.random.binomial(num_trials, true_probs[1], size=None))
            node_data["b_alpha"] += successes
            node_data["b_beta"] += max(0, num_trials - successes)
            b_denom = node_data["b_alpha"] + node_data["b_beta"]
            node_data["b_expectation"] = (
                node_data["b_alpha"] / b_denom if b_denom > 0 else 0
            )


    trials_per_experiment = model["num_trials_per_step"]

    node_list = list(graph.nodes(data=True))
    actions = {}  # node_id -> action
    experiment_outcomes = {}  # node_id -> (successes, trials_conducted)

    # Phase 1: Choose actions and run experiments
    for node_id, node_data in node_list:
        alpha1, beta1 = node_data["s_alpha1"], node_data["s_beta1"]
        alpha2, beta2 = node_data["s_alpha2"], node_data["s_beta2"]

        ev1 = alpha1 / (alpha1 + beta1) if (alpha1 + beta1) > 0 else 0
        ev2 = alpha2 / (alpha2 + beta2) if (alpha2 + beta2) > 0 else 0

        action = 0 if ev1 >= ev2 else 1
        actions[node_id] = action

        successes = np.random.binomial(trials_per_experiment, true_probs[action])
        experiment_outcomes[node_id] = (successes, trials_per_experiment)

    # Phase 2: Update beliefs based on own and neighbors' experiments
    new_node_attributes = {}
    for node_id, current_node_data in node_list:
        # Initialize belief parameters for update from the current node state
        updated_s_alpha1 = current_node_data["s_alpha1"]
        updated_s_beta1 = current_node_data["s_beta1"]
        updated_s_alpha2 = current_node_data["s_alpha2"]
        updated_s_beta2 = current_node_data["s_beta2"]

        # Update from own experiment
        own_action = actions[node_id]
        own_successes, own_trials = experiment_outcomes[node_id]
        if own_action == 0:
            updated_s_alpha1 += own_successes
            updated_s_beta1 += own_trials - own_successes
        else:
            updated_s_alpha2 += own_successes
            updated_s_beta2 += own_trials - own_successes

        # Update from neighbors' experiments
        for neighbor_id in graph.neighbors(node_id):
            if neighbor_id in actions and neighbor_id in experiment_outcomes:
                neighbor_action = actions[neighbor_id]
                neighbor_successes, neighbor_trials = experiment_outcomes[
                    neighbor_id
                ]
                if neighbor_action == 0:
                    updated_s_alpha1 += neighbor_successes
                    updated_s_beta1 += neighbor_trials - neighbor_successes
                else:
                    updated_s_alpha2 += neighbor_successes
                    updated_s_beta2 += neighbor_trials - neighbor_successes

        new_node_attributes[node_id] = {
            "s_alpha1": updated_s_alpha1,
            "s_beta1": updated_s_beta1,
            "s_alpha2": updated_s_alpha2,
            "s_beta2": updated_s_beta2,
        }


    for node_id, data_to_set in new_node_attributes.items():
        for key, value in data_to_set.items():
            graph.nodes[node_id][key] = value

    model.set_graph(graph)


# --- Model Construction ---
def constructModel() -> AgentModel:
    model = AgentModel()

    # Define all parameters that might be used by any variation.
    # Emergent will allow editing these in the UI.
    model.update_parameters(
        {
            # Common parameters editable in UI, used for graph setup by Emergent
            "num_nodes": 10,
            "graph_type": "complete",  # Options: "complete", "wheel", "cycle"
            # Parameter to control which variation's logic is run.
            # This will be updated by Emergent when the user selects a variation.
            "true_probs": [0.3, 0.7],  # [prob_methodology1, prob_methodology2]
            "num_trials_per_step": 10,
            "max_prior_value": 4.0,
        }
    )


    # Set the unified functions
    model.set_initial_data_function(generateInitialData)
    model.set_timestep_function(generateTimestepData)

    return model


if __name__ == "__main__":
    model = constructModel()
    model.initialize_graph()
    print(model.get_graph().nodes(data=True))