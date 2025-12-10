import numpy as np

def compute_reward(num_agents, old_positions, agent_positions, evacuated_agents, deactivated_agents, goal_area):
    rewards = np.zeros(num_agents)
    shaping_factor = 30.0

    # Compute reward for each agent
    for i, (old_pos, new_pos) in enumerate(zip(old_positions, agent_positions)):
        if i in evacuated_agents:
            continue

        if i in deactivated_agents:   # Penalties for each deactivated agent
            rewards[i] = -100.0

        if tuple(new_pos) in goal_area:   # One-time reward for each agent reaching the goal
            rewards[i] = 1000.0
            evacuated_agents.add(i)

            # Compute distance from the agent's old and new positions to its assigned goal.
        old_distances = [np.abs(np.array(old_pos) - np.array(goal)).sum() for goal in goal_area]
        new_distances = [np.abs(np.array(new_pos) - np.array(goal)).sum() for goal in goal_area]

            # Choose the smallest distance (nearest goal) as the effective distance.
        old_distance = min(old_distances)
        new_distance = min(new_distances)
            
            # The shaping reward is the reduction in distance to the goal,
            # multiplied by a scaling factor.
        if new_distance < old_distance:
            reward_shaping = shaping_factor * (old_distance - new_distance)
            rewards[i] = reward_shaping

        else :
            # Base penalty for not reaching the goal, combined with the shaping term.
            rewards[i] = -20.0

        if np.array_equal(old_pos, new_pos):
            rewards[i] = -10.0

    return rewards, evacuated_agents