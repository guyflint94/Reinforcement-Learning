# Reinforcement Learning

Exercises from learning reinforcement learning from the ground up, in preparation for multi-agent PPO (MAPPO).

## Layout

    dynamic_programming/
        DP_GridWorld.py   GridWorld MDP with slip, policy evaluation and greedy policy improvement

## Running

Pure Python, no extra packages needed:

    python dynamic_programming/DP_GridWorld.py

The script runs one step of policy iteration on a 4x4 grid (goal at (3,3), obstacle at (1,1), slip 0.1, gamma 0.9) and prints:

- `V(goal)` is 0 under the uniform policy, because the goal is terminal.
- The values under the uniform policy are about -199.6 at (0,0) and -31.9 at (3,2) and (2,3).
- After one improvement step, every state's value is at least as large as before.
