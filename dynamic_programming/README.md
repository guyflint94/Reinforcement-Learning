# Dynamic programming on GridWorld

## Files

- `DP_GridWorld.py`: a GridWorld MDP with slip, policy evaluation and greedy policy improvement.

## Requirements

Standard library only.

## Running

    python DP_GridWorld.py

The script runs one step of policy iteration on a 4x4 grid (goal at (3,3), obstacle at (1,1), slip 0.1, gamma 0.9) and prints:

- `V(goal)` is 0 under the uniform policy, because the goal is terminal.
- Under the uniform policy, V is about -199.6 at (0,0) and -31.9 at (3,2) and (2,3).
- After one improvement step, every state's value is at least as large as before.
- The first moves from (0,0), (3,2) and (2,3) are up, up and right.
