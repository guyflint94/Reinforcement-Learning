import random

class GridWorld:
    def __init__(self, width, height, start_state, goal_state, slip=0.1, obstacles=None):
        self.width = width
        self.height = height
        self.start_state = start_state
        self.goal_state = goal_state
        self.slip = slip
        if obstacles is not None:
            self.obstacles = obstacles
        else:
            self.obstacles = []
        self.actions = {'up': [0,1], 'down': [0,-1], 'left':[-1,0], 'right':[1,0]}
        self.curr_state = self.start_state


    # gather ALL possible states in a list
    def all_states(self):
        all_states = []
        for i in range(self.width):
            for j in range(self.height):
                all_states.append((i, j))
        return all_states


    # checks in general that the environment is valid. currently unused
    def is_valid(self, curr_state):
        if self.width <= 0 or self.height <= 0 :  # check boundaries are non-negative
            return False
        elif self.start_state == self.goal_state:  # check start is not goal
            return False
        elif self.start_state not in self.all_states():  # check start_state is in grid
            return False
        elif self.goal_state not in self.all_states():  # check goal_state is in grid
            return False
        elif not ((0 <= curr_state[0] < self.width) and (0 <= curr_state[1] < self.height)) :
            return False
        elif curr_state in self.obstacles:
            return False

        for obstacle in self.obstacles:  # check that each obstacle is in grid
            if obstacle not in self.all_states():
                return False

        return True


    # reset the episode (reset environment)
    def reset(self):
        self.curr_state = self.start_state
        return self.curr_state


    # create the transition dynamics
    def transitions(self, curr_state, action):
        outcomes = {}

        for actual_action, movement in self.actions.items():  # 'up', (0,1)

            if actual_action == action:
                probability = 1 - self.slip + self.slip / 4
            else:
                probability = self.slip / 4

            # Calculate next_state using actual_action
            next_state = (curr_state[0] + movement[0], curr_state[1] + movement[1])

            # Handle walls and obstacles, goal, or valid move
            if next_state in self.obstacles or not (0 <= next_state[0] < self.width) or not (0 <= next_state[1] < self.height) :
                reward = -50
                next_state = curr_state

            elif next_state == self.goal_state:
                reward = 100

            else:
                reward = -1

            key = (next_state, reward)
            outcomes[key] = outcomes.get(key, 0) + probability  # sums probability in case 2 different actions led to the same state

        return outcomes


    def step(self, state, action):
        outcomes = self.transitions(state, action)

        possibilities = list(outcomes.keys())
        probabilities = list(outcomes.values())

        next_state, reward = random.choices(possibilities, weights=probabilities, k=1)[0]

        done = next_state == self.goal_state

        return next_state, reward, done


    def act(self, action):
        self.curr_state, reward, done = self.step(self.curr_state, action)

        return self.curr_state, reward, done


# in this function we track our experience in one episode
def generate_episode(env, max_steps=50):
    state = env.reset()

    trajectory = []  # a list of tuples [(state, action, reward), ...]
    lst_of_actions = list(env.actions.keys())

    for i in range(max_steps):
        action = random.choice(lst_of_actions)  # take a random action
        next_state, reward, done = env.step(state, action)
        trajectory.append((state, action, reward))
        state = next_state

        if done:
            return trajectory
    return trajectory

# policy_evaluation(env, policy, gamma, theta) -> V
# we are going to evaluate each state sum(pi(a|s) * sum(P(s', r | s, a) * (r + gamma*values(next_state))
# policy is pi(a|s)
"""
Loop over all states.
For each state, loop over all actions.
For each action, loop over all possible transitions.
Multiply the policy probability by the transition probability and the discounted return.
Sum everything to calculate values[state].
Repeat until the values converge within theta.
"""
def policy_evaluation(env, policy, gamma, theta):
    values = {}

    # Initialize all state values to zero
    for st in env.all_states():
        values[st] = 0

    while True:
        delta = 0

        for st in env.all_states():
            if st == env.goal_state or st in env.obstacles:
                continue

            old_value = values[st]
            new_value = 0

            for i, act in enumerate(env.actions):
                for (next_state, rew), prob in env.transitions(st, act).items():
                    if next_state == env.goal_state:
                        new_value += policy[st][i] * prob * rew
                    else:
                        new_value += (policy[st][i] * prob * (rew + gamma * values[next_state]))

            values[st] = new_value

            delta = max(delta, abs(old_value - new_value))

        if delta < theta:
            break

    return values


# in each state pick the action which will give the highest expected return
# pi'(s) = argmax(a) * sum(P(s', r | s, a) * (r + gamma * values(s'))
"""
It receives:
- env: The environment (states, actions, transitions, rewards).
- values: The value function V(s), which tells us how good each state is.
- gamma: Discount factor.
- it needs to return: pi(s)=best action in state s
"""
def greedy_policy(env, values, gamma):

    grd_policy = {}  # key: [4 values]  1.0 probability for the best action, 0.0 for the rest

    for st in env.all_states():
        if st == env.goal_state or st in env.obstacles:
            continue
        st_returns = []

        for act in env.actions.keys():
            calc_val = 0
            for (next_state, reward), prob in env.transitions(st, act).items():

                if next_state == env.goal_state:
                    calc_val += prob * reward
                else:
                    calc_val += prob * (reward + gamma * values[next_state])

            st_returns.append(calc_val)  # [val_up, val_down, val_left, val_right]

        best_act = max(st_returns)
        best_act_pos = st_returns.index(best_act)
        probs = []
        for i in range(len(st_returns)):
            if  i == best_act_pos:
                probs.append(1)
            else:
                probs.append(0)

        grd_policy[st] = probs

    return grd_policy


if __name__ == "__main__":
    gamma = 0.9
    theta = 1e-8
    width = 4
    height = 4
    start_state = (0, 0)
    goal_state = (3,3)
    obstacles = [(1, 1)]
    env = GridWorld(width, height, start_state, goal_state, slip=0.1, obstacles=obstacles)
    uniform_policy = {}
    for st in env.all_states():
        uniform_policy[st] = [0.25, 0.25, 0.25, 0.25]
    V_old = policy_evaluation(env, uniform_policy, gamma, theta)
    new_policy = greedy_policy(env, V_old, gamma)
    V_new = policy_evaluation(env, new_policy, gamma, theta)
    print(f"Old State Values: V_old[goal_state]: {V_old[goal_state]:.2f}, V_old[(0,0)]: {V_old[(0,0)]:.2f}, V_old[(3,2)]: {V_old[(3,2)]:.2f}, V_old[(2,3)]: {V_old[(2,3)]:.2f}", '\n')
    print(f"New State Values: V_new[start_state]: {V_new[start_state]:.2f}, V_new[(3,2)]: {V_new[(3,2)]:.2f}, V_new[(2,3)]: {V_new[(2,3)]:.2f}", '\n')
    for st in V_old:
        print(f"{st}: {V_old[st] <= V_new[st]}")
    print('\n')
    for s in [(0, 0), (3, 2), (2, 3)]:
        print(s, list(env.actions)[new_policy[s].index(1)])