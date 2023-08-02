import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import numpy as np
import os

from mdp import MDP, solve_MDP, compute_marginal_distribution, OptiDICE, policy_evaluation, generate_trajectory, compute_MLE_MDP


# Define four-rooms MDP
class Action:
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

WIDTH, HEIGHT = 11, 11
S = WIDTH * HEIGHT + 1
A = 4  # UP, DOWN, LEFT, RIGHT
gamma = 0.99

wall = {(5, 0), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7), (5, 9), (5, 10), (0, 5), (2, 5), (3, 5), (4, 5), (6, 4), (7, 4), (9, 4), (10, 4)}
pos_to_state = {}
state_to_pos = {}
for x in range(WIDTH):
    for y in range(HEIGHT):
        s = y * WIDTH + x
        pos_to_state[(x, y)] = s
        state_to_pos[s] = (x, y)

def move(s, a):
    s_x, s_y = state_to_pos[s]
    if a == Action.UP: s_y += 1
    elif a == Action.DOWN: s_y -= 1
    elif a == Action.LEFT: s_x -= 1
    elif a == Action.RIGHT: s_x += 1
    else: raise ValueError(f"a={a}")
    if (s_x, s_y) in wall: return s
    s_x, s_y = np.clip(s_x, 0, WIDTH - 1), np.clip(s_y, 0, HEIGHT - 1)
    return pos_to_state[(s_x, s_y)]

initial_state_pos = (1, HEIGHT - 2)
goal_state_pos = (WIDTH - 2, 1)

T = np.zeros((S, A, S))
R = np.zeros((S, A))
for s in range(S - 1):
    for a in range(A):
        EPS = 0.1
        T[s, a, move(s, 0)] += EPS
        T[s, a, move(s, 1)] += EPS
        T[s, a, move(s, 2)] += EPS
        T[s, a, move(s, 3)] += EPS
        T[s, a, move(s, a)] += 1 - np.sum(T[s, a, :])
        assert np.isclose(np.sum(T[s, a]), 1), f"{T[s, a, :]}"
T[S - 1, :, S - 1] = 1
T[pos_to_state[goal_state_pos], :, :] = 0 
T[pos_to_state[goal_state_pos], :, S - 1] = 1  # goal state to absorbing state
R[pos_to_state[goal_state_pos], :] = 1

true_mdp = MDP(S, A, T, R, gamma)
true_mdp.initial_state = pos_to_state[initial_state_pos]

# Compute an optimal policy
_, V_opt, Q_opt = solve_MDP(true_mdp)
pi_opt = np.zeros((S, A))
for s in range(S):
    best_a = np.argmax(Q_opt[s, :])
    best_a_indices = np.where(np.isclose(Q_opt[s, :], Q_opt[s, best_a]))[0]
    pi_opt[s, best_a_indices] = 1. / len(best_a_indices)

# Prepare for the data-collection policy
pi_dir = np.random.dirichlet(np.ones(A), S)
pi_b = pi_opt * 0.5 + pi_dir * 0.5

# Collect trajectories
trajectory_all = generate_trajectory(0, true_mdp, pi_b, num_episodes=50, max_timesteps=50)
mle_mdp, N_all = compute_MLE_MDP(S, A, R, gamma, trajectory_all)
mle_mdp.initial_state = pos_to_state[initial_state_pos]

# Run OptiDICE
pi_optidice = OptiDICE(mle_mdp, pi_b, None, alpha=0.0005)[0]


## Visualize FourRooms, d^pi, and pi
def draw(mdp, pi, filepath):
    COLOR_WALL = '#990000'
    COLOR_INITIAL_STATE = '#ff9966'
    COLOR_GOAL_STATE = '#99ff99'
    COLOR_ARROW = '#bd534d'

    # Draw background
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, aspect='equal')
    for x in range(-1, WIDTH + 1):
        ax.add_patch(patches.Rectangle((x, -1), 1, 1, linewidth=0, facecolor=COLOR_WALL))
        ax.add_patch(patches.Rectangle((x, HEIGHT), 1, 1, linewidth=0, facecolor=COLOR_WALL))
    for y in range(-1, HEIGHT + 1):
        ax.add_patch(patches.Rectangle((-1, y), 1, 1, linewidth=0, facecolor=COLOR_WALL))
        ax.add_patch(patches.Rectangle((WIDTH, y), 1, 1, linewidth=0, facecolor=COLOR_WALL))
    for x, y in wall:
        ax.add_patch(patches.Rectangle((x, y), 1, 1, linewidth=0, facecolor=COLOR_WALL))

    cm = plt.get_cmap('Blues') 
    cNorm  = colors.Normalize(vmin=0, vmax=1)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)

    d_pi = compute_marginal_distribution(mdp, pi).reshape(mdp.S, mdp.A)
    d_pi_s = np.sum(d_pi, axis=1)

    # Draw d_pi
    for s in range(mdp.S - 1):
        x, y = state_to_pos[s]
        if (x, y) in wall: continue
        d_s = d_pi_s[s] / np.max(d_pi_s[:S-1])  # 0 ~ 1
        ax.add_patch(patches.Rectangle((x, y), 1, 1, linewidth=1, edgecolor=None, facecolor=scalarMap.to_rgba(1 - np.power(1 - d_s, 2))))

    ax.add_patch(patches.Rectangle(initial_state_pos, 1, 1, linewidth=3, edgecolor=COLOR_INITIAL_STATE, fill=False))
    ax.add_patch(patches.Rectangle((WIDTH - 2, 1), 1, 1, linewidth=3, edgecolor=COLOR_GOAL_STATE, fill=False))

    # Draw policy
    for s in range(mdp.S - 1):
        x, y = state_to_pos[s]
        if (x, y) in wall or (x, y) == (WIDTH - 2, 1): continue
        for a in range(mdp.A):
            if pi[s, a] > 1e-10:
                if a == Action.UP:
                    # ax.arrow(x + 0.5, y + 0.5, 0, +0.45 * pi[s, a], head_width=0.05, head_length=0.05, fc='k', ec='k') alpha=min(pi[s, a] * 2, 1)
                    ax.arrow(x + 0.5, y + 0.5, 0, 0.225, head_width=0.35, head_length=0.25, fc=COLOR_ARROW, ec=COLOR_ARROW, alpha=pi[s, a])
                elif a == Action.DOWN:
                    ax.arrow(x + 0.5, y + 0.5, 0, -0.225, head_width=0.35, head_length=0.25, fc=COLOR_ARROW, ec=COLOR_ARROW, alpha=pi[s, a])
                elif a == Action.LEFT:
                    ax.arrow(x + 0.5, y + 0.5, -0.225, 0, head_width=0.35, head_length=0.25, fc=COLOR_ARROW, ec=COLOR_ARROW, alpha=pi[s, a])
                elif a == Action.RIGHT:
                    ax.arrow(x + 0.5, y + 0.5, 0.225, 0, head_width=0.35, head_length=0.25, fc=COLOR_ARROW, ec=COLOR_ARROW, alpha=pi[s, a])

    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()


os.makedirs('fourrooms', exist_ok=True)
draw(true_mdp, pi_b, "fourrooms/true_mdp_behavior.png")
filepath = f"fourrooms/mle_mdp_optidice.png"
draw(mle_mdp, pi_optidice, filepath)
filepath = f"fourrooms/true_mdp_optidice.png"
draw(true_mdp, pi_optidice, filepath)
print(f'est_perf={policy_evaluation(mle_mdp, pi_optidice)[0][pos_to_state[initial_state_pos]]:.3f},',
      f'true_perf={policy_evaluation(true_mdp, pi_optidice)[0][pos_to_state[initial_state_pos]]:.3f}')
