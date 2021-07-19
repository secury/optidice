import os
import time
from collections import defaultdict

import numpy as np

from mdp import (OptiDICE, compute_MLE_MDP, generate_baseline_policy,
                 generate_random_mdp, generate_trajectory, policy_evaluation,
                 solve_MDP)

np.set_printoptions(precision=3, suppress=True, linewidth=250)

def run(seed, optimality, trajectory_num_episodes):
    print('===================================', flush=True)
    print('seed={} / optimality={} / trajectory_num_episodes={}'.format(seed, optimality, trajectory_num_episodes))
    result_dir = 'result_random_mdp'
    os.makedirs(result_dir, exist_ok=True)
    result_filepath = "{}/optimality_{}_N_trajectory_{}.npy".format(result_dir, optimality, trajectory_num_episodes)

    if os.path.exists(result_filepath):
        result = np.load(result_filepath, allow_pickle=True)[()]
    else:
        result = {}
    if result.get(seed) is None:
        result[seed] = {}

    # Generate a random MDP
    start_time = time.time()
    mdp = generate_random_mdp(seed, S=50, A=4, gamma=0.95)
    pi_b = generate_baseline_policy(seed, mdp, optimality)
    print('MDP constructed ({:6.3f} secs)'.format(time.time() - start_time))

    # Construct MLE MDP
    start_time = time.time()
    trajectory_all = generate_trajectory(seed, mdp, pi_b, num_episodes=trajectory_num_episodes)
    mdp_all, N_all = compute_MLE_MDP(mdp.S, mdp.A, mdp.R, mdp.gamma, trajectory_all)
    print('MLE MDP constructed ({:6.3f} secs)'.format(time.time() - start_time))

    # In order to normalize performance
    if result[seed].get('V_opt') is None:
        start_time = time.time()
        V_opt = solve_MDP(mdp)[1][0]  # 1: optimal policy
        V_b = policy_evaluation(mdp, pi_b)[0][0]  # 0: baseline policy
        V_unif = policy_evaluation(mdp, np.ones((mdp.S, mdp.A)) / mdp.A)[0][0]
        result[seed]['V_opt'] = V_opt
        result[seed]['V_b'] = V_b
        result[seed]['V_unif'] = V_unif
        print('V_opt=%.3f / V_unif=%.3f / V_b=%.3f (ratio=%f) (%6.3f secs)' % (V_opt, V_unif, V_b, (V_b - V_unif) / (V_opt - V_unif), time.time() - start_time))

    # Baseline: solve MLE MDP without regularization
    if result[seed].get('V_mbrl') is None:
        start_time = time.time()
        pi_all, _, _ = solve_MDP(mdp_all)
        V_mbrl = policy_evaluation(mdp, pi_all)[0][0]
        result[seed]['V_mbrl'] = V_mbrl
        print('{:20s}: {:.3f} ({:6.3f} secs)'.format("MBRL", V_mbrl, time.time() - start_time))

    # OptiDICE
    if result[seed].get('V_optidice') is None:
        start_time = time.time()
        pi_optidice = OptiDICE(mdp_all, pi_b, trajectory_all, alpha=1.0 / len(trajectory_all))[0]
        V_optidice = policy_evaluation(mdp, pi_optidice)[0][0]
        result[seed]['V_optidice'] = V_optidice
        print('{:20s}: {:.3f} ({:6.3f} secs)'.format("OptiDICE", V_optidice, time.time() - start_time))

    # Save the result...
    np.save(result_filepath + '.tmp.npy', result)
    # rename the resultfile
    os.rename(result_filepath + '.tmp.npy', result_filepath)


if __name__ == "__main__":
    for seed in range(10000):
        for trajectory_num_episodes in [10, 20, 50, 100, 200, 500, 1000, 2000]:
            for optimality in [0.9, 0.5]:
                start_time = time.time()
                run(seed, optimality, trajectory_num_episodes=trajectory_num_episodes)
                print(f'total_time: {time.time() - start_time:.3f}s')
