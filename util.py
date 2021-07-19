import os
import time

import d4rl
import numpy as np
from tqdm import tqdm


def dice_dataset(env, standardize_observation=True, absorbing_state=True, standardize_reward=True):
    """
    env: d4rl environment
    """
    dataset = env.get_dataset()
    N = dataset['rewards'].shape[0]
    initial_obs_, obs_, next_obs_, action_, reward_, done_ = [], [], [], [], [], []

    use_timeouts = ('timeouts' in dataset)

    episode_step = 0
    for i in range(N-1):
        obs = dataset['observations'][i].astype(np.float32)
        new_obs = dataset['observations'][i+1].astype(np.float32)
        action = dataset['actions'][i].astype(np.float32)
        reward = dataset['rewards'][i].astype(np.float32)
        done_bool = bool(dataset['terminals'][i])

        is_final_timestep = dataset['timeouts'][i] if use_timeouts else (episode_step == env._max_episode_steps - 1)
        if is_final_timestep:
            # Skip this transition and don't apply terminals on the last step of an episode
            episode_step = 0
            continue

        if episode_step == 0:
            initial_obs_.append(obs)

        obs_.append(obs)
        next_obs_.append(new_obs)
        action_.append(action)
        reward_.append(reward)
        done_.append(done_bool)
        episode_step += 1

        if done_bool or is_final_timestep:
            episode_step = 0

    initial_obs_dataset = {
        'initial_observations': np.array(initial_obs_, dtype=np.float32)
    }
    dataset = {
        'observations': np.array(obs_, dtype=np.float32),
        'actions': np.array(action_, dtype=np.float32),
        'next_observations': np.array(next_obs_, dtype=np.float32),
        'rewards': np.array(reward_, dtype=np.float32),
        'terminals': np.array(done_, dtype=np.float32)
    }
    dataset_statistics = {
        'observation_mean': np.mean(dataset['observations'], axis=0),
        'observation_std': np.std(dataset['observations'], axis=0),
        'reward_mean': np.mean(dataset['rewards']),
        'reward_std': np.std(dataset['rewards']),
        'N_initial_observations': len(initial_obs_),
        'N': len(obs_),
        'observation_dim': dataset['observations'].shape[-1],
        'action_dim': dataset['actions'].shape[-1]
    }

    if standardize_observation:
        initial_obs_dataset['initial_observations'] = (initial_obs_dataset['initial_observations'] - dataset_statistics['observation_mean']) / (dataset_statistics['observation_std'] + 1e-10)
        dataset['observations'] = (dataset['observations'] - dataset_statistics['observation_mean']) / (dataset_statistics['observation_std'] + 1e-10)
        dataset['next_observations'] = (dataset['next_observations'] - dataset_statistics['observation_mean']) / (dataset_statistics['observation_std'] + 1e-10)
    if standardize_reward:
        dataset['rewards'] = (dataset['rewards'] - dataset_statistics['reward_mean']) / (dataset_statistics['reward_std'] + 1e-10)

    if absorbing_state:
        # add additional dimension to observations to deal with absorbing state
        initial_obs_dataset['initial_observations'] = np.concatenate((initial_obs_dataset['initial_observations'], np.zeros((dataset_statistics['N_initial_observations'], 1))), axis=1).astype(np.float32)
        dataset['observations'] = np.concatenate((dataset['observations'], np.zeros((dataset_statistics['N'], 1))), axis=1).astype(np.float32)
        dataset['next_observations'] = np.concatenate((dataset['next_observations'], np.zeros((dataset_statistics['N'], 1))), axis=1).astype(np.float32)
        terminal_indices = np.where(dataset['terminals'])[0]
        absorbing_state = np.eye(dataset_statistics['observation_dim'] + 1)[-1].astype(np.float32)
        dataset['observations'], dataset['actions'], dataset['rewards'], dataset['next_observations'], dataset['terminals'] = \
            list(dataset['observations']), list(dataset['actions']), list(dataset['rewards']), list(dataset['next_observations']), list(dataset['terminals'])
        for terminal_idx in terminal_indices:
            dataset['next_observations'][terminal_idx] = absorbing_state
            dataset['observations'].append(absorbing_state)
            dataset['actions'].append(dataset['actions'][terminal_idx])
            dataset['rewards'].append(0)
            dataset['next_observations'].append(absorbing_state)
            dataset['terminals'].append(1)

        dataset['observations'], dataset['actions'], dataset['rewards'], dataset['next_observations'], dataset['terminals'] = \
            np.array(dataset['observations'], dtype=np.float32), np.array(dataset['actions'], dtype=np.float32), np.array(dataset['rewards'], dtype=np.float32), \
            np.array(dataset['next_observations'], dtype=np.float32), np.array(dataset['terminals'], dtype=np.float32)

    return initial_obs_dataset, dataset, dataset_statistics


def evaluate(env, agent, dataset_statistics, absorbing_state=True, num_evaluation=3, pid=None):
    normalized_scores = []

    for eval_iter in range(num_evaluation):
        start_time = time.time()
        obs = env.reset()
        episode_reward = 0
        for t in tqdm(range(env._max_episode_steps), ncols=70, desc='evaluate', ascii=True, disable=os.environ.get("DISABLE_TQDM", False)):
            if absorbing_state:
                obs_standardized = np.append((obs - dataset_statistics['observation_mean']) / (dataset_statistics['observation_std'] + 1e-10), 0)
            else:
                obs_standardized = (obs - dataset_statistics['observation_mean']) / (dataset_statistics['observation_std'] + 1e-10)

            actions = agent.step((np.array([obs_standardized])).astype(np.float32))
            action = actions[0][0].numpy()

            next_obs, reward, done, info = env.step(action)
            episode_reward += reward
            if done:
                break
            obs = next_obs
        normalized_score = 100 * (episode_reward - d4rl.infos.REF_MIN_SCORE[env.spec.id]) / (d4rl.infos.REF_MAX_SCORE[env.spec.id] - d4rl.infos.REF_MIN_SCORE[env.spec.id])
        if pid is not None:
            print(f'PID [{pid}], Eval Iteration {eval_iter}')
        print(f'normalized_score: {normalized_score} (elapsed_time={time.time() - start_time:.3f}) ')
        normalized_scores.append(normalized_score)

    return np.mean(normalized_scores)
