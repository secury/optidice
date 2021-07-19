import argparse

POLICY_EXTRACTION = ['wbc', 'iproj']
ENV_NAME = [
    # Maze 2D
    'maze2d-umaze-v1',
    'maze2d-medium-v1',
    'maze2d-large-v1',

    # Gym
    'halfcheetah-random-v0',
    'walker2d-random-v0',
    'hopper-random-v0',
    'halfcheetah-medium-v0',
    'walker2d-medium-v0',
    'hopper-medium-v0',
    'halfcheetah-medium-replay-v0',
    'walker2d-medium-replay-v0',
    'hopper-medium-replay-v0',
    'halfcheetah-medium-expert-v0',
    'walker2d-medium-expert-v0',
    'hopper-medium-expert-v0',
]
DATA_POLICY = ['tanh_normal', 'tanh_mdn']
F = ['chisquare', 'kl', 'elu']
GENDICE_LOSS_TYPE = ['gendice', 'bestdice']
E_LOSS_TYPE = ['mse', 'minimax']


def boolean(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--policy_extraction', default='iproj', type=str, choices=POLICY_EXTRACTION)
    parser.add_argument('--env_name', default='maze2d-umaze-v1', type=str, choices=ENV_NAME)
    parser.add_argument('--total_iterations', default=int(3e6), type=int)
    parser.add_argument('--warmup_iterations', default=int(5e5), type=int)
    parser.add_argument('--log_iterations', default=int(1e4), type=int)
    parser.add_argument('--data_policy', default='tanh_mdn', type=str, choices=DATA_POLICY)
    parser.add_argument('--data_policy_num_mdn_components', default=5, type=int)
    parser.add_argument('--data_policy_mdn_temperature', default=1.0, type=float)
    parser.add_argument('--mean_range', default=(-7.24, 7.24))
    parser.add_argument('--logstd_range', default=(-5., 2.))
    parser.add_argument('--data_mean_range', default=(-7.24, 7.24))
    parser.add_argument('--data_logstd_range', default=(-5., 2.))
    parser.add_argument('--use_policy_entropy_constraint', default=True, type=boolean)
    parser.add_argument('--use_data_policy_entropy_constraint', default=False, type=boolean)
    parser.add_argument('--target_entropy', default=None, type=float)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--hidden_sizes', default=(256, 256))
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--resume', default=False, type=boolean)
    parser.add_argument('--alpha', default=1.0, type=float)
    parser.add_argument('--f', default='elu', type=str, choices=F)
    parser.add_argument('--gendice_v', default=True, type=boolean)
    parser.add_argument('--gendice_e', default=True, type=boolean)
    parser.add_argument('--gendice_loss_type', default='bestdice', type=str, choices=GENDICE_LOSS_TYPE)
    parser.add_argument('--lr', default=3e-4, type=float)
    parser.add_argument('--absorbing_state', default=True, type=boolean)
    parser.add_argument('--standardize_reward', default=True, type=boolean)
    parser.add_argument('--reward_scale', default=0.1, type=float)
    parser.add_argument('--e_loss_type', default='minimax', type=str, choices=E_LOSS_TYPE)
    parser.add_argument('--v_l2_reg', default=None, type=float)
    parser.add_argument('--lamb_scale', default=1., type=float)

    return parser
