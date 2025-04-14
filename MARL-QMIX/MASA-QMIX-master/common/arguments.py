"""
训练时需要的参数
"""
import argparse


def get_common_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--replay_dir', type=str, default=r'', help='absolute path to save the replay')
    parser.add_argument('--model_dir', type=str, default='./model', help='model directory of the policy')
    parser.add_argument('--result_dir', type=str, default='./result', help='result directory of the policy')
    parser.add_argument('--load_model', type=bool, default=False, help='whether to load the pretrained model')
    parser.add_argument('--alg', type=str, default='qmix', help='the algorithm to train the agent')
    parser.add_argument('--last_action', type=bool, default=False, help='whether to use the last action to choose action')
    parser.add_argument('--reuse_network', type=bool, default=True, help='whether to use one network for all agents')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
    parser.add_argument('--optimizer', type=str, default="RMS", help='optimizer')
    parser.add_argument('--evaluate_epoch', type=int, default=50, help='number of the epoch to evaluate the agent')
    parser.add_argument('--learn', type=bool, default=True, help='whether to train the model')
    parser.add_argument('--cuda', type=bool, default=True, help='whether to use the GPU')
    parser.add_argument('--map', type=str, default="Schedule", help='map name')
    parser.add_argument('--log_step_data', type=bool, default=False, help='Log step data for debugging')
    parser.add_argument("--use_tensorboard", type=bool, default=True, help="Enable TensorBoard logging")
    parser.add_argument("--run_name", type=str, default="default_run", help="Name of the current run")
    args = parser.parse_args()
    return args


# arguments of q-mix
def get_mixer_args(args):
    # network
    args.rnn_hidden_dim = 128
    args.qmix_hidden_dim = 64
    args.two_hyper_layers = False
    args.hyper_hidden_dim = 128
    # 学习率与探索
    args.lr = 1e-4  #
    args.epsilon = 1.0  # 初始探索率不变
    args.min_epsilon = 0.1  #
    anneal_steps = 600000  # 原为 200000，缩短衰减周期
    args.anneal_epsilon = (args.epsilon - args.min_epsilon) / anneal_steps
    args.epsilon_anneal_scale = 'step'

    # 训练总轮数与每轮收集
    args.n_epoch = 10000  # 原为 10000，先试 6000，看训练曲线后再调整
    args.n_episodes = 8  # 保持不变，每轮收集 8 个 episode 的数据

    # 每轮训练步数
    args.train_steps = 2

    # 评估、保存与目标网络更新
    args.evaluate_cycle = 100  # 原为 50，评估不必过于频繁
    args.save_cycle = 1000  # 原为 50，保存过于频繁会产生日志冗余
    args.target_update_cycle = 200

    # 经验回放
    args.batch_size = 64  # 保持不变，常见取值
    args.buffer_size = int(1e4)  # 原为 5e3，增大后可存更多样本，提升稳定性

    # 梯度裁剪
    args.grad_norm_clip = 10  # 保持不变，防止梯度爆炸
    return args
