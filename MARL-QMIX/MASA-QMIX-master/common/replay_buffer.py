import numpy as np
import threading
from loguru import logger

class ReplayBuffer:
    def __init__(self, args):
        self.args = args
        self.n_actions = self.args.n_actions
        self.n_agents = self.args.n_agents
        self.state_shape = self.args.state_shape
        self.obs_shape = self.args.obs_shape
        self.size = self.args.buffer_size
        self.episode_limit = self.args.episode_limit
        # memory management
        self.current_idx = 0
        self.current_size = 0
        # create the buffer to store info
        self.buffers = {'o': np.empty([self.size, self.episode_limit, self.n_agents, self.obs_shape], dtype=np.float16),
                        'u': np.empty([self.size, self.episode_limit, self.n_agents, 1], dtype=np.float16),
                        's': np.empty([self.size, self.episode_limit, self.state_shape], dtype=np.float16),
                        'r': np.empty([self.size, self.episode_limit, 1], dtype=np.float16),
                        'o_next': np.empty([self.size, self.episode_limit, self.n_agents, self.obs_shape], dtype=np.float16),
                        's_next': np.empty([self.size, self.episode_limit, self.state_shape], dtype=np.float16),
                        'avail_u': np.empty([self.size, self.episode_limit, self.n_agents, self.n_actions], dtype=np.float16),
                        'avail_u_next': np.empty([self.size, self.episode_limit, self.n_agents, self.n_actions], dtype=np.float16),
                        'u_onehot': np.empty([self.size, self.episode_limit, self.n_agents, self.n_actions], dtype=np.float16),
                        'padded': np.empty([self.size, self.episode_limit, 1], dtype=np.float16),
                        'terminated': np.empty([self.size, self.episode_limit, 1], dtype=np.float16)
                        }
        # thread lock
        self.lock = threading.Lock()

        # store the episode
    def store_episode(self, episode_batch):
        batch_size = episode_batch['o'].shape[0]  # episode_number
        with self.lock:
            idxs = self._get_storage_idx(inc=batch_size)
            # store the informations
            self.buffers['o'][idxs] = episode_batch['o']
            self.buffers['u'][idxs] = episode_batch['u']
            self.buffers['s'][idxs] = episode_batch['s']
            self.buffers['r'][idxs] = episode_batch['r']
            self.buffers['o_next'][idxs] = episode_batch['o_next']
            self.buffers['s_next'][idxs] = episode_batch['s_next']
            self.buffers['avail_u'][idxs] = episode_batch['avail_u']
            self.buffers['avail_u_next'][idxs] = episode_batch['avail_u_next']
            self.buffers['u_onehot'][idxs] = episode_batch['u_onehot']
            self.buffers['padded'][idxs] = episode_batch['padded']
            self.buffers['terminated'][idxs] = episode_batch['terminated']

    def sample(self, batch_size):
        temp_buffer = {}
        idx = np.random.randint(0, self.current_size, batch_size)
        for key in self.buffers.keys():
            temp_buffer[key] = self.buffers[key][idx]
        return temp_buffer

    def sample_probabilistic(self, batch_size):
        """
        使用概率方法采样样本
        """
        temp_buffer = {}
        replace_idxs = self._get_probabilistic_idx(batch_size)
        for key in self.buffers.keys():
            temp_buffer[key] = self.buffers[key][replace_idxs]
        return temp_buffer
    
    def _get_storage_idx(self, inc=None):
        inc = inc or 1
        if self.current_idx + inc <= self.size:
            idx = np.arange(self.current_idx, self.current_idx + inc)
            self.current_idx += inc
        elif self.current_idx < self.size:
            overflow = inc - (self.size - self.current_idx)
            idx_a = np.arange(self.current_idx, self.size)
            idx_b = np.arange(0, overflow)
            idx = np.concatenate([idx_a, idx_b])
            self.current_idx = overflow
        else:
            idx = np.arange(0, inc)
            self.current_idx = inc
        self.current_size = min(self.size, self.current_size + inc)
        if inc == 1:
            idx = idx[0]
        return idx

    def _get_probabilistic_idx(self, batch_size):
        try:
            # 安全获取当前缓冲区有效大小
            valid_size = min(self.current_size, self.buffers['r'].shape[0])
            if valid_size == 0:
                return np.array([], dtype=np.int32)  # 返回空数组

            # 获取奖励数据并转换为float32以提高精度
            rewards = self.buffers['r'][:valid_size].astype(np.float32)

            # 处理可能的NaN和Inf值
            rewards = np.nan_to_num(rewards, nan=0.0, posinf=0.0, neginf=0.0)

            # 计算累积奖励（沿时间轴求和）
            cumulative_rewards = np.sum(rewards, axis=1)

            # 处理全零奖励的特殊情况
            if np.all(cumulative_rewards == 0):
                # 所有奖励都为零时使用均匀分布
                priorities = np.ones(valid_size, dtype=np.float32)
            else:
                # 确保优先级值为正（避免负值）
                min_val = np.min(cumulative_rewards)
                priorities = cumulative_rewards - min_val + 1e-5  # 确保正值

                # 数值稳定处理：防止过大值导致溢出
                max_priority = np.max(priorities)
                if max_priority > 1e6:  # 检测可能溢出的情况
                    priorities = priorities / (max_priority / 1e6)  # 缩小数值范围

            # 计算概率
            probs = priorities / np.sum(priorities)

            # 检查概率和并修正浮点精度误差
            probs_sum = np.sum(probs)
            if abs(probs_sum - 1.0) > 1e-6:
                probs = probs / probs_sum

            # 动态决定是否允许重复采样
            replace = batch_size > valid_size

            # 确保请求的批量大小不超过可用样本数
            sample_size = min(batch_size, valid_size)

            # 安全采样
            return np.random.choice(
                np.arange(valid_size),
                size=sample_size,
                replace=replace,
                p=probs.flatten()
            )

        except Exception as e:
            print(f"概率采样失败: {e}, 改用随机采样")
            # 出错时回退到随机采样
            valid_size = min(self.current_size, self.buffers['r'].shape[0])
            replace = batch_size > valid_size
            sample_size = min(batch_size, valid_size)
            return np.random.choice(
                np.arange(valid_size),
                size=sample_size,
                replace=replace
            )


