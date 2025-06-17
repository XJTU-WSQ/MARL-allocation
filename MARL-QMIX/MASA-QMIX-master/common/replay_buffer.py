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
        """
        基于奖励值概率选择要抽取的样本
        """
        try:
            # 获取当前缓冲区中的所有奖励
            current_rewards = self.buffers['r'][:self.current_size,-1,]
            
            probs = abs(current_rewards)+0.1 # 确保值为正
            probs = probs / np.sum(probs) # 归一化概率
            probs = 1 - probs # 绝对值越大，被选中删除的可能性越小
            probs = probs / np.sum(probs)
            probs = np.asarray(probs).flatten() # 确保 probs 是一维数组

            # 根据概率随机选择要替换的索引
            sample_idxs = np.random.choice(
                np.arange(self.current_size), 
                size=batch_size, 
                replace=False,  # 不重复选择
                p=probs
            )
        except Exception as e:
            logger.error(f"Error in _get_probabilistic_idx: {e}")
            # 如果发生错误，改为简单的随机抽样
            sample_idxs = np.random.randint(0, self.current_size, batch_size)
        return sample_idxs