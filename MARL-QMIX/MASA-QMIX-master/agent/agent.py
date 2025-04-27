import numpy as np
import pandas as pd
import torch
from policy.qmix import QMIX
from scipy.optimize import linear_sum_assignment
from loguru import logger

def random_choice_with_mask(avail_actions):
    temp = []
    for i, eve in enumerate(avail_actions):
        if eve == 1:
            temp.append(i)
    return np.random.choice(temp, 1, False)[0]


class Agents:
    def __init__(self, args, writer=None):
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        self.policy = QMIX(args, writer)
        self.args = args
        print('Init Agents')

    def choose_action(self, obs, agent_num, avail_actions, epsilon):
        """
        根据 Q 值和动态权重选择动作。
        """
        inputs = obs.copy()
        agent_id = np.zeros(self.n_agents)
        agent_id[agent_num] = 1.

        if self.args.reuse_network:
            inputs = np.hstack((inputs, agent_id))
        hidden_state = self.policy.eval_hidden[:, agent_num, :]

        # 转换输入维度
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        avail_actions = torch.tensor(avail_actions, dtype=torch.float32).unsqueeze(0)

        if self.args.cuda:
            inputs = inputs.cuda()
            hidden_state = hidden_state.cuda()

        # 计算 Q 值
        q_value, self.policy.eval_hidden[:, agent_num, :] = self.policy.eval_rnn(inputs, hidden_state)
        # choose action from q value
        q_value[avail_actions == 0.0] = -float("inf")
        if np.random.uniform() < epsilon:
            action = random_choice_with_mask(avail_actions[0])
        else:
            action = torch.argmax(q_value).cpu()
        return action

    def choose_actions_batch(self, obs, avail_actions, epsilon, constraint_type=4):
        """
        为所有代理选择动作
        constraint_type: 1 表示agent优先策略，每个agent根据q值最大进行选择
        constraint_type: 2 表示匈牙利算法，通过q值构建成本矩阵，根据全局q值最大进行选择
        constraint_type: 3 表示task优先策略，每个task根据q值最大进行选择
        constraint_type: 4 表示task优先策略，每个task根据q值最大进行选择，但探索策略更保守
        补充：类型1中每个任务可以分配给多个agent，类型2与3中每个任务只会分配给单个agent
        """
        inputs = obs.copy()
        batch_size = len(obs)
        
        # 创建代理ID的one-hot编码
        agent_ids = np.eye(self.n_agents)

        if self.args.reuse_network:
            inputs = np.hstack((inputs, agent_ids))

        # 转换输入维度
        inputs = torch.tensor(inputs, dtype=torch.float32)
        avail_actions = torch.tensor(avail_actions, dtype=torch.float32)

        if self.args.cuda:
            inputs = inputs.cuda()

        # 初始化隐藏状态
        hidden_states = self.policy.eval_hidden[:, :batch_size, :]

        with torch.no_grad():
            q_values, self.policy.eval_hidden[:, :batch_size, :] = self.policy.eval_rnn(inputs, hidden_states)
            
            if constraint_type == 2: # 动作优先策略：基于匈牙利算法，每个agent选择最佳的动作，带随机探索和1对1约束
                fillna_value = 1e6
                cost_matrix = 1/(q_values.cpu()-q_values.cpu().min()+0.001)
                
                for i in range(batch_size):
                    if np.random.uniform() < epsilon:
                        cost_matrix[i] = cost_matrix[i][torch.randperm(cost_matrix[i].size(0))]
                    if obs[i][0] == 1:
                        cost_matrix[i] = fillna_value
                cost_matrix[avail_actions == 0.0] = fillna_value
                cost_matrix = cost_matrix.nan_to_num(fillna_value)
                try:
                    row_ind, col_ind = linear_sum_assignment(cost_matrix)
                except:
                    print(cost_matrix)
                    print(q_values)
                actions = col_ind
                for i in range(batch_size):
                    if avail_actions[i][actions[i]]==0:
                        actions[i]=10
            else: 
                q_values[avail_actions == 0.0] = -float("inf")
                if constraint_type == 1: # 动作优先策略，每个agent选择最佳的动作，带随机探索，不支持1对1约束
                    actions = []
                    for i in range(batch_size):
                        if np.random.uniform() < epsilon:
                            action = random_choice_with_mask(avail_actions[i]) # 按照一定探索率随机选择
                        else:
                            action = torch.argmax(q_values[i]).cpu().item() # 根据Q值最大化进行最优选择
                        actions.append(action)
                elif constraint_type == 3: # 任务优先策略，每个任务选择最佳的agent，带随机探索和1对1约束
                    agents = []
                    q_values_clone = q_values.clone()
                    for i in range(self.n_actions):
                        if q_values_clone[:-1,i].cpu().max()==-float("inf"):
                            agent_id = np.nan
                        elif np.random.uniform() < epsilon:
                            agent_id = random_choice_with_mask(avail_actions[:,i]) # 按照一定探索率随机选择
                            q_values_clone[agent_id,:] = -float("inf")
                        else:
                            agent_id = torch.argmax(q_values_clone[:,i]).cpu().item() # 根据Q值最大化进行最优选择
                            q_values_clone[agent_id,:] = -float("inf")
                        agents.append(agent_id)
                    actions = [10 for i in range(batch_size)] # 
                    for i in range(len(agents)):
                        if not pd.isna(agents[i]):
                            actions[agents[i]] = i
                elif constraint_type == 4: # 任务优先策略，每个任务选择最佳的agent，带随机探索和1对1约束
                    agents = []
                    q_values_clone = q_values.clone()
                    for i in range(self.n_actions):
                        if q_values_clone[:-1, i].cpu().max() == -float("inf"):
                            agent_id = np.nan
                        elif np.random.uniform() < epsilon: # 以 epsilon 的概率进行探索
                            rand_choice = np.random.uniform()
                            if rand_choice < 0.25: # 25% 的概率选择次优结果
                                q_values_clone[:, i] = q_values_clone[:, i].cpu()
                                q_values_clone[:, i][q_values_clone[:-1, i].argmax()] = -float("inf")  # 排除最优
                                agent_id = torch.argmax(q_values_clone[:, i]).cpu().item()
                                if q_values_clone[:, i].cpu().max() == -float("inf"):
                                    agent_id = np.nan
                            elif rand_choice < 0.5: # 25% 的概率选择更次优的结果
                                q_values_clone[:, i] = q_values_clone[:, i].cpu()
                                q_values_clone[:, i][q_values_clone[:-1, i].argmax()] = -float("inf")  # 排除最优
                                second_best = torch.argmax(q_values_clone[:-1, i]).cpu().item()
                                q_values_clone[:, i][second_best] = -float("inf")  # 排除次优
                                agent_id = torch.argmax(q_values_clone[:, i]).cpu().item()
                                if q_values_clone[:, i].cpu().max() == -float("inf"):
                                    agent_id = np.nan
                            else: # 50% 的概率保留原有的规则进行随机选择
                                agent_id = random_choice_with_mask(avail_actions[:, i])
                                q_values_clone[agent_id, :] = -float("inf")
                        else:
                            # 根据Q值最大化进行最优选择
                            agent_id = torch.argmax(q_values_clone[:, i]).cpu().item()
                            q_values_clone[agent_id, :] = -float("inf")
                        agents.append(agent_id)
                    actions = [10 for i in range(batch_size)]
                    for i in range(len(agents)):
                        if not pd.isna(agents[i]):
                            actions[agents[i]] = i
                else:
                    logger.error(f'unsupport paramater value with constraint_type={constraint_type}')
        return actions

    def _get_max_episode_len(self, batch):
        terminated = batch['terminated']
        episode_num = terminated.shape[0]
        max_episode_len = 0
        # 由于episodelimit的长度内没有terminal==1，所以导致max_episode_len == 0
        for episode_idx in range(episode_num):
            for transition_idx in range(self.args.episode_limit):
                if terminated[episode_idx, transition_idx, 0] == 1:
                    if transition_idx + 1 >= max_episode_len:
                        max_episode_len = transition_idx + 1
                    break
                if transition_idx == self.args.episode_limit - 1:
                    max_episode_len = self.args.episode_limit
        return max_episode_len

    def train(self, batch, train_step, epsilon=None):

        # different episode has different length, so we need to get max length of the batch
        max_episode_len = self._get_max_episode_len(batch)
        for key in batch.keys():
            if key != 'z': # 另外一个算法的参数，qmix可不考虑（无影响）
                batch[key] = batch[key][:, :max_episode_len]
        self.policy.learn(batch, max_episode_len, train_step, epsilon)

        if train_step > 0 and train_step % self.args.save_cycle == 0:
            # print("\n开始保存模型", 'train_step:', train_step, 'save_cycle:', self.args.save_cycle)
            self.policy.save_model(train_step)
