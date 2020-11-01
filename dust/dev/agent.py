import logging
import os

import torch
import torch.nn as nn

import numpy as np
import scipy.signal

from torch.optim import Adam
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical

from dust import _dust
from dust.core import progress_log
from dust.utils import np_utils

_argparser = _dust.argparser()

_argparser.add_argument('--cuda', action='store_true',
    help='Use CUDA')

def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

class Actor(nn.Module):

    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and 
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a

class MLPCategoricalActor(Actor):
    
    def __init__(self, obs_dim, act_dim, hidden_sizes):
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim])

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)

class MLPCritic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1])

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1) # Critical to ensure v has right shape.

class MLPActorCritic(nn.Module):
    
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.pi = MLPCategoricalActor(obs_dim, act_dim, (16,16))
        self.v  = MLPCritic(obs_dim, (16,16))
        self.use_cuda = _dust.project().args.cuda
        if self.use_cuda:
            self.pi.cuda()
            self.v.cuda()

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        if self.use_cuda:
            a = a.cpu()
            v = v.cpu()
            logp_a = logp_a.cpu()
        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]

class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        def combined_shape(length, shape=None):
            if shape is None:
                return (length,)
            return (length, shape) if np.isscalar(shape) else (length, *shape)
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        
        def discount_cumsum(x, discount):
            """
            magic from rllab for computing discounted cumulative sums of vectors.

            input: 
                vector x, 
                [x0, 
                x1, 
                x2]

            output:
                [x0 + discount * x1 + discount^2 * x2,  
                x1 + discount * x2,
                x2]
            """
            return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]
        
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)
        
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]
        
        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        #adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        #self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf)
        
        if _dust.project().args.cuda:
            return {k: torch.as_tensor(v, dtype=torch.float32).cuda() for k,v in data.items()}
        else:
            return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}


# Set up function for computing PPO policy loss
def compute_loss_pi(ac, data):
    clip_ratio = 0.2
    obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']

    # Policy loss
    pi, logp = ac.pi(obs, act)
    ratio = torch.exp(logp - logp_old)
    clip_adv = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * adv
    loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

    # Useful extra info
    approx_kl = (logp_old - logp).mean().item()
    ent = pi.entropy().mean().item()
    clipped = ratio.gt(1+clip_ratio) | ratio.lt(1-clip_ratio)
    clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
    pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

    return loss_pi, pi_info

# Set up function for computing value loss
def compute_loss_v(ac, data):
    obs, ret = data['obs'], data['ret']
    return ((ac.v(obs) - ret)**2).mean()

def network_update(ac, data, pi_optim, vf_optim):
    train_v_iters = 80
    train_pi_iters = 80
    target_kl = 0.01
    
    pi_l_old, pi_info_old = compute_loss_pi(ac, data)
    pi_l_old = pi_l_old.item()
    v_l_old = compute_loss_v(ac, data).item()

    # Train policy with multiple steps of gradient descent
    for i in range(train_pi_iters):
        pi_optim.zero_grad()
        loss_pi, pi_info = compute_loss_pi(ac, data)
        #kl = mpi_avg(pi_info['kl'])
        kl = pi_info['kl']
        if kl > 1.5 * target_kl:
            logging.info('Early stopping at step %d due to reaching max kl.'%i)
            break
        loss_pi.backward()
        # TODO
        #mpi_avg_grads(ac.pi)    # average grads across MPI processes
        pi_optim.step()

    #logger.store(StopIter=i)

    # Value function learning
    for i in range(train_v_iters):
        vf_optim.zero_grad()
        loss_v = compute_loss_v(ac, data)
        loss_v.backward()
        # TODO
        #mpi_avg_grads(ac.v)    # average grads across MPI processes
        vf_optim.step()

    # Log changes from update
    kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
    log_dict = dict(LossPi=pi_l_old, LossV=v_l_old,
                    KL=kl, Entropy=ent, ClipFrac=cf,
                    DeltaLossPi=(loss_pi.item() - pi_l_old),
                    DeltaLossV=(loss_v.item() - v_l_old))

_EPOCH_LENGTH = 200

class Agent(object):
    
    def __init__(self, env, is_training):
        self.env = env
        self.num_players = 1
        
        ac = MLPActorCritic(25, 4)
        self.ac = ac
        
        def count_vars(module):
            return sum([np.prod(p.shape) for p in module.parameters()])
        var_counts = tuple(count_vars(module) for module in [ac.pi, ac.v])
        logging.info('\nNumber of parameters: \t pi: %d, \t v: %d\n'%var_counts)
        
        self.training = is_training
        
        if self.training:
            obs_dim = 25
            act_dim = 4
            gamma = 0.999
            lam = 0.97
            buf = PPOBuffer(obs_dim, 1, _EPOCH_LENGTH, gamma, lam)
            self.buf = buf
            
            pi_lr = 3e-4
            vf_lr = 1e-3
            pi_optim = Adam(ac.pi.parameters(), lr=pi_lr)
            vf_optim = Adam(ac.v.parameters(), lr=vf_lr)
            self.pi_optim = pi_optim
            self.vf_optim = vf_optim
            self.progress = progress_log.ProgressLog()
            
        self.curr_epoch_tick = 0
        self.curr_epoch = 0
        self.epoch_reward = 0 # reward collected in the epoch (NOT round)
    
    def _get_observation(self):
        """ Extract observation from the current env
        """
        def index_to_coords(idxs, h):
            return np.stack((idxs // h, idxs % h), -1)
        env = self.env
        # Generate vision at each player position
        coords = index_to_coords(env.player_coords, env.map_shape[1])
        map_state = np.zeros(env.map_shape, np.float32)
        map_state_flatten = map_state.reshape(-1)
        map_state_flatten[env.wall_coords] = -1
        map_state_flatten[env.food_coords] = 1
        obs = np_utils.extract_view(map_state, coords, 2, True)
        obs = obs.reshape((len(obs), -1))
        return obs
    
    def act(self):
        
        # Check any event and calculate the reward from the last stepb 
        
        # observe
        env = self.env
        ac = self.ac
        
        obs = self._get_observation()
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
        if ac.use_cuda:
            obs_tensor = obs_tensor.cuda()
        a, v, logp = ac.step(obs_tensor)
        
        #self.env.set_action(np.random.randint(0, 4, self.num_players))
        self.env.set_action(a)
        self.ac_data = (a, v, logp, obs)
        #logging.info('act: {}'.format(env.curr_round_tick))

    def update(self):
        # collect rewards
        env = self.env
        
        self.epoch_reward += env.tick_reward
        
        status_msg = 'tick: {} round: {} round_tick: {} ' \
                     'epoch: {} epoch_tick: {} epoch_reward: {} round_reward: {}'.format(
                        env.curr_tick, env.curr_round,
                        env.curr_round_tick, self.curr_epoch,
                        self.curr_epoch_tick, self.epoch_reward, env.round_reward)
        
        if self.training:
            self._update_tick()
            end_of_epoch = self.curr_epoch_tick + 1 == _EPOCH_LENGTH
            
            if env.end_of_round or end_of_epoch:
                logging.info('end_of_round: ' + status_msg)
                self._update_round()
                
            if end_of_epoch:
                logging.info('end_of_epoch')
                self.progress.set_fields(epoch=self.curr_epoch, score=self.epoch_reward)
                self.progress.finish_line()
                self._update_epoch()
                
                # Force env to end the round
                env.end_of_round = True
            
                if self.curr_epoch % 10 == 0:
                    logging.info('save actor critic')
                    self._save_actor_critic()
                
                self.curr_epoch += 1
                self.curr_epoch_tick = 0
                self.epoch_reward = 0
            else:
                self.curr_epoch_tick += 1
    
    def _update_tick(self):
        a, v, logp, obs = self.ac_data
        r = self.env.tick_reward
        ac = self.ac
        buf = self.buf
        buf.store(obs, a, r, v, logp)
    
    def _update_round(self):
        obs = self._get_observation()
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
        if self.ac.use_cuda:
            obs_tensor = obs_tensor.cuda()
        _, v, _ = self.ac.step(obs_tensor)
        self.buf.finish_path(v)
        
    def _update_epoch(self):
        data = self.buf.get()
        network_update(self.ac, data, self.pi_optim, self.vf_optim)
    
    def _save_actor_critic(self):
        proj = _dust.project()
        net_file = os.path.join(proj.proj_dir, 'network.pth')
        torch.save(self.ac, net_file)
        
