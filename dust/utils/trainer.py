import logging

import torch

from torch.optim import Adam

# Set up function for computing PPO policy loss
def _compute_loss_pi(pi_model, data, clip_ratio=0.2):
    
    # logp_old is the policy evaluated at the point before
    # the training process in this epoch.
    # The difference between the new logp and the old logp
    # is treated as the approx of the KL-divergence, represeting
    # the policy change made by the training.
    # PPO stops iterations once it becomes greater than the threshold.
    obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']

    # Policy loss
    pi, logp = pi_model(obs, act)
    # We should give act_dim=None to PPOBuffer
    # Giving a number, like 1, makes problems and is hard to find.
    # Here we try to detect this issue. 
    # I don't know what would happen if act_dim > 1
    assert logp.shape == logp_old.shape,\
        '{} and {}'.format(logp.shape, logp_old.shape)
    ratio = torch.exp(logp - logp_old)
    clip_adv = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * adv
    loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

    # Useful extra info
    approx_kl = (logp_old - logp).mean().item()
    ent = pi.entropy().mean().item()
    clipped = ratio.gt(1+clip_ratio) | ratio.lt(1-clip_ratio)
    clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
    pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac, loss=loss_pi.item())

    return loss_pi, pi_info

# Set up function for computing value loss
def _compute_loss_v(v_model, data):
    obs, ret = data['obs'], data['ret']
    loss_v = ((v_model(obs) - ret)**2).mean()
    return loss_v, dict(loss=loss_v.item())

# Temp function: using global variables
#def show_logp_change():
    #act = buf_data['act']
    #logp_old = buf_data['logp']
    #with torch.no_grad():
        #_, logp_new = pi_model(obs, act)
    #print('obs {}'.format(obs))
    #print('actions {}'.format(act))
    #print('logp_old {}'.format(logp_old))
    #print('logp_new {}'.format(logp_new))


class Trainer(object):
    """
    
    Attributes:
        v_model: Value model
        
        pi_model: Policy model
        
        pi_lr: Learning rate for the policy model
        
        vf_lr: Learning rate for the value model
        
        target_kl: Target KL-divergence
        
        clip_ratio: 
    
    """
    def __init__(self, pi_model, v_model):
        self.v_model = v_model
        self.pi_model = pi_model
        self.clip_ratio = 0.2
        self.target_kl = 0.01
        self.train_pi_iters = 80
        self.train_v_iters = 80
        self.create_optimizer()
    
    def create_optimizer(self, pi_lr=3e-4, vf_lr=1e-3):
        self.pi_optim = Adam(self.pi_model.parameters(), lr=pi_lr)
        self.vf_optim = Adam(self.v_model.parameters(), lr=vf_lr)
    
    def compute_loss_pi(self, data):
        """ Computes the policy loss and caches the gradient info
        """
        return _compute_loss_pi(self.pi_model, data, self.clip_ratio)

    def compute_loss_v(self, data):
        """ Computes the value loss and caches the gradient info
        """
        return _compute_loss_v(self.v_model, data)
    
    def update_policy_gradient(self, data):
        kl_thres = 1.5 * self.target_kl
        # Train policy with multiple steps of gradient descent
        for i in range(self.train_pi_iters):
            self.pi_optim.zero_grad()
            loss_pi, pi_info = self.compute_loss_pi(data)
            #kl = mpi_avg(pi_info['kl'])
            kl = pi_info['kl']
            if kl > kl_thres:
                logging.info(
                    'Pi iteration stopped at step {} '\
                    'due to reaching max kl {} > {}.'.format(i, kl, kl_thres))
                #show_logp_change()
                break
            loss_pi.backward()
            #mpi_avg_grads(ac.pi)    # average grads across MPI processes
            self.pi_optim.step()
        return pi_info
    
    def update_value_model(self, data):
        # Value function learning
        # Supervised learning to let v approximate the actual
        # return stored in the buf.
        for i in range(self.train_v_iters):
            self.vf_optim.zero_grad()
            loss_v, v_info = self.compute_loss_v(data)
            loss_v.backward()
            #mpi_avg_grads(ac.v)    # average grads across MPI processes
            self.vf_optim.step()
        return v_info
        
    def update(self, data):
        
        # Preserve the pre-training status
        _, pi_info_old = self.compute_loss_pi(data)
        _, v_info_old = self.compute_loss_v(data)
        
        # Train and update
        # Get the training info in the last iteration
        pi_info = self.update_policy_gradient(data)
        v_info = self.update_value_model(data)
        return pi_info_old, v_info_old, pi_info, v_info
