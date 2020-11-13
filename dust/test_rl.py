import numpy as np
import torch

from dust.utils.su_core import create_default_actor_crtic
from dust.utils.ppo_buffer import PPOBuffer
from dust.utils.trainer import Trainer

def make_one_line(obj):
    """ Join lines into one
    Useful when you want to show a small object in one line.
    """
    return ''.join(str(obj).split())

# -----------------------------------------

def test_reinforcement_learning():

    obs_size = 3
    num_actions = 2
    buf_size = 3
    hidden_size = []
    pi_model, v_model = create_default_actor_crtic(
        obs_size, num_actions, hidden_size)

    pi_net = pi_model.logits_net
    pi_w, pi_b = pi_net.parameters()
    assert len(pi_net) == 2 # Linear + Identity
    assert pi_w.shape == torch.Size([num_actions, obs_size])
    assert pi_b.shape == torch.Size([num_actions])

    v_net = v_model.v_net
    v_w, v_b = v_net.parameters()
    assert len(v_net) == 2 # Linear + Identity
    assert v_w.shape == torch.Size([1, obs_size])
    assert v_b.shape == torch.Size([1])
    with torch.no_grad():
        pi_w.copy_(torch.tensor([[0.4, -0.2, -0.1],[-0.4, 0.2, 0.1]]))
        pi_b.copy_(torch.tensor([0.0, 0.0]))
        v_w.copy_(torch.tensor([[0.5, -0.5, 0.5],]))
        v_b.copy_(torch.tensor([0.0]))

    """
    print('pi_net')
    print(pi_net)
    print('w = {}'.format(make_one_line(pi_w)))
    print('b = {}'.format(make_one_line(pi_b)))
    print('')
    print('v_net')
    print(v_net)
    print('w = {}'.format(make_one_line(v_w)))
    print('b = {}'.format(make_one_line(v_b)))
    print('\n')
    """

    # -----------------------------------------

    def step(obs, a=None):
        """ Makes one prediction of policy and value
        
        If an action is not given, it takes a random action according to the
        observation and the policy, and returns the selected action, the
        estimated value of the observation, and the logit of which this action
        is selected. If an action is given, then this function behaves as if
        that action is randomly been selected.
        
        This function operates in a no_grad context. 
        """
        with torch.no_grad():
            assert isinstance(obs, torch.Tensor)
            pi = pi_model._distribution(obs)
            a = pi.sample() if a is None else a
            assert isinstance(a, torch.Tensor)
            assert a.ndim == 0 or a.ndim == 1
            logp_a = pi_model._log_prob_from_distribution(pi, a)
            v = v_model(obs)
        assert a.shape == logp_a.shape
        return a, v, logp_a

    def make_policy_value_table():
        # Three one-hot state and two actions
        # Assumes step function
        obs = torch.from_numpy(np.eye(3, dtype=np.float32))
        given_a = torch.full((3,), 0, dtype=torch.int64)
        a_0, v, logp_a_0 = step(obs, given_a)
        obs_table = obs
        value_table = v

        given_a = torch.full((3,), 1, dtype=torch.int64)
        a_1, v, logp_a_1 = step(obs, given_a)
        assert torch.equal(value_table, v)

        logp_a_table = torch.stack((logp_a_0, logp_a_1), 1)
        assert abs(torch.sum(torch.exp(logp_a_table)).item() - 3.0) < 1e-4
        return obs_table, value_table, logp_a_table

    obs_table, value_table, logp_a_table = make_policy_value_table()

    """
    print('obs_table: {}'.format(make_one_line(obs_table)))
    print('value_table: {}'.format(make_one_line(value_table)))
    print('logp_a_table: {}'.format(make_one_line(logp_a_table)))
    """
    
    # -----------------------------------------

    # Make atricicial trajectory
    buf = PPOBuffer(obs_size, None, buf_size)

    # obs(index)-action-reward sequence
    iar_seq = [(0, 0, 0.0),(1, 1, 0.0), (2, 0, 1.0)]

    for i, a, r in iar_seq:
        o = obs_table[i]
        v = value_table[i]
        logp = logp_a_table[i, a]
        buf.store(o, a, r, v, logp)
    buf.finish_path(0)
    buf_data = buf.get()

    #print('buf_data:')
    #for key, val in buf_data.items():
    #    print('  - {}: {}'.format(key, make_one_line(val)))

    # -----------------------------------------


    trainer = Trainer(pi_model, v_model)
    pi_info_old, v_info_old, pi_info, v_info = trainer.update(buf_data)

    kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
    delta_loss_pi = pi_info['loss'] - pi_info_old['loss']
    delta_loss_v = v_info['loss'] - v_info_old['loss']
    """
    print('LossPi: {}'.format(pi_info_old['loss']))
    print('LossV: {}'.format(v_info_old['loss']))
    print('KL: {}'.format(kl))
    print('Entropy: {}'.format(ent))
    print('ClipFrac: {}'.format(cf))
    print('DeltaLossPi: {}'.format(delta_loss_pi))
    print('DeltaLossV: {}'.format(delta_loss_pi))
    """
    
    # Guaranteed for linear
    assert delta_loss_pi < 0
    assert delta_loss_v < 0
    #print('')
    
    #print('Result')
    test_obs_table, test_value_table, test_logp_a_table = make_policy_value_table()
    #print('obs_table: {}'.format(make_one_line(test_obs_table)))
    #print('value_table: {}'.format(make_one_line(test_value_table)))
    #print('logp_a_table: {}'.format(make_one_line(test_logp_a_table)))
    #print('')

    #print ('Reference reuslt:')
    
    torch.testing.assert_allclose(test_obs_table,
        torch.tensor([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]))
    torch.testing.assert_allclose(test_value_table,
        torch.tensor([0.6528,-0.3439,0.6530]))
    torch.testing.assert_allclose(test_logp_a_table,
        torch.tensor([[-0.3710,-1.1714],[-0.9712,-0.4758],[-0.7979,-0.5984]]))
    
    #print('obs_table: tensor([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])')
    #print('value_table: tensor([0.6528,-0.3439,0.6530])')
    #print('logp_a_table: tensor([[-0.3710,-1.1714],[-0.9712,-0.4758],[-0.7979,-0.5984]])')
