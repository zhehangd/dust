import torch

from dust.utils import su_core
from torch.distributions.categorical import Categorical

def test_action_dist():
    logits = torch.rand(8,3)
    act_dist = su_core.ActionDistribution(Categorical(logits=logits))
    sample = act_dist.sample()
    assert sample['a'].shape == torch.Size([8])
    
    logp = act_dist.log_prob(sample)
    assert logp.shape == torch.Size([8])
