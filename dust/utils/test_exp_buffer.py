import numpy as np
import pytest

from dust.utils.exp_buffer import ExpBuffer

def test_exp_buffer():
    """
    """
    obs_dtype = 'i4'
    act_dtype = 'i4'
    ext_dtype = 'f4'
    buf_capacity = 10
    buf = ExpBuffer(
        buf_capacity, obs_dtype, act_dtype, ext_dtype,
        gamma=0.9, lam=0.8)
    
    # 1 point reward is given at the end
    buf.store(0, 0, 0.1, 0, 1)
    buf.store(1, 1, 0.2, 0, 0)
    buf.store(2, 0, 0.3, 0, 1)
    buf.store(3, 2, 0.4, 0, 0)
    buf.store(4, 3, 0.4, 1, 1)
    assert buf.buf_capacity == buf_capacity
    assert buf.buf_size == 5
    buf.finish_path(0)
    
    # Since only 1 point reward is given at the end,
    # the observed burn is just
    # R(t) = 0.9**(4-t)
    for t in range(5):
        assert buf.buf_data['ret'][t] == pytest.approx(0.9**(4-t)),\
            '{}'.format(buf.buf_data['ret'])
    
    # For the first 4 samples burns are all in (0, 1)
    # So value=1 is an overestimate and value=0 is an underestimate
    # The first case results in negative advantages, as the second case
    # results in positive advantages.
    assert np.all(buf.buf_data['adv'][0:4:2] < 0) \
        and np.all(buf.buf_data['adv'][1:4:2] > 0) \
        and np.all(buf.buf_data['adv'][4] == 0) \
        and np.all(buf.buf_data['adv'][5:] == 0), \
            '{}'.format(buf.buf_data['adv'])
    
    # Verify the observation and action
    assert np.all(np.equal(
        buf.buf_data['obs'],np.array([0,1,2,3,4,0,0,0,0,0]))),\
            '{}'.format(buf.buf_data['obs'])
    assert np.all(np.equal(
        buf.buf_data['act'], np.array([0,1,0,2,3,0,0,0,0,0]))),\
            '{}'.format(buf.buf_data['act'])
    
    # Extract the first two elements manually, this should be the same as
    # the burn value of buf.get(2) 
    # NOTE: no convenient way to compare structured array
    ret_ref = buf.buf_data[:2].copy()
    new_ref = buf.buf_data[2:5].copy()
    ret = buf.get(length=2)
    new = buf.buf_data[:3]
    assert buf.buf_size == 3
    
    def compare_buf_data(a, b):
        assert np.all(np.equal(a['obs'], b['obs'])) \
            and np.all(np.equal(a['act'], b['act'])) \
            and np.all(np.equal(a['ext'], b['ext'])) \
            and np.all(np.equal(a['act'], b['act'])) \
            and np.all(np.equal(a['rew'], b['rew'])) \
            and np.all(np.equal(a['val'], b['val'])) \
            and np.all(np.equal(a['adv'], b['adv'])) \
            and np.all(np.equal(a['ret'], b['ret'])), \
            '{}, {}'.format(a, b)
    
    compare_buf_data(ret_ref, ret)
    compare_buf_data(new_ref, new)

    


