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
    # the observed return is just
    # R(t) = 0.9**(4-t)
    for t in range(5):
        assert buf.buf_data['ret'][t] == pytest.approx(0.9**(4-t)),\
            '{}'.format(buf.buf_data['ret'])
    
    # For the first 4 samples returns are all in (0, 1)
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
    # the return value of buf.get(2) 
    # NOTE: no convenient way to compare structured array
    ret_ref = buf.buf_data[:2].copy()
    rem_ref = buf.buf_data[2:5].copy()
    ret = buf.get(length=2)
    rem = buf.buf_data[:3].copy()
    
    # Three samples left
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
    compare_buf_data(rem_ref, rem)
    
    # Add 6 new experience samples
    # We have retrieved 2 samples so 11 samples in total shouldn't
    # make the buffer full.
    buf.store(5, 0, 0.8, 1, 1)
    buf.store(6, 0, 0.7, 0, 1)
    buf.store(7, 0, 0.6, 1, 1)
    buf.store(8, 1, 0.5, 0, 1)
    buf.store(9, 1, 0.4, -5, 1)
    buf.store(0, 1, 0.3, 5, 1)
    
    # Let's retrieve a slice that includes the old path and a part of
    # the new path without finishing the new path first.
    # Retrieve 5 samples in the new path, so only the last one is excluded.
    ret = buf.get(length=3+5)
    
    # The first 3 samples should be exactly the same as before
    compare_buf_data(rem, ret[:3])
    
    # Check observation
    assert np.all(np.equal(ret['obs'], np.arange(2, 10)))
    
    # The last reward will not be counted into the return, so
    # the returns in that path should be all negative.
    ret['ret'][3:] < 0
    
    assert buf.buf_size == 1
