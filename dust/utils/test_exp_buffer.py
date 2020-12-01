from dust.utils.exp_buffer import ExpBuffer

def test_exp_buffer():
    """
    """
    obs_dtype = [('o','f4', 4)]
    act_dtype = 'i4'
    buf_size = 10
    buf = ExpBuffer(buf_size, obs_dtype, act_dtype)
    assert buf.buf_data.shape == (buf_size,)





