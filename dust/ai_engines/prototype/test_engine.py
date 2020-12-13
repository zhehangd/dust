import logging
import os

import numpy as np

from .engine import Engine, BrainDef, Terminal, Brain

def make_obs_data(o):
    obs = np.empty((), [('o', 'f4', len(o))])
    obs['o'] = np.asarray(o, dtype='f4')
    return obs

def test_brain():
    brain_def = BrainDef(5, 3, [])
    brain = Brain.create_new_instance(brain_def)
    state_dict = brain.state_dict()
    obs = make_obs_data([1.,0.,1.,0.,1.])
    exp = brain.evaluate(obs)
    assert np.array_equal(exp['obs'], obs)
    assert not np.array_equal(exp['ext']['logp'], 0)
    assert not np.array_equal(exp['val'], 0)
    
    # Save and load
    sd = brain.state_dict()
    brain = Brain.create_from_state_dict(sd)
    
    # Given the action, the brain should produce exactly
    # the same result.
    exp2 = brain.evaluate(exp['obs'], exp['act'])
    assert np.array_equal(exp['obs'], exp2['obs'])
    assert np.array_equal(exp['act'], exp2['act'])
    assert np.array_equal(exp['ext'], exp2['ext'])
    assert np.array_equal(exp['val'], exp2['val'])

def test_create_terminal():
    brain_def = BrainDef(5, 3, [])
    term = Terminal.create_new_instance("b1", brain_def)
    assert term.brain_name == "b1"
    assert term.buf is not None
    assert term.buf.buf_size == 0
    
def test_ai_engine():
    brain_name = 'brain_01'
    term_name = 'terminal_01'
    brain_def = BrainDef(5, 3, [])
    
    engine = Engine.create_new_instance()
    engine.add_brain(brain_name, brain_def)
    assert len(engine.brains) == 1
    
    engine.add_terminal(term_name, brain_name)
    assert len(engine.terminals) == 1
    
    t1_obs = engine.create_empty_obs(term_name)
    t1_obs['o'] = [1.,0.,1.,0.,1.]
    obs_dict = {term_name: t1_obs}
    exp_dict = engine.evaluate(obs_dict)
    assert len(exp_dict) == 1
    exp = exp_dict[term_name]
    assert np.array_equal(exp['obs'], t1_obs)
    assert not np.array_equal(exp['ext']['logp'], 0)
    assert not np.array_equal(exp['val'], 0)
    
    exp['rew'] = 1
    engine.add_experiences(exp_dict)
    
    
    
    
