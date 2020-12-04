import logging
import os

import numpy as np

from dust.core.ai_engine_dev import AIEngineDev, BrainDef, Terminal, Brain

def make_obs_data(o):
    obs = np.empty((), [('o', 'f4', len(o))])
    obs['o'] = np.asarray(o, dtype='f4')
    return obs
    

def test_create_brain():
    brain_def = BrainDef(5, 3, [])
    brain = Brain.create_new_instance(brain_def)
    state_dict = brain.state_dict()
    exp = brain.evaluate(make_obs_data([1.,0.,1.,0.,1.]))

def test_create_terminal():
    brain_def = BrainDef(5, 3, [])
    brain = Brain.create_new_instance(brain_def)
    term = Terminal.create_new_instance(brain)

def test_ai_engine():
    brain_name = 'brain_01'
    term_name = 'terminal_01'
    brain_def = BrainDef(5, 3, [])
    
    engine = AIEngineDev()
    engine.add_brain(brain_name, brain_def)
    assert len(engine.brains) == 1
    
    engine.add_terminal(term_name, brain_name)
    assert len(engine.terminals) == 1
    
