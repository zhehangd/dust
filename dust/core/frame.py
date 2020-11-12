

env = create_env("env01")

f_env = EnvFrame(env.core)
f_ai = AIFrame(env.ai_stub)
f_disp = DispFrame(f_env, f_ai)

disp.render()
while True:
    
    f_ai.perceive_and_act()
    
    f_env.evolve()
    
    f_ai.update()
    
    f_env.next_tick()


class EnvFrame(object):
    
    def __init__(self):
        pass
    
    def evolve(self):
        pass
    
    def update(self):
        pass

class AIFrame(object):
    """ Entrypoint of AI system
    """
    
    def __init__(self, env):
        pass
    
    def perceive_and_act(self):
        """ Perceives the environment and takes action
        """
        pass

    def update(self):
        """ Receives environment feedback and updates AI
        """
        pass

    
