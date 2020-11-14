from dust.core.env import EnvCore, EnvAIStub

class AIEngine(object):
    """ Base class of AI engines
    """
    
    def __init__(self, ai_stub: EnvAIStub, freeze: bool):
        """
        TODO: AI engine shouldn't see EnvCore
        Args:
            env (EnvCore): environment core
            ai_stub (EnvAIStub): AI stub
            freeze (bool): Disable learning if True
        """
        pass
    
    def act(self) -> None:
        raise NotImplementedError()

    def update(self) -> None:
        raise NotImplementedError()
    
