from dust.core.env import EnvCore, EnvAIStub

class AIEngine(object):
    """ Base class of AI engines
    """
    
    def __init__(self):
        """
        Args:
            ai_stub (EnvAIStub): AI stub
            freeze (bool): Disable learning if True
        """
        pass
    
    @classmethod
    def create_new_instance(cls, ai_stub, **kwargs) -> 'AIEngine':
        raise NotImplementedError()
    
    @classmethod
    def create_from_state_dict(cls, ai_stub, state_dict, **kwargs) -> 'AIEngine':
        raise NotImplementedError()
    
    def perceive_and_act(self) -> None:
        raise NotImplementedError()

    def update(self) -> None:
        raise NotImplementedError()
    
class AIEngineRecord(object):
    """
    """

    @property
    def ai_engine(self) -> type(AIEngine):
        raise NotImplementedError()
    
    def register_args(self):
        raise NotImplementedError()
        
