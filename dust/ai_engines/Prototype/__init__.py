import importlib

from dust import _dust
from dust.core.ai_engine import AIEngine, AIEngineRecord

class PrototypeAIEngineRecord(AIEngineRecord):
    """
    """
    
    @property
    def ai_engine(self) -> type(AIEngine):
        core_module = importlib.import_module(__package__ + '.prototype')
        return core_module.PrototypeAIEngine
    
    def register_args(self):
        pass

_dust.register_ai_engine("prototype", PrototypeAIEngineRecord())
 
