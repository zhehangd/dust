from dust.utils import state_dict

class _ClassA(object):
    
    def __init__(self):
        self.x = 'foo'
        self.y = None

class _ClassB(object):
    
    def __init__(self):
        self.a = [1, 2, 3]
        self.b = {'aa': 1, 'bb': None}

class _ClassC(object):
    
    def __init__(self):
        self.p = []
        self.q = {'a':None, 'b':None}

    def state_dict(self):
        sd = state_dict.auto_make_state_dict(self, ['p', 'q'])
        sd['extra'] = 42
        return sd
    
    def load_state_dict(self, sd):
        return state_dict.auto_load_state_dict(self, sd)

def test_auto_make_state_dict_1():
    """ Tests simple attributes
    """
    obj = _ClassA()
    sd = state_dict.auto_make_state_dict(obj, ['x', 'y'])
    assert len(sd) == 2
    assert sd['x'] == 'foo'
    assert sd['y'] == None

def test_auto_make_state_dict_2():
    """ Tests container attributes
    """
    obj = _ClassB()
    sd = state_dict.auto_make_state_dict(obj, ['a', 'b'])
    assert len(sd) == 2
    assert sd['a'] == obj.a
    assert sd['b'] == obj.b


def test_auto_make_state_dict_3():
    """ Tests class with state_dict behaves as normal
    """
    obj = _ClassC()
    sd = state_dict.auto_make_state_dict(obj, ['p', 'q'])
    assert len(sd) == 2
    assert sd['p'] == obj.p
    assert sd['q'] == obj.q

def test_auto_make_state_dict_4():
    """ Tests regular class attribute
    """
    obj = _ClassA()
    obj.z = _ClassB()
    sd = state_dict.auto_make_state_dict(obj, ['x', 'y', 'z'])
    assert len(sd) == 3
    assert sd['x'] == obj.x
    assert sd['y'] == obj.y
    assert sd['z'] == obj.z # _ClassB instance is added directly

def test_auto_make_state_dict_5():
    """ Tests class with state_dict attribute
    """
    obj = _ClassA()
    obj.z = _ClassC()
    sd = state_dict.auto_make_state_dict(obj, ['x', 'y', 'z'])
    assert len(sd) == 3
    assert sd['x'] == obj.x
    assert sd['y'] == obj.y
    assert type(sd['z']) == dict
    z_sd = sd['z']
    assert len(z_sd) == 3
    assert z_sd['extra'] == 42

def test_auto_load_state_dict_1():
    """ Tests simple attributes
    """
    src = _ClassA()
    sd = state_dict.auto_make_state_dict(src, ['x', 'y'])
    #dst = 
    #state_dict.auto_load_state_dict(
    
    assert len(sd) == 2
    assert sd['x'] == 'foo'
    assert sd['y'] == None





