
def auto_make_state_dict(obj, symbol_names):
    sd = {}
    for name in symbol_names:
        assert hasattr(obj, name)
        attr = getattr(obj, name)
        if hasattr(attr, 'state_dict'):
            attr_sd = attr.state_dict()
            assert isinstance(attr_sd, dict)
            sd[name] = attr_sd
        else:
            sd[name] = attr
    return sd

def auto_load_state_dict(obj, sd):
    for attr_name, attr_data in sd:
        is_done = False
        if hasattr(obj, attr_name):
            attr = getattr(obj, attr_name)
            if hasattr(attr, 'load_state_dict'):
                attr.load_state_dict(attr_data)
                is_done = True
        if not is_done:
            setattr(obj, attr_name, attr_data)
            
