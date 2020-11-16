import logging

def auto_make_state_dict(obj, attr_names):
    """ Automatically creates a state dict
    If an attribute has a state_dict method, it is called to generate
    a sub state_dict. Otherwise, the attribute object is directly added.
    """
    sd = {}
    for name in attr_names:
        assert hasattr(obj, name), name
        attr = getattr(obj, name)
        if hasattr(attr, 'state_dict'):
            attr_sd = attr.state_dict()
            assert isinstance(attr_sd, dict)
            sd[name] = attr_sd
        else:
            sd[name] = attr
    return sd

def auto_load_state_dict(obj, sd):
    for attr_name, attr_data in sd.items():
        # Attr must exist. Otherwise, we may end up with
        # assigning the state dict to that attribute.
        assert hasattr(obj, attr_name)
        attr = getattr(obj, attr_name)
        if hasattr(attr, 'load_state_dict'):
            attr.load_state_dict(attr_data)
        else:
            #assert type(attr) == type(attr_data), '{}, {}'.format(attr, attr_data)
            setattr(obj, attr_name, attr_data)

def _make_object_summary(obj):
    has_more = False
    summary = str(obj)
    if '\n' in summary:
        has_more = True
        summary = summary.splitlines()[0]
    len_limit = 30
    if len(summary) > int(len_limit*1.5):
        has_more = True
        summary = summary[:len_limit]
    if has_more:
        summary += ' ...'
    return summary

def _show_state_dict_content_rec(name: str, obj, level: int) -> None:
    indent = '  ' * level
    if isinstance(obj, dict):
        logging.info('{}{}:'.format(indent, name))
        for key, val in obj.items():
            _show_state_dict_content_rec(key, val, level+1)
    else:
        summary = _make_object_summary(obj)
        logging.info('{}{}: {}'.format(indent, name, summary))

def show_state_dict_content(sd) -> None:
    """ Lists the entry names in a state dict recursively
    """
    _show_state_dict_content_rec('state dict', sd, 0) 
