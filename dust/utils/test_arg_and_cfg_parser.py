from dust.utils import arg_and_cfg_parser

class TestArgumentAndConfigParser(object):
  
    def __init__(self):
        parser = arg_and_cfg_parser.ArgumentAndConfigParser()
        parser.add_argument('--arg1')
        parser.add_argument('--arg2')
        parser.add_argument('--arg3', action='store_true')
        parser.add_configuration('--cfg1')
        parser.add_configuration('--cfg2', type=int, default=42)
        parser.add_configuration('--cfg3', action='store_true')

        parser.add_argument('--dash-arg')

        parser.add_configuration('--foo', default="Bar")

        parser.add_argument('--faa', type=int)

        parser.add_configuration('--flag', action='store_true')
        
        self.parser = parser

    def test_simple_arg(self):
        args, cfg = parser.parse_args(['--arg1', 'foo'])
        assert args.arg1 == 'foo'
        assert args.arg2 == None
        assert args.arg3 == False
        assert cfg.cfg1 == None
        assert cfg.cfg2 == 42
        assert hasattr(cfg, 'arg1') == False
        assert hasattr(cfg, 'arg2') == False
        assert hasattr(args, 'cfg1') == False
        assert hasattr(args, 'cfg2') == False
    
    def test_namespace(self):
        cfg_namespace = arg_and_cfg_parser.Namespace()
        cfg_namespace.foo = 123 # not defined in parser, should be preserved
        cfg_namespace.cfg2 = 24
        
        raw_args = ["--arg1", "abc", '--arg3', '--cfg1', 'def']
        args, cfg = parser.parse_args(raw_args, cfg_namespace=cfg_namespace)
        
        assert args.arg1 == abc
        assert args.arg2 == None
        assert args.arg3 == True
        assert cfg.foo == 123
        assert cfg.cfg1 == 'def'
        assert cfg.cfg2 == 24
        assert cfg.cfg3 == False
