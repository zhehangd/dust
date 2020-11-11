from dust.utils.arg_cfg_parse import ArgCfgParser

def test_ArgCfgParser():

    PARSER = ArgCfgParser()

    PARSER.add_argument('--proj_name', default='unnamed')
    PARSER.add_argument('--proj_dir', default='foo')
    PARSER.add_argument('--version', '-v')
    PARSER.add_configuration('--use_cpu', type=bool)
    PARSER.add_configuration('--use_cuda', type=bool)
    PARSER.add_configuration('--use_tpu', type=bool)
    PARSER.add_configuration('--lr', '--rate', type=float, default=1e-5)
    PARSER.add_configuration('--lam', type=float, default=0.99)

    args, cfg = PARSER.parse_args(
        ['--proj_name', 'name', '--use_cpu', '--no-use_cuda',
        '--rate', '3.14'])

    assert hasattr(args, 'proj_name')
    assert args.proj_name == 'name'
    assert hasattr(args, 'proj_dir')
    assert args.proj_dir == 'foo'
    assert hasattr(args, 'version')
    assert args.version == None
    assert not hasattr(args, 'v')

    assert hasattr(cfg, 'use_cpu')
    assert cfg.use_cpu == True
    assert hasattr(cfg, 'use_cuda')
    assert cfg.use_cuda == False
    assert not hasattr(cfg, 'use_tpu')
    assert not hasattr(cfg, 'rate')
    assert hasattr(cfg, 'lr')
    assert cfg.lr == 3.14
    assert not hasattr(cfg, 'lam')

    PARSER.fill_default_config(cfg)
    assert hasattr(cfg, 'lam')
    assert cfg.lam == 0.99
    assert hasattr(cfg, 'use_tpu')
    assert cfg.use_tpu == None

