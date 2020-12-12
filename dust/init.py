import logging

from dust import _dust

_ARGPARSER = _dust.argparser()

_ARGPARSER.add_configuration('--env', default="env01",
    help="Environment to simulate.")

if __name__ == '__main__':
    
    try:
        proj = _dust.create_project('init')
        proj.parse_args()
        proj.log_proj_info()
        proj.save_project()
        logging.info('Initialized the project')
    except KeyboardInterrupt:
        logging.info('Interrupted by user')
    
    logging.info('End')
