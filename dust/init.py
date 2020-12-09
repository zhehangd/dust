import logging

from dust import _dust

if __name__ == '__main__':
    
    try:
        proj = _dust.create_project('init')
        proj.save_project()
        logging.info('Initialized the project')
    except KeyboardInterrupt:
        logging.info('Interrupted by user')
    
    logging.info('End')
