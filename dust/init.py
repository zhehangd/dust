import logging

from dust.core import project

if __name__ == '__main__':
    
    try:
        proj = project.create_project('init')
        logging.info('Initialize the project')
    except KeyboardInterrupt:
        logging.info('Interrupted by user')
    finally:
        logging.info('Abort')
    
    logging.info('End')
