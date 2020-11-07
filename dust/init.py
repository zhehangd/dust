import logging

from dust.core import project

if __name__ == '__main__':
    
    try:
        proj = project.create_project('init')
        logging.info('Initialize the project')
    except KeyboardInterrupt:
        logging.info('Interrupted by user')
    #except BaseException as error:
    #    logging.info('An exception occurred: \n{}'.format(error))
    #    logging.info('Abort')
    
    logging.info('End')
