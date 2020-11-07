import logging
import time

import numpy as np

from dust.core import project
from dust.dev import simulation

if __name__ == '__main__':
    
    try:
        proj = project.load_project('train')
        logging.info('Starting training...')
        sim = simulation.SimulationDemo(True)
        sim.start()
    except KeyboardInterrupt:
        logging.info('Interrupted by user')
    #except BaseException as error:
    #    logging.info('An exception occurred: \n{}'.format(error))
    #    logging.info('Abort')
    
    logging.info('End')
