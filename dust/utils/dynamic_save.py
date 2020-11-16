import logging
import os

class DynamicSave(object):
    
    def __init__(self, start, period, max_n):
        assert max_n > 0
        assert max_n % 2 == 0
        self._curr_tick = start
        self._period = period
        self._max_n = max_n
        self._files = []
    
    @property
    def next_update_tick(self) -> int:
        """ Next tick a save should be given
        """
        return self._curr_tick + self._period
    
    def add_save(self, filename: str, tick: int = None):
        """ Adds a save file under its management
        
        It is assumed that the file is added at the time returned
        by ``next_update_tick``. An optional parameter ``tick`` can be
        given to let this method check if it satisifies the condition.
        Calling this function trigger the check rountine and may remove
        the obsolete files.
        
        Args:
            filename (str): Path to the file to be managed
            tick (int): The current tick. 
        """
        if tick is not None:
            assert tick == self.next_update_tick
        
        self._files.append(filename)
        num_files = len(self._files)
        assert num_files <= self._max_n
        
        self._curr_tick = self.next_update_tick
        
        if num_files == self._max_n:
            files_to_remove = self._files[::2]
            assert filename not in files_to_remove
            self._period *= 2
            self._files = self._files[1::2]
            assert filename in self._files
            logging.info('Removing the following saves:')
            for f in files_to_remove:
                logging.info(' - {}'.format(f))
                os.remove(f)
