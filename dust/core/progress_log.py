import os
import sys

from dust import _dust

class ProgressLog(object):
    """ Dumps training progress
    
    Currently one should create only one such instance in a process.
    
    This class helps an AI engine to dump the status of training.
    The information is orgnaized to facilitate plotting.
    
    """
    
    def __init__(self):
        """ Initializes the progress log
        
        Upon construction a progress log file is created, named with the 
        project time tag.
           
        """
        if _dust.inside_project():
            proj = _dust.project()
            progress_filename = '{}/logs/progress.{}.txt'.format(
                proj.proj_dir, proj.timestamp)
            os.makedirs(os.path.dirname(progress_filename), exist_ok=True)
            self._file = open(progress_filename, 'w')
        else:
            self._file = sys.stderr
        self._line_count = 0
        self._table = {}
    
    def set_fields(self, **kwargs):
        """ Assign one or more fields
        
        New field keys are allowed only before the first record is flushed.
        After that, only existing keys can be provided.
        The key must be a string and the value must be able to converted to a
        string. Both of them must only contain numbers, letters and underscores.
        
        """
        # We expect kwargs is ordered so the order of the fields
        # can be guranteed. This requires Python 3.7+ (+CPython 3.6).
        # Introducing new keys after the first line is disallowed.
        if self._line_count > 0:
            assert kwargs.keys() <= self._table.keys()
        self._table.update(kwargs)
    
    def finish_line(self):
        """ Flushes one record
        
        Unset fields are filled with '0'.
        """
        if self._line_count == 0:
            head_line = '\t'.join([str(k) for k in self._table.keys()])
            assert not '\n' in head_line
            self._file.write(head_line + '\n')
        data_line = '\t'.join([self._format_val(v) for v in self._table.values()])
        assert not '\n' in data_line
        self._file.write(data_line + '\n')
        self._table = dict.fromkeys(self._table.keys(), '0')
        self._line_count += 1
        
    def _format_val(self, val):
        return str(val)
