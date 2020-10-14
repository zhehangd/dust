import os
import sys

from dust import _dust

class ProgressLog(object):
    
    def __init__(self):
        if _dust.inside_project():
            proj = _dust.project()
            progress_filename = '{}/progress/{}.progress.txt'.format(
                proj.proj_dir, proj.time_tag)
            os.makedirs(os.path.dirname(progress_filename), exist_ok=True)
            self._file = open(progress_filename, 'w')
        else:
            self._file = sys.stderr
        self._line_count = 0
        self._table = {}
    
    def set_fields(self, **kwargs):
        """ Set one or more fields to the current line
        Adding a new key in the first line will create a new field.
        Starting from the second line you can only add existing fields.
        """
        # We expect kwargs is ordered so the order of the fields
        # can be guranteed. This requires Python 3.7+ (+CPython 3.6).
        # Introducing new keys after the first line is disallowed.
        if self._line_count > 0:
            assert kwargs.keys() <= self._table.keys()
        self._table.update(kwargs)
    
    def finish_line(self):
        """ Dump collected fields in a line
        Unset fields are filled with 'n/a'.
        """
        if self._line_count == 0:
            head_line = '\t'.join([str(k) for k in self._table.keys()])
            assert not '\n' in head_line
            self._file.write(head_line + '\n')
        data_line = '\t'.join([self._format_val(v) for v in self._table.values()])
        assert not '\n' in data_line
        self._file.write(data_line + '\n')
        self._table = dict.fromkeys(self._table.keys(), 'n/a')
        self._line_count += 1
        
    def _format_val(self, val):
        return str(val)
