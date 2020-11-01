import os
import re
import time

class Timer(object):
    def __init__(self, parent, name):
        self.name = name
        self.parent = parent
        
    def __enter__(self):
        self.t = time.time()
        
    def __exit__(self, exc_type, exc_value, traceback):
        self.t = time.time() - self.t
        if self.name in self.parent:
            self.parent[self.name] += self.t
        else:
            self.parent[self.name] = self.t

class FindTimestampedFile(object):
    def __init__(self, fdir, pattern):
        """
        Pattern must include one '*' to represent the timestamp.
        """
        timestamp_pattern = '([0-9]{4}-[0-9]{2}-[0-9]{2}-' \
                            '[0-9]{2}-[0-9]{2}-[0-9]{2})'
        assert pattern.count('*') == 1
        pattern = pattern.replace('*', timestamp_pattern)
        pattern = '^{}$'.format(pattern)
        matches = [  re.match(pattern, f) for f in os.listdir(fdir) ]
        matches = [ m for m in matches if m ] # remove the unmatched
        matches.sort(key=lambda m: m.groups()[0]) # sort by time
        self._matches = matches
        self._dir = fdir
        
    def get_latest_file(self):
        if not self._matches:
            raise RuntimeError('There is no file found')
        return os.path.join(self._dir, self._matches[-1].string)
