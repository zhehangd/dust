import logging
import os
import re
import pickle

from dust import _dust

_ARGPARSER = _dust.argparser()

_ARGPARSER.add_configuration('--minor_save_interval', type=int, default=10000,
    help='Interval between each save.')

_ARGPARSER.add_configuration('--major_save_interval', type=int, default=100000,
    help='Interval between each major save.')

# TODO: scan existing saves, ignore 

class SaveManager(object):
    """ Manages the save and load of environments.
    
    Ideally, the manager makes a save every 'minor_save_interval' ticks,
    and and cleans the intermediate ones every 'major_save_interval' ticks.
    In practice, the manager does not force the user to save exactly
    at the expected tick. Instead, it only provides the expected tick
    of the next save, and it is up to the user when to call the save method.
    
    """
    
    def __init__(self, **kwargs):
        proj = kwargs.pop('project', None) or _dust.project()
        self._proj = proj
        self._saves = []
        self._save_dir = kwargs.pop('save_dir', os.path.join(proj.proj_dir, 'saves'))
        self._minor_save_interval = kwargs.pop('minor_save_interval', proj.cfg.minor_save_interval)
        self._major_save_interval = kwargs.pop('major_save_interval', proj.cfg.major_save_interval)
        assert self._major_save_interval > self._minor_save_interval
        self._start_tick = kwargs.pop('start_tick', 0)
        self._curr_tick = self._start_tick
        assert len(kwargs) == 0, \
            'Unknown arguments {}'.format(', '.join(kwargs.keys()))
    
    @property
    def save_dir(self):
        return self._save_dir
    
    def scan_saves(self):
        """ Finds and records existing saves in the save directory
        """
        
        pattern = 'save\.({})\.({})\.pickle'.format(
            '[0-9]{4}-[0-9]{2}-[0-9]{2}-[0-9]{2}-[0-9]{2}-[0-9]{2}',
            '[0-9]+')
        matches = [re.match(pattern, f) for f in os.listdir(self._save_dir)]
        matches = [match for match in matches if match] # remove the unmatched

        # Saves are stored in a 3-tuple (timestamp, tick, filename)
        # We first sort them in descending order.
        saves = [(m.group(1), int(m.group(2)), m.group(0)) for m in matches]
        saves.sort(reverse=True)

        # Now we check if the tick is strictly decreasing.
        # If not, they are likely more than one sequence of saves,
        # in which case we only take the latest one.
        end = next((i + 1 for i, curr, prev \
            in zip(range(len(saves)), saves, saves[1:]) \
            if curr[1] <= prev[1]), None)
        saves = saves[:end][::-1]
        self._saves = saves
        return len(saves)
    
    def get_save_list(self):
        return [record[2] for record in self._saves]
    
    def load_latest_save(self):
        return self._load_save(self._saves[-1])
    
    def load_save(self, tick):
        raise NotImplementedError()
    
    def _load_save(self, save_filename):
        obj = pickle.loads(open(save_filename, 'rb').read())
        return obj
    
    def next_save_tick(self) -> int:
        return self._next_minor_save_tick(self._curr_tick)
    
    def save(self, tick, obj):
        assert self._curr_tick < tick
        
        save_filename = self._make_filename(tick)
        os.makedirs(os.path.dirname(save_filename), exist_ok=True)
        with open(save_filename, 'wb') as f:
            pickle.dump(obj, f)
        
        self._saves.append((self._proj.timestamp, tick, save_filename))
        
        if tick >= self._next_major_save_tick(self._curr_tick):
            # Remove intermediate minor saves
            min_tick = self._prev_major_save_tick(self._curr_tick) # closed
            # find all saves in [min_tick, tick)
            # Remove all expcet the first one, if exists
        
        self._curr_tick = tick
    
    def _next_minor_save_tick(self, tick):
        """ Returns the next planned save tick after 'tick'.
        """
        interval = self._minor_save_interval
        return (tick // interval + 1) * interval
    
    def _prev_major_save_tick(self, tick):
        """ Returns the last planned save tick before or equal to 'tick'.
        """
        interval = self._minor_save_interval
        return tick // interval * interval
    
    def _next_major_save_tick(self, tick):
        """ Checks a planned tick is a major save tick
        """
        interval = self._major_save_interval
        return (tick // interval + 1) * interval
    
    def _make_filename(self, tick):
        basename = 'save.{}.{}.pickle'.format(self._proj.timestamp, tick)
        return os.path.join(self._save_dir, basename)
