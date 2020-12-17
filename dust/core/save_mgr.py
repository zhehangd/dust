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

class SaveRecord(tuple):
  
    def __new__(cls, timestamp, tick, filename):
        assert isinstance(timestamp, str)
        assert isinstance(tick, int)
        assert isinstance(filename, str)
        return super(SaveRecord, cls).__new__(cls, (timestamp, tick, filename))
    
    @property
    def timestamp(self):
        return self[0]
    
    @property
    def tick(self):
        return self[1]
    
    @property
    def filename(self):
        return self[2]
        

class SaveManager(object):
    """ Manages the save and load of environments.
    
    Ideally, the manager makes a save every 'minor_save_interval' ticks,
    and and cleans the intermediate ones every 'major_save_interval' ticks.
    In practice, the manager does not force the user to save exactly
    at the expected tick. Instead, it only provides the expected tick
    of the next save, and it is up to the user when to call the save method.
    
    Keyword Arguments:
    
        save_dir (str | None): Directory of the save file (default: <proj>/saves)
        minor_save_interval (int): Minor save interval (default: configuration)
        major_save_interval (int): Major save interval (default: configuration)
        project (Project): Project
        start_tick (int):
        
    Attributes:
      save_dir (str): Directory of the saves
      
    
    """
    
    def __init__(self, **kwargs):
        proj = kwargs.pop('project')
        self._proj = proj
        self._saves = []
        self._save_dir = kwargs.pop('save_dir', os.path.join(proj.proj_dir, 'saves'))
        os.makedirs(self._save_dir, exist_ok=True)
        self._minor_save_interval = kwargs.pop('minor_save_interval', proj.cfg.minor_save_interval)
        self._major_save_interval = kwargs.pop('major_save_interval', proj.cfg.major_save_interval)
        assert isinstance(self._minor_save_interval, int)
        assert isinstance(self._major_save_interval, int)
        assert self._minor_save_interval > 0
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
        
        Searches the files that match the pattern in the save directory.
        The method will attempt to distinguish saves from different sequences
        and only take the latest one. It is okay if the saves are generated
        from different sessions (with different timestamps).
        
        This cleans the current save list and update it with the found saves.
        
        Return:
          The number of saves found in the directory.
        
        """
        
        pattern = 'save\\.({})\\.({})\\.pickle'.format(
            '[0-9]{4}-[0-9]{2}-[0-9]{2}-[0-9]{2}-[0-9]{2}-[0-9]{2}',
            '[0-9]+')
        matches = [re.match(pattern, f) for f in os.listdir(self._save_dir)]
        matches = [match for match in matches if match] # remove the unmatched

        # Saves are stored in a 3-tuple (timestamp, tick, filename)
        # We first sort them in descending order.
        
        def make_save_record(m):
            filename = os.path.join(self._save_dir, m.group(0))
            timestamp = m.group(1)
            tick = int(m.group(2))
            return SaveRecord(timestamp, tick, filename)
        
        saves = [make_save_record(m) for m in matches]
        saves.sort(reverse=True)

        # Now we check if the tick is strictly decreasing.
        # If not, they are likely more than one sequence of saves,
        # in which case we only take the latest one.
        end = next((i + 1 for i, curr, prev \
            in zip(range(len(saves)), saves, saves[1:]) \
            if curr.tick <= prev.tick), None)
        saves = saves[:end][::-1]
        
        if len(saves) == 0:
            return 0
        
        self._saves = saves
        self._start_tick = saves[-1].tick
        self._curr_tick = self._start_tick
        return len(saves)
    
    def get_save_list(self):
        """ Returns the list of managed save files
        """
        return [record[2] for record in self._saves]
    
    def load_latest_save(self):
        """ Loads the lastest managed save file
        """
        proj = self._proj
        filename = self._saves[-1].filename
        disp_filename = proj.relpath(filename)
        proj.log.info('Load the lastest save: {}'.format(disp_filename))
        return self._load_save(filename)
    
    def load_save(self, tick):
        raise NotImplementedError()
    
    def _load_save(self, save_filename):
        obj = pickle.loads(open(save_filename, 'rb').read())
        return obj
    
    def next_save_tick(self) -> int:
        return self._next_minor_save_tick(self._curr_tick)
    
    def save(self, tick, obj):
        """ Creates a new save and update the previous saves
        """
        assert self._curr_tick < tick
        proj = self._proj
        
        save_filename = self._make_filename(tick)
        proj.log.info('Save {}'.format(proj.relpath(save_filename)))
        os.makedirs(os.path.dirname(save_filename), exist_ok=True)
        with open(save_filename, 'wb') as f:
            pickle.dump(obj, f)
        # Remove intermediate minor saves
        if tick >= self._next_major_save_tick(self._curr_tick):
            min_tick = self._prev_major_save_tick(self._curr_tick)
            # Remove all saves satisfying tick >= min_tick except the first one
            remove_srt = next((i+1 for i, save in enumerate(self._saves) \
                if save.tick >= min_tick and save.tick > self._start_tick), \
                len(self._saves))
            proj.log.info("Remove minor saves in [{},{})".format(min_tick, tick))
            saves_to_remove = self._saves[remove_srt:]
            self._saves = self._saves[:remove_srt]
            
            if len(saves_to_remove) > 0:
                preserved_file = self._saves[-1].filename
                proj.log.info(" - Preserve {}".format(proj.relpath(preserved_file)))
                for save in saves_to_remove:
                    proj.log.info(" - Remove {}".format(proj.relpath(save.filename)))
                    os.remove(save.filename)
        self._saves.append(SaveRecord(self._proj.timestamp, tick, save_filename))
        self._curr_tick = tick
        return save_filename
    
    def _next_minor_save_tick(self, tick):
        """ Returns the next planned save tick after 'tick'.
        """
        interval = self._minor_save_interval
        return (tick // interval + 1) * interval
    
    def _prev_major_save_tick(self, tick):
        """ Returns the last planned save tick before or equal to 'tick'.
        """
        interval = self._major_save_interval
        return tick // interval * interval
    
    def _next_major_save_tick(self, tick):
        """ Checks a planned tick is a major save tick
        """
        interval = self._major_save_interval
        return (tick // interval + 1) * interval
    
    def _make_filename(self, tick):
        basename = 'save.{}.{}.pickle'.format(self._proj.timestamp, tick)
        return os.path.join(self._save_dir, basename)
