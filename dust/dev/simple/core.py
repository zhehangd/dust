import logging

import numpy as np

_MAP_WIDTH = 16
_MAP_HEIGHT = 9
_MAP_SIZE = (_MAP_WIDTH, _MAP_HEIGHT)
_MAP_ELEMS = _MAP_WIDTH * _MAP_HEIGHT

_NUM_WALLS = _MAP_ELEMS // 8
_NUM_FOODS = 5
_NUM_PLAYERS = 1 # NOTE: current implementation does not support #player > 1

_MOVE_LUT = np.array([_MAP_HEIGHT, 1, -_MAP_HEIGHT, -1])

_TYPE_WALL = 1

_REWARD_FOOD = 50

_MAX_TICKS_PER_EPOCH = 100

# Coordinte system:
# The *coordinates* of a position is represented by (xi, yi), where xi and
# yi are both integers. The x-axis points toward right, and the y-axis points
# toward left. As we arrange columns along the first dimension of arrays,
# the (xi, yi) coordinates can directly access the corresponding elements.
# The *index* of a position is defined as xi * H + yi, which is used to
# access the elements of a flattened array.

class Env(object):
    
    """
    
    Attributes:
    
    
    """
    
    def __init__(self):
        
        # Tick since the very beginning of the environment
        self.curr_tick = 0
        
        # Epoch number
        self.curr_epoch = 0
        
        self.new_epoch()
        
    def new_epoch(self):
        global _MAP_WIDTH, _MAP_HEIGHT, _MAP_SIZE, _MAP_ELEMS
        global _NUM_WALLS, _NUM_FOODS, _NUM_PLAYERS
        
        num_objs = np.array([_NUM_WALLS, _NUM_FOODS, _NUM_PLAYERS])
        num_objs_cumsum = np.cumsum(num_objs)
        
        # Generate random coords in the plane except for the edges
        w = _MAP_WIDTH
        h = _MAP_HEIGHT
        
        map_descr = '111111111100000001101101101100000101'\
                    '111010001100011011100010001101010001'\
                    '101000111101010101100000001100111011'\
                    '101100001101001101100000001111111111'
        map_descr = np.array(list(map_descr), dtype=np.int)
        wall_coords = np.where(map_descr == 1)[0]
        empty_coords = np.where(map_descr == 0)[0]
        np.random.shuffle(empty_coords)
        
        food_coords = empty_coords[:_NUM_FOODS]
        player_coords = empty_coords[_NUM_FOODS:_NUM_FOODS+_NUM_PLAYERS]
        
        self.map_shape = (w, h)
        
        self.wall_coords = wall_coords
        self.player_coords = player_coords
        self.food_coords = food_coords
        self.move_count = np.zeros(_MAP_ELEMS, dtype=int)
        
        self.tick_reward = 0
        self.epoch_score = 0
        self.epoch_reward = 0
        
        # Epoch tick reset every epoch
        self.curr_epoch_tick = 0
        
        # True after the environment evolves to the
        # termination state, and is reset in the subsequent
        # next phase
        self.epoch_end = False
        
        self.ticks_per_epoch = _MAX_TICKS_PER_EPOCH
        
        # Next action for each player
        # Filled by agents between steps
        self._reset_action()
    
    def _reset_action(self):
        self.next_action = np.zeros(_NUM_PLAYERS, dtype='i1')
        
    def evolve(self):
        actions = self.next_action
        move_coords = self.player_coords + _MOVE_LUT[actions]
        
        move_success = np.isin(move_coords, self.wall_coords,
                               assume_unique=True, invert=True)
        
        num_colision = np.count_nonzero(np.logical_not(move_success))
        
        self.player_coords[move_success] = move_coords[move_success]
        
        self.move_count[self.player_coords] += 1
        
        _, feed_player_idxs, feed_food_idxs = np.intersect1d(
            self.player_coords, self.food_coords, return_indices=True)
        num_obtained_foods = len(feed_player_idxs)
        self.food_coords = np.delete(self.food_coords, feed_food_idxs)
        
        self.tick_reward = 0
        self.tick_reward += _REWARD_FOOD * num_obtained_foods
        #self.tick_reward -= np.sum(np.maximum(0, self.move_count[self.player_coords] - 2))
        self.tick_reward -= num_colision
        
        #logging.info(self.tick_reward)
        self.epoch_reward += self.tick_reward
        self.epoch_score += self.tick_reward
        
        assert self.curr_epoch_tick <= _MAX_TICKS_PER_EPOCH - 1
        if self.curr_epoch_tick == _MAX_TICKS_PER_EPOCH - 1:
            self.epoch_end = True
        #logging.info('evolve: {}'.format(self.curr_epoch_tick))
        return self.epoch_score
    
    def next_tick(self):
        """ Ends the current tick and moves to the next one
        """
        self.curr_tick += 1 
        self.curr_epoch_tick += 1
        self._reset_action()
        if self.epoch_end == True:
            self.new_epoch()
            self.curr_epoch += 1
        #logging.info('next_tick: {}'.format(self.curr_epoch_tick))
    
    def set_action(self, action):
        """ Temp function used by agents to set actions
        """
        self.next_action[:] = action
    
    
