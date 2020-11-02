import logging

import numpy as np

from dust.core.env import BaseEnv

_MAP_WIDTH = 16
_MAP_HEIGHT = 9
_MAP_SIZE = (_MAP_WIDTH, _MAP_HEIGHT)
_MAP_ELEMS = _MAP_WIDTH * _MAP_HEIGHT

_MOVE_LUT = np.array([_MAP_HEIGHT, 1, -_MAP_HEIGHT, -1])

_TYPE_WALL = 1

_REWARD_FOOD = 50

_MAX_TICKS_PER_ROUND = 100

# Coordinte system:
# The *coordinates* of a position is represented by (xi, yi), where xi and
# yi are both integers. The x-axis points toward right, and the y-axis points
# toward left. As we arrange columns along the first dimension of arrays,
# the (xi, yi) coordinates can directly access the corresponding elements.
# The *index* of a position is defined as xi * H + yi, which is used to
# access the elements of a flattened array.

class Env(BaseEnv):
    
    """
    
    Attributes:
    
    
    """
    
    def __init__(self):
        super().__init__()
        
    def _create_new_round(self):
        
        global _MAP_WIDTH, _MAP_HEIGHT, _MAP_SIZE, _MAP_ELEMS
        global _NUM_WALLS, _NUM_FOODS, _NUM_PLAYERS
        
        # Generate random coords in the plane except for the edges
        w = _MAP_WIDTH
        h = _MAP_HEIGHT
        
        map_descr = '111111111''100000001''101101101''100111001' \
                    '110000011''111101111''100000011''101101111' \
                    '100101001''110111101''100000101''100000101' \
                    '100000101''100000101''100000001''111111111'
        map_descr = np.array(list(map_descr), dtype=np.int)
        wall_coords = np.where(map_descr == 1)[0]
        #empty_coords = np.where(map_descr == 0)[0]
        #np.random.shuffle(empty_coords)
        
        #food_coords = empty_coords[:_NUM_FOODS]
        #player_coords = empty_coords[_NUM_FOODS:_NUM_FOODS+_NUM_PLAYERS]
        
        food_coords = np.array([16,  22,  60,  78,  95, 111, 127])
        #player_coords = np.array([10])
        #player_coords = np.array([97])
        player_coords = np.array([88])
        
        self.map_shape = (w, h)
        
        self.wall_coords = wall_coords
        self.player_coords = player_coords
        self.food_coords = food_coords
        self.move_count = np.zeros(_MAP_ELEMS, dtype=int)
        
        self.tick_reward = 0
        self.round_reward = 0
        
        # Epoch tick reset every epoch
        self.curr_round_tick = 0
        
        self.num_round_collisions = 0
        
        # Flag indicating a round is ended.
        # This flag is refreshed by 'evolve' every tick.
        # 'next_tick' checks this flag and resets the environment if true.
        # One may manually set this flag between 'evolve' and 'next_tick'
        # to trigger the resetting.
        self.end_of_round = False
        
        self.ticks_per_round = _MAX_TICKS_PER_ROUND
        
        # Next action for each player
        # Filled by agents between steps
        self._reset_action()
    
    def _reset_action(self):
        self.next_action = np.zeros(1, dtype='i1')
        
    def evolve(self):
        actions = self.next_action
        move_coords = self.player_coords + _MOVE_LUT[actions]
        
        move_success = np.isin(move_coords, self.wall_coords,
                               assume_unique=True, invert=True)
        
        num_collision = np.count_nonzero(np.logical_not(move_success))
        self.num_round_collisions += num_collision
        
        self.player_coords[move_success] = move_coords[move_success]
        
        self.move_count[self.player_coords] += 1
        
        _, feed_player_idxs, feed_food_idxs = np.intersect1d(
            self.player_coords, self.food_coords, return_indices=True)
        num_obtained_foods = len(feed_player_idxs)
        self.food_coords = np.delete(self.food_coords, feed_food_idxs)
        
        self.tick_reward = 0
        self.tick_reward += _REWARD_FOOD * num_obtained_foods
        #self.tick_reward -= np.sum(np.maximum(0, self.move_count[self.player_coords] - 2))
        self.tick_reward -= num_collision
        
        assert self.curr_round_tick <= _MAX_TICKS_PER_ROUND - 1
        if self.curr_round_tick == _MAX_TICKS_PER_ROUND - 1:
            self.end_of_round = True
            
        if self.player_coords[0] == 78:
            self.end_of_round = True
            self.tick_reward += round(200 * (1.0 - float(self.curr_round_tick)/_MAX_TICKS_PER_ROUND))
        
        if self.player_coords[0] == (127+4):
            self.end_of_round = True
            self.tick_reward -= 200
        
        #logging.info(self.tick_reward)
        self.round_reward += self.tick_reward
    
    def set_action(self, action):
        """ Temp function used by agents to set actions
        """
        self.next_action[:] = action
    
    
