import numpy as np

_MAP_WIDTH = 16
_MAP_HEIGHT = 9
_MAP_SIZE = (_MAP_WIDTH, _MAP_HEIGHT)
_MAP_ELEMS = _MAP_WIDTH * _MAP_HEIGHT

_NUM_WALLS = _MAP_ELEMS // 8
_NUM_FOODS = _MAP_ELEMS // 8
_NUM_PLAYERS = 1 # NOTE: current implementation does not support #player > 1

_MOVE_LUT = np.array([_MAP_HEIGHT, 1, -_MAP_HEIGHT, -1])

_TYPE_WALL = 1

_REWARD_FOOD = 50

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
        self.reset()
        
    def reset(self):
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
        
        # ground:0, wall:1
        map_dtype = [('type', 'i1')]
        map_data = np.zeros(_MAP_SIZE, dtype=map_dtype)
        map_data_flat = map_data.reshape(-1)
        map_data_flat[wall_coords] = _TYPE_WALL
        
        self.map_shape = (w, h)
        self.map_data = map_data
        self.map_data_flat = map_data_flat
        
        self.wall_coords = wall_coords
        self.player_coords = player_coords
        self.food_coords = food_coords
        
        self.tick_reward = 0
        self.total_reward = 0
        self.curr_step = 0
        self.curr_epoch = 0
        self.epoch_end = False
        
        #self.world_reset_time = 200
        
        # Next action for each player
        # Filled by agents between steps
        self._reset_action()
    
    def _reset_action(self):
        self.next_action = np.zeros(_NUM_PLAYERS, dtype='i1')
        
    def evolve(self):
        global _MOVE_LUT, _TYPE_WALL
        actions = self.next_action
        move_coords = self.player_coords + _MOVE_LUT[actions]
        move_success = np.equal(self.map_data_flat['type'][move_coords], 0)
        #print(move_success, actions, move_coords)
        self.player_coords[move_success] = move_coords[move_success]
        
        isect, _, food_idxs = np.intersect1d(
            self.player_coords, self.food_coords, return_indices=True)
        num_obtained_foods = len(isect)
        self.food_coords = np.delete(self.food_coords, food_idxs)
        
        self.tick_reward = 0
        self.tick_reward += _REWARD_FOOD * num_obtained_foods
        self.total_reward += self.tick_reward
        
        if self.curr_step == 200:
            self.epoch_end = True
        
        return self.total_reward
    
    def next(self):
        self.curr_step += 1 
        self._reset_action()
        
        if self.epoch_end == True:
            self.reset()
    
    def set_action(self, action):
        self.next_action[:] = action
    
    
