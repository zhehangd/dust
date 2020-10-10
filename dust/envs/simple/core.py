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

# Coordinte system:
# The *coordinates* of a position is represented by (xi, yi), where xi and
# yi are both integers. The x-axis points toward right, and the y-axis points
# toward left. As we arrange columns along the first dimension of arrays,
# the (xi, yi) coordinates can directly access the corresponding elements.
# The *index* of a position is defined as xi * H + yi, which is used to
# access the elements of a flattened array.



class Core(object):
    
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
        rand_coords = np.random.permutation((w-2)*(h-2))[:num_objs_cumsum[-1]]
        rand_coords = rand_coords + 1 + rand_coords // (h-2) * 2 + h 
        wall_coords, food_coords, player_coords = np.split(
            rand_coords, num_objs_cumsum[:-1])
        
        # ground:0, wall:1
        map_dtype = [('type', 'i1')]
        map_data = np.zeros(_MAP_SIZE, dtype=map_dtype)
        map_data[0] = map_data[-1] = map_data[:,0] = map_data[:,-1] = _TYPE_WALL
        map_data_flat = map_data.reshape(-1)
        map_data_flat[wall_coords] = _TYPE_WALL
        self.map_data = map_data
        self.map_data_flat = map_data_flat
        
        self.wall_coords = wall_coords
        self.player_coords = player_coords
        self.food_coords = food_coords
        
    def step(actions):
        global _MOVE_LUT, _TYPE_WALL
        move_coords = _MOVE_LUT[actions]
        move_success = np.equal(self.map_data_flat[move_coords], 0)
        player_coords[move_success] = move_coords[move_success]
        
        isect, _, food_idxs = np.intersect1d(
            player_coords, self.food_coords, return_indices=True)
        num_obtained_foods = len(isect)
        self.food_coords = np.delete(self.food_coords, food_idxs)
    
