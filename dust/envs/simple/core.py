import numpy as np

MAP_WIDTH_ = 16
MAP_HEIGHT_ = 9
MAP_SIZE_ = (MAP_WIDTH_, MAP_HEIGHT_)
MAP_ELEMS_ = MAP_WIDTH_ * MAP_HEIGHT_

NUM_WALLS_ = MAP_ELEMS_ // 8
NUM_FOODS_ = MAP_ELEMS_ // 8
NUM_PLAYERS_ = 1 # NOTE: current implementation does not support #player > 1

MOVE_LUT_ = np.array([MAP_HEIGHT_, 1, -MAP_HEIGHT_, -1])

TYPE_WALL_ = 1

# Coordinte system:
# 

class Core(object):
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        global MAP_WIDTH_, MAP_HEIGHT_, MAP_SIZE_, MAP_ELEMS_
        global NUM_WALLS_, NUM_FOODS_, NUM_PLAYERS_
        
        num_objs = np.array([NUM_WALLS_, NUM_FOODS_, NUM_PLAYERS_])
        num_objs_cumsum = np.cumsum(num_objs)
        
        # Generate random coords in the plane except for the edges
        w = MAP_WIDTH_
        h = MAP_HEIGHT_
        rand_coords = np.random.permutation((w-2)*(h-2))[:num_objs_cumsum[-1]]
        rand_coords = rand_coords + 1 + rand_coords // (h-2) * 2 + h 
        wall_coords, food_coords, player_coords = np.split(
            rand_coords, num_objs_cumsum[:-1])
        
        # ground:0, wall:1
        map_dtype = [('type', 'i1')]
        map_data = np.zeros(MAP_SIZE_, dtype=map_dtype)
        map_data[0] = map_data[-1] = map_data[:,0] = map_data[:,-1] = TYPE_WALL_
        map_data_flat = map_data.reshape(-1)
        map_data_flat[wall_coords] = TYPE_WALL_
        self.map_data = map_data
        self.map_data_flat = map_data_flat
        
        self.player_coords = player_coords
        self.food_coords = food_coords
        
    def step(actions):
        global MOVE_LUT_, TYPE_WALL_
        move_coords = MOVE_LUT_[actions]
        move_success = np.equal(self.map_data_flat[move_coords], 0)
        player_coords[move_success] = move_coords[move_success]
        
        isect, _, food_idxs = np.intersect1d(
            player_coords, self.food_coords, return_indices=True)
        num_obtained_foods = len(isect)
        self.food_coords = np.delete(self.food_coords, food_idxs)
        
