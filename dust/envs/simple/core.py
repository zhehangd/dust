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

def _extract_view(src, pos, vr, batched=False):
    """
    Args:
      src N-d source array for view extraction.
      pos Coordinates of the view center.
          If batched is False it is an 1-d array of len k, where k <= n.
          If batched is True it is a 2-d array where the 1st dimension is batch
      vr int radius of the view
    """
    if batched:
        bsize = len(pos)
        vsize = len(pos[0])
    else:
        bsize = 1
        vsize = len(pos)
        pos = (pos,)
    
    def make_src_dst_slices(x, w, vr):
        ss = x - vr
        se = x + vr + 1 - w
        ds = de = None
        if ss < 0: ss, ds = ds, -ss
        if se > 0: se, de = de, -se
        if se == 0: se = None
        return slice(ss, se), slice(ds, de)
    
    dst_shape = (bsize,) + (2*vr+1,) * vsize + src.shape[vsize:]
    dst = np.zeros(dst_shape, src.dtype)
    for k in range(bsize):
        src_idxs = []
        dst_idxs = []
        for i in range(vsize):
            p = pos[k][i]
            assert p >= 0 and p < src.shape[i], '{}, {}, {}'.format(p, i, src.shape)
            src_idx, dst_idx = make_src_dst_slices(p, src.shape[i], vr)
            src_idxs.append(src_idx)
            dst_idxs.append(dst_idx)
        src_idxs = tuple(src_idxs)
        dst_idxs = tuple(dst_idxs)
        dst[k][dst_idxs] = src[src_idxs]
    if not batched:
        dst = dst[0]
    return dst


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
        self.map_data = map_data
        self.map_data_flat = map_data_flat
        
        self.wall_coords = wall_coords
        self.player_coords = player_coords
        self.food_coords = food_coords
        
        self.step_reward = 0
        self.total_reward = 0
        self.curr_time = 0
        
        # Next action for each player
        # Filled by agents between steps
        self._reset_action()
    
    def _reset_action(self):
        self.next_action = np.zeros(_NUM_PLAYERS, dtype='i1')
        
    def step(self):
        global _MOVE_LUT, _TYPE_WALL
        actions = self.next_action
        move_coords = self.player_coords + _MOVE_LUT[actions]
        move_success = np.equal(self.map_data_flat['type'][move_coords], 0)
        print(move_success, actions, move_coords)
        self.player_coords[move_success] = move_coords[move_success]
        
        isect, _, food_idxs = np.intersect1d(
            self.player_coords, self.food_coords, return_indices=True)
        num_obtained_foods = len(isect)
        self.food_coords = np.delete(self.food_coords, food_idxs)
        
        self.step_reward = 0
        self.step_reward += _REWARD_FOOD * num_obtained_foods
        self.total_reward += self.step_reward
        
        self.curr_time += 1 
        self._reset_action()
        return self.total_reward
    
    def index_to_coords(self, idxs):
        h = _MAP_HEIGHT
        return np.stack((idxs // h, idxs % h), -1)
    
    def extract_obs(self, pos, vr, batched):
        assert pos.ndim == 2
        return _extract_view(self.map_data, pos, vr, batched)
    
    def set_action(self, action):
        self.next_action[:] = action
    
    
