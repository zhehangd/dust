import random
import time

import numpy as np 

import dust
from dust import rendering 

# Sleep after each render
# This allows us to control the actual FPS
_RENDER_SLEEP_TIME = 0.01

# Window size
_DISP_WIN_WIDTH = 800
_DISP_WIN_HEIGHT = 600

# Defines the dimensions for rendering
# By doing so we decouple the coordinate system used in rendering
# from the actual window size.
# The range of the x coordinates would be [0, _DISP_WORLD_WIDTH]
# and y coordinates be [0, _DISP_WORLD_HEIGHT].
_DISP_WORLD_SIZE = 100
_DISP_WORLD_WIDTH = _DISP_WORLD_SIZE * _DISP_WIN_WIDTH // _DISP_WIN_HEIGHT
_DISP_WORLD_HEIGHT = _DISP_WORLD_SIZE

_COLOR_GROUND = (0.75, 0.93, 0.73)
_COLOR_WALL = (0.2, 0.2, 0.2)
_COLOR_FOOD = (0.9,0.1,0.1)
_COLOR_PLAYER = (0.1,0.9,0.1)
# We use (sx, sy) to represents a point in the screen.
# The origin is at the bottom-left of the screen.

def _make_polygon(tl, br):
    return [tl, (br[0], tl[1]), br, (tl[0], br[1])]


class Disp(object):
    
    def __init__(self, core):
        self.core = core
        assert core.map_data.ndim == 2
        w, h = core.map_data.shape
        
        # Size of a square
        square_size = min(_DISP_WORLD_WIDTH / w, _DISP_WORLD_HEIGHT / h)
        
        # Top-left coordinate of the first square
        offset_coords = np.array([
          (_DISP_WORLD_WIDTH - w * square_size) // 2,
          (_DISP_WORLD_HEIGHT - h * square_size) // 2])
        
        self.offset_coords = offset_coords
        self.square_size = square_size
        self.viewer = None
        
    def render(self):
        if self.viewer is None:
            self.viewer = rendering.Viewer(_DISP_WIN_WIDTH, _DISP_WIN_HEIGHT)
            self.viewer.set_bounds(0, _DISP_WORLD_WIDTH, 0, _DISP_WORLD_HEIGHT)
        
        core = self.core
        w, h = core.map_data.shape
        
        global _COLOR_GROUND, _COLOR_WALL, _COLOR_FOOD
        
        ground_tl, _ = self._square_coords(0, 0)
        ground_br, _ = self._square_coords(w, h)
        self.viewer.draw_polygon(_make_polygon(ground_tl, ground_br), color=_COLOR_GROUND)
        
        wall_xi_list, wall_yi_list = np.where(core.map_data['type'] == 1)
        for xi, yi in zip(wall_xi_list, wall_yi_list):
            tl, br = self._square_coords(xi, yi)
            self.viewer.draw_polygon(_make_polygon(tl, br), color=_COLOR_WALL)
        
        for food_i in core.food_coords:
            xi = food_i // h
            yi = food_i % h
            t = rendering.Transform(translation=self._square_center_(xi, yi))
            radius = self.square_size / 8.0
            res = max(8, round(radius * 8))
            self.viewer.draw_circle(radius, res, color=_COLOR_FOOD).add_attr(t)
        
        for player_i in core.player_coords:
            xi = player_i // h
            yi = player_i % h
            t = rendering.Transform(translation=self._square_center_(xi, yi))
            radius = self.square_size / 5.0
            res = max(8, round(radius * 8))
            self.viewer.draw_circle(radius, res, color=_COLOR_PLAYER).add_attr(t)
        
        #t = rendering.Transform(translation=self._square_center_(*self.core.goal_pos))
        #self.viewer.draw_circle(1, 8, color=(0.9,0.1,0.1)).add_attr(t)
        
        #t = rendering.Transform(translation=self._square_center_(*self.core.player_pos))
        #self.viewer.draw_circle(1, 8, color=(0.2,0.8,0.2)).add_attr(t)
        
        if _RENDER_SLEEP_TIME > 0:
            time.sleep(_RENDER_SLEEP_TIME)
        
        mode='human'
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')
        
    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        
    def _square_coords(self, xi, yi):
        """ Calculates the top-left and the bottom-right coordinates of a cell
        Returns:
          ((x,y), (x,y)) 
        """
        w, h = self.core.map_data.shape
        sx = self.offset_coords[0] + self.square_size * xi
        sy = self.offset_coords[1] + self.square_size * (h - yi)
        tl = (sx, sy)
        br = (sx + self.square_size, sy - self.square_size)
        return tl, br
    
    def _square_center_(self, xi, yi):
        """ Calculates the center coordinates of a cell
        """
        tl, br = self._square_coords(xi, yi)
        return ((tl[0]+br[0])/2, (tl[1]+br[1])/2)
 
