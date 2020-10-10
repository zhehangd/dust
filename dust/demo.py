import numpy as np
import time

import dust.envs

core = dust.envs.simple.Core()
disp = dust.envs.simple.Disp(core)
disp.render()
print(core.map_data)
for i in range(1000):
    
    num_players = len(core.player_coords)
    actions = np.random.randint(0, 4, num_players)
    core.step(actions)
    
    disp.render()
    time.sleep(0.1)
