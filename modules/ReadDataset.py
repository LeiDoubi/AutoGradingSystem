import numpy as np
E_x_tilde_x_tilde_T = 0
assert(np.all(np.abs(
    E_x_tilde_x_tilde_T
    - np.zeros_like(E_x_tilde_x_tilde_T) < 1e-3), 'Result is incorrect!')
