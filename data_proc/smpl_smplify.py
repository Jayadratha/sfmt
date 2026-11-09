import os, sys
sys.path.append(sys.path[0]+"/../")
import numpy as np
from utils.o3d_wrapper import Mesh
from utils.geometry import get_nearest_face, barycentric
from global_var import *


if __name__ == '__main__':
    m_high = Mesh(filename="/raid/t2/sfpose_transfer/data/smplh/male_rest_T.obj")
    m_low = Mesh(filename="/raid/t2/sfpose_transfer/data/smplh/male_rest_T_simplify.obj")
    nearest_face = get_nearest_face(m_low.v, m_high.v, m_high.f)
    bary = barycentric(m_low.v, m_high.v[m_high.f[:, 0][nearest_face]],
                       m_high.v[m_high.f[:, 1][nearest_face]],
                       m_high.v[m_high.f[:, 2][nearest_face]])
    np.savez(f"{SMPLH_PATH}/simplify.npz", nearest_face=nearest_face, bary=bary, f=m_low.f)