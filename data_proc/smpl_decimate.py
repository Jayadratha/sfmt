import os, sys 
sys.path.append(sys.path[0]+"/../")
import pymeshlab
from global_var import *
from utils.o3d_wrapper import Mesh
from tqdm import tqdm

obj_dir = "/raid/t2/sfpose_transfer/data/amass"
obj_remesh_dir = "/raid/t2/sfpose_transfer/data/amass"
name = "male_rest_T"
obj_path = obj_dir + '/' + name + '.obj'

ms = pymeshlab.MeshSet()
ms.load_new_mesh(obj_path)
face_num = 5510
ms.remove_isolated_pieces_wrt_face_num(mincomponentsize=25, removeunref=True)
if face_num > 0:
    print("\n----decimating | ", obj_path)
    ms.simplification_quadric_edge_collapse_decimation(targetfacenum=face_num, autoclean=True)
m = ms.current_mesh()
vs = m.vertex_matrix()
ff = m.face_matrix()

Mesh(v=vs, f=ff).write_obj(obj_remesh_dir + '/' + name + '_simplify.obj')
# ms.save_current_mesh(obj_remesh_dir + '/' + name + '_simplify.obj', save_vertex_normal=False)