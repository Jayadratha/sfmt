import os, sys 
sys.path.append(sys.path[0]+"/../")
import pymeshlab
from global_var import *
from utils.o3d_wrapper import Mesh
from tqdm import tqdm

obj_dir = MIXAMO_PATH + '/Mixamo_obj_Char'
obj_remesh_dir = MIXAMO_SIMPLIFY_PATH + '/obj_remesh'
names = [k.replace('.obj', '') for k in os.listdir(obj_dir) if k.endswith('.obj')]
os.makedirs(obj_remesh_dir, exist_ok=True)

# simplify using pymeshlab
for name in tqdm(names):
    print(name)
    ms = pymeshlab.MeshSet()
    obj_path = obj_dir + '/' + name + '.obj'
    ms.load_new_mesh(obj_path)
    face_num = len(Mesh(filename=obj_path, custom_read=True).f)
    if face_num > 2000:
        face_num = 2000
    elif face_num > 1000:
        face_num = (face_num - 1000) * (1/2) + 1500
        face_num = int(face_num)
    else:
        face_num = -1

    ms.remove_isolated_pieces_wrt_face_num(mincomponentsize=25, removeunref=True)
    if face_num > 0:
        ms.simplification_quadric_edge_collapse_decimation(targetfacenum=face_num, autoclean=True)
    m = ms.current_mesh()
    vs = m.vertex_matrix()
    ff = m.face_matrix()

    Mesh(v=vs, f=ff).write_obj(obj_remesh_dir + '/' + name + '.obj')
    # ms.save_current_mesh(obj_remesh_dir + '/' + name + '.obj', save_vertex_normal=False)