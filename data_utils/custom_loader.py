import os, sys
sys.path.append(sys.path[0]+"/../")
import numpy as np
import torch
from torch_geometric.data import Dataset, Data
from torch_geometric.utils import add_self_loops
from utils.lbs import lbs
from utils.geometry import get_tpl_edges, fps_np, get_normal
from utils.o3d_wrapper import Mesh, MeshO3d
from global_var import *
import cv2
from scipy.spatial.transform import Rotation
from utils.bvh import Bvh
from utils.general import np2torch
from data_utils.common import get_geo_index
from scipy.sparse import csr_matrix



def merge_skinning_weight(weight, joint_names, parents):
    children = [[] for _ in range(len(parents))]
    for i, p in enumerate(parents):
        if p >= 0:
            children[p].append(i)

    def add_children(idx, joint_idxs):
        joint_idxs.append(idx)
        if len(children[idx]) == 0:
            return
        for c in children[idx]:
            add_children(c, joint_idxs)
    ee_joint_idxs = [[] for k in range(5)]
    ee_names = ["Head", "LeftHand", "RightHand", "LeftToeBase", "RightToeBase"]
    ee_ext_names = []
    for i, ee_name in enumerate(ee_names):
        joint_idx = joint_names.index(ee_name)
        add_children(joint_idx, ee_joint_idxs[i])
        for idx in ee_joint_idxs[i]:
            ee_ext_names.append(joint_names[idx])

    n_vertices = weight.shape[0]
    new_weight = np.zeros([n_vertices, len(MIXAMO_JOINTS)])
    for i, name in enumerate(MIXAMO_JOINTS):
        if name not in ee_ext_names:
            new_weight[:, i] = weight[:, joint_names.index(name)]
        else:
            if name in ee_names:
                new_weight[:, i] = np.sum(weight[:, ee_joint_idxs[ee_names.index(name)]], 1)
    return new_weight



def normalize_y_rotation(raw_theta):
    """
    rotate along y axis so that root rotation can always face the camera
    theta should be a [3] or [66] numpy array
    """
    only_global = True
    if raw_theta.shape == (66,):
        theta = raw_theta[:3]
        only_global = False
    else:
        theta = raw_theta[:]
    raw_rot = cv2.Rodrigues(theta)[0]
    rot_z = raw_rot[:, 2]
    # we should rotate along y axis counter-clockwise for t rads to make the object face the camera
    if rot_z[2] == 0:
        t = (rot_z[0] / np.abs(rot_z[0])) * np.pi / 2
    elif rot_z[2] > 0:
        t = np.arctan(rot_z[0]/rot_z[2])
    else:
        t = np.arctan(rot_z[0]/rot_z[2]) + np.pi
    cost, sint = np.cos(t), np.sin(t)
    norm_rot = np.array([[cost, 0, -sint],[0, 1, 0],[sint, 0, cost]])
    final_rot = np.matmul(norm_rot, raw_rot)
    final_theta = cv2.Rodrigues(final_rot)[0][:, 0]
    if not only_global:
        return np.concatenate([final_theta, raw_theta[3:]], 0)
    else:
        return final_theta
    

def extend_pose(joint_names, poses):
    poses = poses.reshape([-1, 3])
    num_joints = len(joint_names)
    ext_poses = np.zeros([num_joints, 3], dtype=np.float32)
    for i, name in enumerate(MIXAMO_JOINTS):
        assert name in joint_names, "{} not in {}".format(name, joint_names)
        ext_poses[joint_names.index(name)] = poses[i]
    return ext_poses.reshape([-1])

class CustomDataset(Dataset):
    def __init__(self, data_dir, flag=None, preload=True):
        super(CustomDataset, self).__init__()
        self.data_dir = data_dir
        self.preload = preload
        if isinstance(flag, list) or isinstance(flag, tuple):
            self.names = flag
        elif flag is None:
            self.names = [k.replace('.npz', '') for k in os.listdir(data_dir) if k.endswith('.npz')]
        else:
            with open(os.path.join(self.data_dir, flag+'.txt')) as f:
                self.names = f.read().splitlines()

        self.vs, self.fs = [], []
        self.joints, self.parents, self.weights = [], [], []
        self.joint_names = []
        self.tpl_edge_indexs, self.geo_edge_indexs = [], []
        if self.preload:
            self._preload()
        else:
            raise ValueError

        print('Number of subjects:', len(self))

    def get(self, index):
        v0_1, f1, joints0_1, joint_names1, parents1, weights1, tpl_edge_index1, \
            geo_edge_index1, name1 = self.load(index)
        tpl_edge_index1 = torch.from_numpy(tpl_edge_index1).long()
        tpl_edge_index1, _ = add_self_loops(tpl_edge_index1, num_nodes=v0_1.shape[0])

        center = (np.max(v0_1, 0, keepdims=True) + np.min(v0_1, 0, keepdims=True)) / 2
        scale = np.max(v0_1[:, 1], 0) - np.min(v0_1[:, 1], 0)
        v0 = (v0_1 - center) / scale

        v0 = torch.from_numpy(v0).float()
        normal_v0 = get_normal(v0, f1)
        return Data(v0=v0, tpl_edge_index=tpl_edge_index1, triangle=f1[None].astype(int),
                    feat0=normal_v0,
                    center=torch.from_numpy(center).float(), scale=torch.from_numpy(np.array(scale)).float(),
                    num_nodes=len(v0))

    def get_by_name(self, name):
        idx = self.names.index(name)
        return self.get(idx)

    def load(self, index):
        if self.preload:
            return self.vs[index], self.fs[index], self.joints[index], self.joint_names[index], \
                   self.parents[index], self.weights[index], self.tpl_edge_indexs[index], \
                   self.geo_edge_indexs[index], self.names[index]
        
    def len(self):
        return len(self.names)

    def _preload(self):
        path=self.data_dir
        for idx in self.names:
            c = np.load(os.path.join(path, idx+'.npz'))
            v = c["v"]
            center = (np.max(v, 0, keepdims=True) + np.min(v, 0, keepdims=True)) / 2
            scale = np.max(v[:, 1], 0) - np.min(v[:, 1], 0)
            v = (v - center) / scale
            joints = c["joints"]
            joints = (joints - center) / scale

            self.vs.append(v)
            self.fs.append(c["f"].astype(int))
            self.joints.append(joints)
            self.parents.append(c["parents"])
            self.joint_names.append(c["joint_names"].tolist())
            self.weights.append(csr_matrix((c["weights_data"], (c["weights_rows"], c["weights_cols"])),
                                           shape=c["weights_shape"], dtype=np.float32))
            self.tpl_edge_indexs.append(c["tpl_edge_index"].astype(int).T)
            self.geo_edge_indexs.append(np.stack([c["geo_rows"].astype(np.int32), c["geo_cols"].astype(np.int32)], 0))


class CustomMotionDataset(Dataset):
    def __init__(self, data_dir, flag=None, preload=True):
        super(CustomMotionDataset, self).__init__()
        self.data_dir = data_dir
        self.src_dir=os.path.join(data_dir,'src_data')
        self.preload = preload
        if isinstance(flag, list) or isinstance(flag, tuple):
            self.names = flag
        elif flag is None:
            self.names = [k.replace('.npz', '') for k in os.listdir(self.src_dir) if k.endswith('.npz') and k != "rest.obj"]
        else:
            with open(os.path.join(self.data_dir, flag+'.txt')) as f:
                self.names = f.read().splitlines()
        self.index1=0
        self.vs, self.fs = [], []
        self.joints, self.parents, self.weights = [], [], []
        self.joint_names = []
        self.tpl_edge_indexs, self.geo_edge_indexs = [], []
        self.poses = []
        if self.preload:
            self._preload()
            self._preload_motion()
        else:
            raise ValueError
        print(self.names)
        print("Number of characters:",len(self.vs))
        print('Number of poses:', len(self))

    def get(self, index):
        if(index%60==0):
            self.index1 = np.random.randint(0, 1000)%len(self.vs)
        v0_1, f1, joints0_1, joint_names1, parents1, weights1, tpl_edge_index1, \
            geo_edge_index1, name1 = self.load(self.index1)

        weights1 = np.array(weights1.todense()).astype(np.float32)
        
        aug_v0_1, aug_joints0_1 = v0_1, joints0_1

        theta1_ = self.poses[index]
        theta1_ = normalize_y_rotation(theta1_)
        theta1 = extend_pose(joint_names1, theta1_)

        tpl_edge_index1 = torch.from_numpy(tpl_edge_index1).long()
        tpl_edge_index1, _ = add_self_loops(tpl_edge_index1, num_nodes=v0_1.shape[0])
        geo_edge_index1 = torch.from_numpy(geo_edge_index1).long()


        v1, joints1, T1 = lbs(v0_1, theta1, joints0_1, parents1, weights1, verbose=True)    
        aug_v1, aug_joints1, aug_T1 = lbs(aug_v0_1, theta1, aug_joints0_1, parents1, weights1, verbose=True)
        
        v0_1,  v1,  T1 = np2torch(v0_1, v1,  T1)
        aug_v0_1, aug_v1,  aug_T1= np2torch(aug_v0_1, aug_v1,  aug_T1)

        normal_v0_1 = get_normal(v0_1, f1)
        normal_v1 = get_normal(v1, f1)
        
        if joint_names1[0] in MIXAMO_JOINTS:
            new_weights1 = merge_skinning_weight(weights1, joint_names1, parents1)
        else:
            # this is not real MIXAMO, just using the interface (e.g., result of NBS)
            new_weights1 = weights1

        # geo_edge_index=geo_edge_index1, geo_edge_index=geo_edge_index2
        # aug_T=aug_T1, aug_T=aug_T2
        return Data(v0=v0_1, v1=v1, tpl_edge_index=tpl_edge_index1, triangle=f1[None].astype(int), name=name1,
                    aug_v0=aug_v0_1, aug_v1=aug_v1, aug_T=aug_T1, aug_joints=aug_joints0_1[None],
                    feat0=normal_v0_1, feat1=normal_v1,
                    joints=joints0_1[None], weights=new_weights1, parents=parents1[None], theta=theta1[None],
                    num_nodes=len(v0_1), dataset=1)
               

    def get_by_name(self, name):
        idx = self.names.index(name)
        return self.get(idx)

    def load(self, index):
        if self.preload:
            return self.vs[index], self.fs[index], self.joints[index], self.joint_names[index], \
                   self.parents[index], self.weights[index], self.tpl_edge_indexs[index], \
                   self.geo_edge_indexs[index], self.names[index]
    def len(self):
        return len(self.poses)

    def _preload(self):
        # rest mesh
        path=os.path.join(self.data_dir,'src_data')
        for idx in self.names:
            c = np.load(os.path.join(path, idx+'.npz'))
            v = c["v"]
            center = (np.max(v, 0, keepdims=True) + np.min(v, 0, keepdims=True)) / 2
            scale = np.max(v[:, 1], 0) - np.min(v[:, 1], 0)
            v = (v - center) / scale
            joints = c["joints"]
            joints = (joints - center) / scale

            self.vs.append(v)
            self.fs.append(c["f"].astype(int))
            self.joints.append(joints)
            self.parents.append(c["parents"])
            self.joint_names.append(c["joint_names"].tolist())
            self.weights.append(csr_matrix((c["weights_data"], (c["weights_rows"], c["weights_cols"])),
                                           shape=c["weights_shape"], dtype=np.float32))
            self.tpl_edge_indexs.append(c["tpl_edge_index"].astype(int).T)
            self.geo_edge_indexs.append(np.stack([c["geo_rows"].astype(np.int32), c["geo_cols"].astype(np.int32)], 0))
    def _preload_motion(self):
        motion_dir = os.path.join(self.data_dir, 'motion_data')
        self.motion_names = [k for k in sorted(os.listdir(motion_dir)) if k.endswith('.npy')]
        self.motion_idxs = {}
        self.count = 0
        for name in self.motion_names:
            frames = np.load(os.path.join(motion_dir, name))
            self.motion_idxs[name] = (self.count, self.count+len(frames))
            self.count += len(frames)
            self.poses.append(frames)
        self.poses = np.concatenate(self.poses, 0).reshape([-1, 66])


