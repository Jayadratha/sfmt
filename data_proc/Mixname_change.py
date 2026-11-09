import os, sys
sys.path.append(sys.path[0]+"/../")


from global_var import MIXAMO_SIMPLIFY_PATH
mix_simp_path = MIXAMO_SIMPLIFY_PATH

file_path = os.path.join(mix_simp_path, "MixNameChange.txt")
filee = open(file_path,"r")


lines = filee.readlines()
ori_names = []
change_names = []
for ii in lines:
    ii = ii.strip('\n').split("|")
    ori_names.append(ii[0])
    change_names.append(ii[1])

for idx in range(len(ori_names)):
    ori_file_name = os.path.join(mix_simp_path, "npz", ori_names[idx])
    ren_file_name = os.path.join(mix_simp_path, "npz", change_names[idx])
    os.rename(ori_file_name+".npz", ren_file_name+".npz")
