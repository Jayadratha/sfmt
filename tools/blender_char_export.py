"""Export a skinned Mixamo character (.fbx, T-pose) to the inputs of data_proc/mixamo_*preproc*.py:
  <out_dir>/Mixamo_obj_Char/<name>.obj        merged mesh, rest pose, metres, Y-up
  <out_dir>/Mixamo_skinning_GT/<name>.txt     RigNet-style rig: joints (cm), root, skin, hier

Run inside Blender:
  blender -b --python tools/blender_char_export.py -- <out_dir> <name>=<file.fbx> [<name>=<file.fbx> ...]
"""
import bpy, sys, os, traceback
import numpy as np

argv = sys.argv[sys.argv.index("--") + 1:]
out_dir = argv[0]
jobs = [a.split("=", 1) for a in argv[1:]]
obj_dir = os.path.join(out_dir, "Mixamo_obj_Char")
rig_dir = os.path.join(out_dir, "Mixamo_skinning_GT")
os.makedirs(obj_dir, exist_ok=True)
os.makedirs(rig_dir, exist_ok=True)


def yup(p):                      # Blender Z-up -> Y-up (same mapping the BVH exporter uses)
    return (p[0], p[2], -p[1])


def export(name, fbx):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=fbx, automatic_bone_orientation=False, use_anim=False)
    arm = [o for o in bpy.data.objects if o.type == 'ARMATURE'][0]
    meshes = [o for o in bpy.data.objects if o.type == 'MESH']
    arm.data.pose_position = 'REST'
    for m in meshes:                                  # make sure no pose / shape-key offsets leak in
        for sk in (m.data.shape_keys.key_blocks if m.data.shape_keys else []):
            sk.value = 0.0
    bpy.ops.object.select_all(action='DESELECT')
    for m in meshes:
        m.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    if len(meshes) > 1:
        bpy.ops.object.join()
    mesh = bpy.context.view_layer.objects.active
    bpy.context.view_layer.update()

    bone_names = [b.name for b in arm.data.bones]
    bone_set = set(bone_names)
    gname = {vg.index: vg.name for vg in mesh.vertex_groups}

    # geometry (evaluated, rest pose) in world space
    dg = bpy.context.evaluated_depsgraph_get()
    ev = mesh.evaluated_get(dg)
    me = ev.to_mesh()
    mw = np.array(mesh.matrix_world)
    co = np.array([v.co[:] for v in me.vertices], dtype=np.float64)
    co = (mw[:3, :3] @ co.T).T + mw[:3, 3]
    co = np.stack([co[:, 0], co[:, 2], -co[:, 1]], 1)
    me.calc_loop_triangles()
    tris = np.array([t.vertices[:] for t in me.loop_triangles], dtype=np.int64)
    # skin weights straight from the (unevaluated) vertex groups, same vertex order as `me`
    weights = []
    for v in mesh.data.vertices:
        ws = [(gname[g.group], g.weight) for g in v.groups if gname.get(g.group) in bone_set and g.weight > 1e-4]
        s = sum(w for _, w in ws)
        weights.append([(n, w / s) for n, w in ws] if s > 0 else [])
    assert len(weights) == len(co), (len(weights), len(co))
    n_unskinned = sum(1 for w in weights if not w)
    ev.to_mesh_clear()

    with open(os.path.join(obj_dir, name + ".obj"), "w") as f:
        for p in co:
            f.write("v %.6f %.6f %.6f\n" % tuple(p))
        for t in tris:
            f.write("f %d %d %d\n" % (t[0] + 1, t[1] + 1, t[2] + 1))

    # rig, centimetres (the preproc scales joints by 0.01)
    mwa = np.array(arm.matrix_world)
    lines = []
    root = None
    for b in arm.data.bones:
        h = mwa[:3, :3] @ np.array(b.head_local[:]) + mwa[:3, 3]
        x, y, z = yup(h * 100.0)
        lines.append("joints %s %.8f %.8f %.8f" % (b.name, x, y, z))
        if b.parent is None:
            root = root or b.name
    lines.append("root %s" % root)
    for i, w in enumerate(weights):
        if w:
            lines.append("skin %d %s" % (i, " ".join("%s %.6f" % (n, x) for n, x in w)))
    for b in arm.data.bones:
        if b.parent is not None:
            lines.append("hier %s %s" % (b.parent.name, b.name))
    with open(os.path.join(rig_dir, name + ".txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("CHAR_OK", name, "verts", len(co), "tris", len(tris), "bones", len(bone_names), "unskinned_verts", n_unskinned,
          "y_range_m %.3f..%.3f" % (co[:, 1].min(), co[:, 1].max()))


for name, fbx in jobs:
    try:
        export(name, fbx)
    except Exception:
        traceback.print_exc()
        print("CHAR_FAIL", name)
