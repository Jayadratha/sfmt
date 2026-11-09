"""Convert Mixamo animation .fbx files to .bvh (Y-up, XYZ Euler, zero rotation = T-pose).

Run inside Blender (one session converts a whole folder):
  blender -b --python tools/blender_fbx_to_bvh.py -- <fbx_dir> <out_dir> [list.txt]
`list.txt` (optional) holds one file stem per line (e.g. "Quad Punch"); default: every *.fbx in <fbx_dir>.
"""
import bpy, sys, os, traceback

argv = sys.argv[sys.argv.index("--") + 1:]
fbx_dir, out_dir = argv[0], argv[1]
stems = None
if len(argv) > 2:
    stems = [l.strip() for l in open(argv[2]) if l.strip()]
    stems = [s[:-4] if s.endswith(('.npy', '.fbx')) else s for s in stems]
else:
    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(fbx_dir) if f.lower().endswith('.fbx'))
os.makedirs(out_dir, exist_ok=True)


def convert(stem):
    out = os.path.join(out_dir, stem + '.bvh')
    if os.path.exists(out):
        return 'skip'
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=os.path.join(fbx_dir, stem + '.fbx'), automatic_bone_orientation=False, use_anim=True)
    arms = [o for o in bpy.data.objects if o.type == 'ARMATURE']
    if not arms or not arms[0].animation_data or not arms[0].animation_data.action:
        return 'no-armature-or-action'
    arm = arms[0]
    bpy.ops.object.select_all(action='DESELECT')
    arm.select_set(True)
    bpy.context.view_layer.objects.active = arm
    f0, f1 = arm.animation_data.action.frame_range
    bpy.ops.export_anim.bvh(filepath=out, frame_start=int(round(f0)), frame_end=int(round(f1)),
                            rotate_mode='XYZ', root_transform_only=True, global_scale=1.0)
    return 'ok'


stats = {}
for i, stem in enumerate(stems):
    try:
        r = convert(stem)
    except Exception:
        traceback.print_exc()
        r = 'error'
    stats[r] = stats.get(r, 0) + 1
    if r in ('error', 'no-armature-or-action'):
        print('CONVERT_FAIL', stem, r)
    if i % 50 == 0:
        print('CONVERT_PROGRESS', i, len(stems), stats, flush=True)
print('CONVERT_DONE', stats)
