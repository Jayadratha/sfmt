import os

# Paths are configured through environment variables (see README.md / .env.example).
# Defaults assume everything lives under ./data and results go to ./runs.
_ROOT = os.path.dirname(os.path.abspath(__file__))
_DATA = os.environ.get("SFMT_DATA", os.path.join(_ROOT, "data"))

AMASS_PATH = os.environ.get("SFMT_AMASS_PATH", os.path.join(_DATA, "amass", "processed"))
SMPLH_PATH = os.environ.get("SFMT_SMPLH_PATH", os.path.join(_DATA, "smplh"))
RIGNET_PATH = os.environ.get("SFMT_RIGNET_PATH", os.path.join(_DATA, "Rignet"))
CUSTOM_PATH = os.environ.get("SFMT_CUSTOM_PATH", os.path.join(_ROOT, "custom"))
MIXAMO_PATH = os.environ.get("SFMT_MIXAMO_PATH", os.path.join(_DATA, "Mixamo"))
MIXAMO_SIMPLIFY_PATH = os.environ.get("SFMT_MIXAMO_SIMPLIFY_PATH", os.path.join(MIXAMO_PATH, "Mixamo_simplify_new"))
BLEND_FILE = './scene.blend'
BLENDER_PATH = os.environ.get("SFMT_BLENDER_PATH", "blender")
TEMP_DIR = os.path.join(_ROOT, 'temp')
LOG_DIR = os.environ.get("SFMT_LOG_DIR", os.path.join(_ROOT, "runs"))

RANDOM_SEED = 2023

os.makedirs(TEMP_DIR, exist_ok=True)

gdr2num = {'male': -1, 'neutral': 0, 'female': 1}
MIXAMO_JOINTS = [ "Hips",   # 0
  "LeftFoot",               # 1
  "Spine1",                 # 2
  "Spine2",                 # 3
  "RightLeg",               # 4
  "RightUpLeg",             # 5
  "RightForeArm",           # 6
  "LeftUpLeg",              # 7
  "RightToeBase",           # 8
  "LeftShoulder",           # 9
  "RightHand",              # 10
  "LeftForeArm",            # 11
  "LeftArm",                # 12
  "RightArm",               # 13
  "RightFoot",              # 14
  "Head",                   # 15
  "LeftHand",               # 16
  "RightShoulder",          # 17
  "LeftLeg",                # 18
  "Spine",                  # 19
  "LeftToeBase",            # 20
  "Neck"]                   # 21
