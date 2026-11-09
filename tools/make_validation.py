"""Build validation.txt (src_char,dst_char,motion_idx) from the test characters/motions that exist on disk.
usage: python tools/make_validation.py <Mixamo_simplify_new> [n_rows=10] [seed=2023]"""
import os, sys, random, itertools
d = sys.argv[1]
n_rows = int(sys.argv[2]) if len(sys.argv) > 2 else 10
random.seed(int(sys.argv[3]) if len(sys.argv) > 3 else 2023)
chars = [l.strip() for l in open(f'{d}/mo60_character_test.txt') if l.strip()]
chars = [c for c in chars if os.path.exists(f'{d}/npz/{c}.npz')]
motions = [m for m in (l.strip() for l in open(f'{d}/motion60_test.txt')) if m and os.path.exists(f'{d}/motion60/{m}')]
pairs = list(itertools.permutations(chars, 2))
assert pairs and motions, (chars, len(motions))
rows = [(*pairs[i % len(pairs)], random.randrange(len(motions))) for i in range(n_rows)]
open(f'{d}/validation.txt', 'w').write('\n'.join(f'{a},{b},{m}' for a, b, m in rows) + '\n')
print('test chars:', chars, '| test motions:', len(motions), '| rows:', rows)
