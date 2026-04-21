import numpy as np, plyfile, trimesh
p = '/mnt/HDD4/ricky/feedforward/amb3r/outputs/semantic_slamv2/exp_stage2v4_mode3/scene0050_00/scene_scene0050_00_semantic_voxels.ply'
d = plyfile.PlyData.read(p)
v = d['vertex']
xyz = np.stack([v['x'], v['y'], v['z']], axis=-1)
rgb = np.stack([v['red'], v['green'], v['blue']], axis=-1)
lbl = np.array(v['label'])

print('N verts         :', len(xyz))
print('Unique labels   :', np.unique(lbl, return_counts=True))
print('Unique RGB rows :', len(np.unique(rgb.reshape(-1,3), axis=0)))

# Spread check — 隨機抽一些點看分佈是否正常
idx = np.random.choice(len(xyz), 10, replace=False)
print('Sample xyz:\n', xyz[idx])
print('Sample rgb:\n', rgb[idx])

# 跟 per_point 的 xyz 範圍對比
p2 = p.replace('_voxels.ply', '_per_point.ply')
d2 = plyfile.PlyData.read(p2)
v2 = d2['vertex']
xyz2 = np.stack([v2['x'], v2['y'], v2['z']], axis=-1)
print('\nper_point xyz range:',
      xyz2.min(axis=0), '→', xyz2.max(axis=0))
print('voxels    xyz range:',
      xyz.min(axis=0), '→', xyz.max(axis=0))