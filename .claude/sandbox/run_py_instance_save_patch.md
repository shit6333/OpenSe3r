# run.py — 需要新增的 npz 儲存段落

## 位置 1：Instance 匯出區塊（第 252 行之後）

在 `print(f"Saved instance per-pt PLY → {pp_ins_ply}")` 後面加入：

```python
        # (C) 儲存 voxel feature npz（供 cluster_instances.py 使用）
        ins_centers, ins_features = memory.instance_voxel_map.get_all()
        ins_centers_np  = ins_centers.numpy().astype(np.float32)
        ins_features_np = ins_features.numpy().astype(np.float32)
        if T_voxel is not None:
            ins_centers_np = transform_points(ins_centers_np, T_voxel)
        ins_feats_npz = os.path.join(
            args.results_path, f"scene_{args.demo_name}_instance_voxel_feats.npz"
        )
        np.savez_compressed(ins_feats_npz,
            voxel_centers  = ins_centers_np,
            voxel_features = ins_features_np,
        )
        print(f"Saved instance feats npz  → {ins_feats_npz}")
```

## 位置 2：Semantic 匯出區塊（第 229 行 legend print 之後）

在 `print(f"Saved semantic legend → {legend_path}")` 後面加入：

```python
        # (C) 儲存 voxel feature npz
        sem_centers, sem_features = memory.semantic_voxel_map.get_all()
        sem_centers_np  = sem_centers.numpy().astype(np.float32)
        sem_features_np = sem_features.numpy().astype(np.float32)
        if T_voxel is not None:
            sem_centers_np = transform_points(sem_centers_np, T_voxel)
        sem_feats_npz = os.path.join(
            args.results_path, f"scene_{args.demo_name}_semantic_voxel_feats.npz"
        )
        np.savez_compressed(sem_feats_npz,
            voxel_centers  = sem_centers_np,
            voxel_features = sem_features_np,
        )
        print(f"Saved semantic feats npz  → {sem_feats_npz}")
```

## 輸出的檔案格式

| 檔名 | keys | shape |
|------|------|-------|
| `scene_X_instance_voxel_feats.npz` | `voxel_centers`, `voxel_features` | `(V,3)`, `(V,16)` |
| `scene_X_semantic_voxel_feats.npz` | `voxel_centers`, `voxel_features` | `(V,3)`, `(V,512)` |

`results.npz` 不變，供 cluster_instances.py 重建 per-point 幾何點使用。
