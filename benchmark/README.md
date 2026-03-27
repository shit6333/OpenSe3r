# AMB3R Benchmark

This is the benchmark that supplies AMB3R paper on various 3D reconstruction tasks.

```bibtex
@article{wang2025amb3r,
  title={AMB3R: Accurate Feed-forward Metric-scale 3D Reconstruction with Backend},
  author={Wang, Hengyi and Agapito, Lourdes},
  journal={arXiv preprint arXiv:2511.20343},
  year={2025}
}
```

## Table of Contents
- [Monocular Depth Estimation](#monocular-depth-estimation)
- [Camera Pose Estimation](#camera-pose-estimation)
- [Multi-view Depth Estimation](#multi-view-depth-estimation)
- [Video Depth Estimation](#video-depth-estimation)
- [Multi-view 3D Reconstruction](#multi-view-3d-reconstruction)
- [Visual Odometry/SLAM](#visual-odometryslam)
- [Structure from Motion](#structure-from-motion)
- [Evaluating Custom Models](#evaluating-custom-models)

## Monocular Depth Estimation

<table>
  <thead>
    <tr>
      <th rowspan="2" align="left"><strong>Method</strong></th>
      <th colspan="2" align="center"><strong>NYUv2</strong></th>
      <th colspan="2" align="center"><strong>KITTI</strong></th>
      <th colspan="2" align="center"><strong>ETH3D</strong></th>
      <th colspan="2" align="center"><strong>ScanNet</strong></th>
      <th colspan="2" align="center"><strong>DIODE</strong></th>
    </tr>
    <tr>
      <th align="center">Rel &darr;</th>
      <th align="center">&delta;<sub>1.25</sub> &uarr;</th>
      <th align="center">Rel &darr;</th>
      <th align="center">&delta;<sub>1.25</sub> &uarr;</th>
      <th align="center">Rel &darr;</th>
      <th align="center">&delta;<sub>1.25</sub> &uarr;</th>
      <th align="center">Rel &darr;</th>
      <th align="center">&delta;<sub>1.25</sub> &uarr;</th>
      <th align="center">Rel &darr;</th>
      <th align="center">&delta;<sub>1.25</sub> &uarr;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="left">VGGT</td>
      <td align="center">3.6</td>
      <td align="center">98.0</td>
      <td align="center">8.8</td>
      <td align="center">92.7</td>
      <td align="center">3.8</td>
      <td align="center">97.9</td>
      <td align="center">2.7</td>
      <td align="center">98.8</td>
      <td align="center">26.9</td>
      <td align="center">79.1</td>
    </tr>
    <tr>
      <td align="left">DA3 (NESTEDGIANT)</td>
      <td align="center">4.4</td>
      <td align="center">96.9</td>
      <td align="center">7.9</td>
      <td align="center">94.8</td>
      <td align="center">4.6</td>
      <td align="center">97.1</td>
      <td align="center">4.2</td>
      <td align="center">97.4</td>
      <td align="center">30.3</td>
      <td align="center">78.7</td>
    </tr>
    <tr>
      <td align="left"><strong>AMB3R </strong></td>
      <td align="center"><strong>3.0</strong></td>
      <td align="center"><strong>98.9</strong></td>
      <td align="center"><strong>7.3</strong></td>
      <td align="center"><strong>95.4</strong></td>
      <td align="center"><strong>3.2</strong></td>
      <td align="center"><strong>98.8</strong></td>
      <td align="center"><strong>2.7</strong></td>
      <td align="center"><strong>98.9</strong></td>
      <td align="center"><strong>24.7</strong></td>
      <td align="center"><strong>80.7</strong></td>
    </tr>
  </tbody>
</table>


### Data Preparation
We follow Marigold and Diffusion-E2E and evaluate on the following datasets: **NYUv2**, **KITTI**, **ETH3D**, **ScanNet**, and **DIODE**. You can download the datasets by running the following script:

```sh
cd ../scripts
bash download_mono.sh
cd ../benchmarks
```

### Evaluation
```bash
python eval_monodepth.py
```



## Camera Pose Estimation

<table>
  <thead>
    <tr>
      <th align="left"><strong>Dataset</strong></th>
      <th align="center"><strong>VGGT</strong></th>
      <th align="center">DA3 (NESTEDGIANT)</th>
      <th align="center"><strong>AMB3R</strong></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="left">mAA_30</td>
      <td align="center">81.8</td>
      <td align="center"><strong>87.5</strong></td>
      <td align="center">86.3</td>
    </tr>
  </tbody>
</table>

Following VGGT, we evaluate on the RealEstate10K dataset. Due to the limited availability of some YouTube videos in RealEstate10K, we provide a pre-processed evaluation split: [re10k_amb3r_split](https://huggingface.co/datasets/HengyiWang/re10k_amb3r_split). This split is generated using the same random frame sampling strategy described in VGGT, enabling the community to reproduce the evaluation on RealEstate10K.

> **Note:** The released split contains 1,721 sequences, each consisting of 10 frames randomly sampled from the corresponding full video. As this split is not exactly identical to the one originally used in VGGT, we kindly ask that you explicitly state the use of the `re10k_amb3r_split` in your paper to ensure fair comparison and reproducibility.

```sh
cd ../scripts
bash download_pose.sh
cd ../benchmarks
```

### Evaluation
```bash
python eval_pose.py
```


## Multi-view Depth Estimation

<table>
  <thead>
    <tr>
      <th rowspan="2" align="left"><strong>Method</strong></th>
      <th colspan="2" align="center"><strong>KITTI</strong></th>
      <th colspan="2" align="center"><strong>ScanNet</strong></th>
      <th colspan="2" align="center"><strong>ETH3D</strong></th>
      <th colspan="2" align="center"><strong>DTU</strong></th>
      <th colspan="2" align="center"><strong>T&amp;T</strong></th>
      <th colspan="2" align="center"><strong>Avg.</strong></th>
    </tr>
    <tr>
      <th align="center">Rel &darr;</th>
      <th align="center">&delta;<sub>1.03</sub> &uarr;</th>
      <th align="center">Rel &darr;</th>
      <th align="center">&delta;<sub>1.03</sub> &uarr;</th>
      <th align="center">Rel &darr;</th>
      <th align="center">&delta;<sub>1.03</sub> &uarr;</th>
      <th align="center">Rel &darr;</th>
      <th align="center">&delta;<sub>1.03</sub> &uarr;</th>
      <th align="center">Rel &darr;</th>
      <th align="center">&delta;<sub>1.03</sub> &uarr;</th>
      <th align="center">Rel &darr;</th>
      <th align="center">&delta;<sub>1.03</sub> &uarr;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="left">VGGT</td>
      <td align="center">4.5</td>
      <td align="center">59.6</td>
      <td align="center">2.3</td>
      <td align="center">80.8</td>
      <td align="center">1.8</td>
      <td align="center">86.3</td>
      <td align="center"><strong>0.9</strong></td>
      <td align="center"><strong>95.6</strong></td>
      <td align="center">2.4</td>
      <td align="center">84.1</td>
      <td align="center">2.4</td>
      <td align="center">81.3</td>
    </tr>
    <tr>
      <td align="left">DA3 (NESTEDGIANT)</td>
      <td align="center">3.9</td>
      <td align="center">59.1</td>
      <td align="center">2.7</td>
      <td align="center">76.7</td>
      <td align="center">2.2</td>
      <td align="center">88.7</td>
      <td align="center">1.5</td>
      <td align="center">92.7</td>
      <td align="center">2.5</td>
      <td align="center">88.0</td>
      <td align="center">2.6</td>
      <td align="center">81.0</td>
    </tr>
    <tr>
      <td align="left"><strong>AMB3R </strong></td>
      <td align="center"><strong>2.8</strong></td>
      <td align="center"><strong>74.4</strong></td>
      <td align="center"><strong>1.9</strong></td>
      <td align="center"><strong>85.8</strong></td>
      <td align="center"><strong>1.4</strong></td>
      <td align="center"><strong>90.9</strong></td>
      <td align="center"><strong>0.9</strong></td>
      <td align="center">95.1</td>
      <td align="center"><strong>1.7</strong></td>
      <td align="center"><strong>90.2</strong></td>
      <td align="center"><strong>1.7</strong></td>
      <td align="center"><strong>87.3</strong></td>
    </tr>
  </tbody>
</table>

### Data Preparation
Following RMVDB, we evaluate using the **Tanks and Temples**, **ETH3D**, **ScanNet**, **DTU**, and **KITTI** datasets.

```sh
cd ../scripts
bash download_rmvd.sh
cd ../benchmarks
```

> **Note:** **ScanNet** and **KITTI** datasets must be downloaded manually.

### Evaluation
To evaluate metric-scale depth, append the `--metric` flag:
```bash
python eval_mvdepth.py
```

---

## Video Depth Estimation

### Data Preparation
We use the **Sintel**, **Bonn**, and **KITTI** datasets for evaluation.

```sh
cd ../scripts
bash download_videodepth.sh
cd ../benchmarks
```

### Evaluation
> **Note:** Evaluating on the Bonn and KITTI dataset would require a GPU with more than 24GB of memory.

```bash
python eval_videodepth.py
```

---

## Multi-view 3D Reconstruction

<table>
  <thead>
    <tr>
      <th rowspan="2" align="left"><strong>Method</strong></th>
      <th colspan="3" align="center"><strong>ETH3D</strong></th>
      <th colspan="3" align="center"><strong>DTU</strong></th>
      <th colspan="3" align="center"><strong>7-Scenes</strong></th>
    </tr>
    <tr>
      <th align="center">Rel &darr;</th>
      <th align="center">Acc &darr;</th>
      <th align="center">Cp &darr;</th>
      <th align="center">Rel &darr;</th>
      <th align="center">Acc &darr;</th>
      <th align="center">Cp &darr;</th>
      <th align="center">Rel &darr;</th>
      <th align="center">Acc &darr;</th>
      <th align="center">Cp &darr;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="left">VGGT</td>
      <td align="center">6.02</td>
      <td align="center">12.81</td>
      <td align="center">11.89</td>
      <td align="center">0.83</td>
      <td align="center"><strong>0.22</strong></td>
      <td align="center">0.08</td>
      <td align="center">5.51</td>
      <td align="center">2.32</td>
      <td align="center">3.51</td>
    </tr>
    <tr>
      <td align="left">DA3 (NESTEDGIANT)</td>
      <td align="center">9.24</td>
      <td align="center">21.22</td>
      <td align="center">11.30</td>
      <td align="center">2.06</td>
      <td align="center">0.82</td>
      <td align="center">0.23</td>
      <td align="center">5.26</td>
      <td align="center">2.11</td>
      <td align="center">2.56</td>
    </tr>
    <tr>
      <td align="left"><strong>AMB3R </strong></td>
      <td align="center"><strong>4.64</strong></td>
      <td align="center"><strong>9.98</strong></td>
      <td align="center"><strong>9.69</strong></td>
      <td align="center"><strong>0.81</strong></td>
      <td align="center"><strong>0.22</strong></td>
      <td align="center"><strong>0.08</strong></td>
      <td align="center"><strong>4.70</strong></td>
      <td align="center"><strong>1.75</strong></td>
      <td align="center"><strong>2.22</strong></td>
    </tr>
  </tbody>
</table>

### Data Preparation
We evaluate on the **ETH3D**, **DTU**, and **7Scenes** datasets.
- For ETH3D and DTU, image tuples are from RMVDB.
- For 7Scenes, image tuples are from Spann3R.

```sh
cd ../scripts
bash download_3d.sh
cd ../benchmarks
```

> **Note:** As the original 7Scenes dataset contains imperfect poses for evaluation, we provide pre-processed poses [here](./data/7scenes_sfm_poses.zip). These are COLMAP poses provided by [visloc_pseudo_gt](https://github.com/tsattler/visloc_pseudo_gt_limitations). The preparation script will automatically extract and place these processed poses in the correct directory.

### Evaluation
```bash
python eval_mvrecon.py
```

---

## Visual Odometry / SLAM

<table>
  <thead>
    <tr>
      <th align="left"><strong>Method</strong></th>
      <th align="center"><strong>Calib.</strong></th>
      <th align="center"><strong>Opt.</strong></th>
      <th align="center"><strong>360</strong></th>
      <th align="center"><strong>desk</strong></th>
      <th align="center"><strong>desk2</strong></th>
      <th align="center"><strong>floor</strong></th>
      <th align="center"><strong>plant</strong></th>
      <th align="center"><strong>room</strong></th>
      <th align="center"><strong>rpy</strong></th>
      <th align="center"><strong>teddy</strong></th>
      <th align="center"><strong>xyz</strong></th>
      <th align="center"><strong>avg</strong></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="left">MASt3R-SLAM</td>
      <td align="center">&#10003;</td>
      <td align="center">&#10003;</td>
      <td align="center">4.9</td>
      <td align="center">1.6</td>
      <td align="center">2.4</td>
      <td align="center"><strong>2.5</strong></td>
      <td align="center"><strong>2.0</strong></td>
      <td align="center">6.1</td>
      <td align="center">2.7</td>
      <td align="center">4.1</td>
      <td align="center">0.9</td>
      <td align="center">3.0</td>
    </tr>
    <tr>
      <td align="left">MASt3R-SLAM</td>
      <td align="center">&#10007;</td>
      <td align="center">&#10003;</td>
      <td align="center">7.0</td>
      <td align="center">3.5</td>
      <td align="center">5.5</td>
      <td align="center">5.6</td>
      <td align="center">3.5</td>
      <td align="center">11.8</td>
      <td align="center">4.1</td>
      <td align="center">11.4</td>
      <td align="center">2.0</td>
      <td align="center">6.0</td>
    </tr>
    <tr>
      <td align="left">VGGT-SLAM</td>
      <td align="center">&#10007;</td>
      <td align="center">&#10003;</td>
      <td align="center">7.1</td>
      <td align="center">2.5</td>
      <td align="center">4.0</td>
      <td align="center">14.1</td>
      <td align="center">2.3</td>
      <td align="center">10.2</td>
      <td align="center">3.0</td>
      <td align="center">3.4</td>
      <td align="center">1.4</td>
      <td align="center">5.3</td>
    </tr>
    <tr>
      <td align="left">KV-Tracker</td>
      <td align="center">&#10007;</td>
      <td align="center">&#10007;</td>
      <td align="center">16.6</td>
      <td align="center">6.0</td>
      <td align="center">8.3</td>
      <td align="center">-</td>
      <td align="center">4.8</td>
      <td align="center">36.6</td>
      <td align="center">4.5</td>
      <td align="center">7.1</td>
      <td align="center">2.1</td>
      <td align="center">-</td>
    </tr>
    <tr>
      <td align="left"><strong>AMB3R-VO</strong></td>
      <td align="center">&#10007;</td>
      <td align="center">&#10007;</td>
      <td align="center"><strong>3.9</strong></td>
      <td align="center">1.7</td>
      <td align="center">2.3</td>
      <td align="center">2.7</td>
      <td align="center">2.7</td>
      <td align="center">5.5</td>
      <td align="center">2.2</td>
      <td align="center"><strong>2.8</strong></td>
      <td align="center">0.8</td>
      <td align="center">2.7</td>
    </tr>
    <tr>
      <td align="left"><strong>AMB3R-VO (DA3)</strong></td>
      <td align="center">&#10007;</td>
      <td align="center">&#10007;</td>
      <td align="center">4.0</td>
      <td align="center"><strong>1.3</strong></td>
      <td align="center"><strong>1.8</strong></td>
      <td align="center">3.2</td>
      <td align="center">3.3</td>
      <td align="center"><strong>3.9</strong></td>
      <td align="center"><strong>1.8</strong></td>
      <td align="center">3.7</td>
      <td align="center"><strong>0.7</strong></td>
      <td align="center"><strong>2.6</strong></td>
    </tr>
  </tbody>
</table>

### Data Preparation
We use the **TUM**, **ETH3D SLAM**, and **7Scenes** datasets for SLAM evaluation.

```sh
cd ../scripts
bash download_slam.sh
cd ../benchmarks
```

### Evaluation
```bash
python eval_slam.py
```

---

## Structure from Motion (SfM)

### Data Preparation
We evaluate on the **ETH3D**, **Tanks and Temples (TnT)**, and **IMC Phototourism** datasets.
ETH3D is downloaded during the Multi-view Depth Estimation setup. Use the following script to download TnT and IMC:

```sh
cd ../scripts
bash download_sfm.sh
cd ../benchmarks
```

### Evaluation
```bash
python eval_sfm.py
```

---

## Evaluating Custom Models

You can easily plug in and evaluate your own models with AMB3R-Benchmark.

For **Monocular Depth Estimation**, **Camera Pose Estimation**, **Multi-view 3D Reconstruction**, and **Video Depth Estimation**, implement a `run_amb3r_benchmark` method in your model class:

```python
def run_amb3r_benchmark(self, frames):
    # frames['images']: (B, T, C, H, W) normalized in [-1, 1]
    images = frames['images'] 
    
    # Run your own base model to get pointmap, pose, and confidence
    pointmap, pose, confidence, pts3d_by_unprojection = self.forward(images)

    return {
        'world_points': pointmap,            # Pointmaps
        'depth': depth_map,                  # Depth predictions
        'pose': pose,                        # Camera poses
        'pts3d_by_unprojection': pts3d       # Unprojected 3D points
    }
```

For **Multi-view Depth Estimation**, you'll need to write custom input and output adapter functions, following the structure documented [here](https://github.com/lmb-freiburg/robustmvd/blob/master/rmvd/models/README.md).


For **Visual Odometry** and **Structure-from-Motion**, please implement the following methods in your model:
- `run_amb3r_vo`
- `run_amb3r_sfm`


## Citation

If you find our code, data, or paper useful, please consider citing:

```bibtex
@article{wang2025amb3r,
  title={AMB3R: Accurate Feed-forward Metric-scale 3D Reconstruction with Backend},
  author={Wang, Hengyi and Agapito, Lourdes},
  journal={arXiv preprint arXiv:2511.20343},
  year={2025}
}
```

