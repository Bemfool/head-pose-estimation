# A C++ Impl for Basel Face Model

A simple implementation to use Basel Face Model (BFM) in C++. 

Depend on:

* Basel Face Model (version>=2017, because we need expression information) 

* HDF5;
* Dlib;

Input format:

```
/path/to/basel_face_model.h5
n_vertice n_face n_id_pc n_expr_pc
n_landmark /path/to/landmark.anl
fx fy cx cy
/h5_path/to/shape_mu /h5_path/to/shape_ev /h5_path/to/shape_pc
/h5_path/to/tex_mu /h5_path/to/tex_ev /h5_path/to/tex_pc
/h5_path/to/expr_mu /h5_path/to/expr_ev /h5_path/to/expr_pc
/h5_path/to_triangle_lists
```

if landmark is not needed, use `-1 -1` to replace `n_landmark /path/to/landmark.anl`  .

If needed, `landmark.anl` format is as follow:

```
landmark_idx[0]
landmark_idx[1]
...
landmark_idx[n_landmark-1]
```

