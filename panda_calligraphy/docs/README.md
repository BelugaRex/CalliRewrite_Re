# Expected simulation image

[`expected_simulation.png`](expected_simulation.png) is embedded in the [Panda viewer README](../README.md). It is a static visualization generated from this folder's implemented simulation components, **not a browser screenshot or a trained-policy result**.

![Generated Panda pose, actual canvas output and deformable brush contact](expected_simulation.png)

Launch the interactive scene with the existing local environment:

```bash
cd /data/CalliRewrite/panda_calligraphy
env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE=1 \
  ./venv/bin/python -u calligraphy_panda_viser_sim.py
```

Open the printed HTTP URL, usually `http://localhost:8080`. Select `yi` and press **Execute Character (Write)**. Stop the server with Ctrl+C. The script has no CLI flags; the [parent README](../README.md) explains the browser controls and environment setup.

The figure contains:

- The cached Panda URDF visual meshes at an IK-computed contact pose, with the table, paper, paperweights and brush dimensions used by the viewer.
- The actual `XuanPaperCanvas.image` after replaying the built-in `yi` waypoint list through `PandaKinematics.solve_ik()`, `RealisticDeformableTuft.compute_deformation()` and `XuanPaperCanvas.deposit_ink()`.
- The computed tuft mesh and bristle segments at the replay sample with the largest model normal-force estimate. Axes in this panel are millimetres relative to the contact tip.

Generation used 28 samples per waypoint segment with the viewer's smoothstep interpolation: 196 samples in total, 154 producing contact. Nominal motion was followed by the existing IK solver; the figure does not substitute desired waypoints for actual tip positions. The displayed contact sample has about 7.2 mm nominal compression and a 1.64 N model force estimate. These values describe the source's parametric model, rather than a hardware measurement.

The source simulation environment supplied NumPy, Pillow, the robot assets and the actual simulation functions. Matplotlib 3.8.0 from the existing RL environment plotted their computed data. The image was generated on 2026-10-20; no simulator Python source or environment packages were changed to produce it.

To see the live controls and animated scene, follow the [launch instructions](../README.md#launch-with-the-existing-environment). The PNG preserves the code's native raster orientation, so interpret its paper axes together with the 3D scene rather than treating the flat image as an independently typeset character.
