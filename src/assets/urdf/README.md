# URDF asset layout

The data-driven fleet assets use three canonical directories:

- `arms/`: standalone manipulator URDFs and their local meshes.
- `satellites/`: standalone satellite URDFs, local meshes, and mount metadata.
- `combinations/`: one merged URDF for each satellite/manipulator pair. Mesh paths
  remain relative to `arms/` and `satellites/` so geometry is not duplicated.

Runtime registration lives in `src/configs/assets`. Each public asset has one YAML
file. A satellite-manipulator declares its source assets with `components`; the
loader takes manipulator control, PID, TCP, reference-pose, and camera settings
from the referenced standalone manipulator config. Fields declared by the combined
asset override component fields, for example mounting the base camera on the
combined URDF's satellite `base_link`.

`extends` is only YAML default inheritance. It is currently used by standalone
satellites to share generic spawn/material defaults from `_profiles/satellite.yaml`.
It does not mean that two assets are physically combined. Files below `_profiles`
are not registered directly.

Fields listed under `parameters.calibration.pending` are placeholders that require
task-specific measurements. Standalone satellite grasp sites currently use the
origin of each geometry-screened satellite mount frame. Combined assets do not own
target grasp sites; their satellite/arm attachment transform is already encoded by
the fixed joint in the combined URDF. Link names, joint names, mesh paths, joint
limits, and known SLDASM9 control values come from the supplied assets or the
existing project configuration.

`airbus_vispa` has no gripper joints in its source URDF. Its standalone and combined
configs therefore declare an empty gripper and `capture_capability:
no_gripper_defined`; they can be loaded and controlled as six-axis manipulators but
cannot complete the current two-finger capture task until an end effector is added.
