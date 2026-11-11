# Asset configuration layout

- `satellites/`: standalone target satellites, including target grasp sites and
  manipulator mount metadata.
- `manipulators/`: standalone arm mechanics, controls, PID, reference poses,
  TCP, and arm-mounted cameras.
- `satellite_manipulators/`: assembled URDFs and assembly-specific overrides.
- `_profiles/`: reusable YAML defaults loaded through `extends`; profile files
  are not registered as assets.

Combination files identify their physical components with `components`. The
registry resolves those IDs across the subdirectories and copies the
manipulator-owned configuration into the assembled asset. Values written in a
combination file override the component values recursively, which is useful for
combined-URDF link names such as the satellite-root base-camera mount.

`extends` is generic YAML default inheritance. It is separate from physical
component composition and should only be used when two configs intentionally
share the same defaults.
