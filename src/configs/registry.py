"""Central registry for YAML-defined simulation assets."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Iterable

import yaml

from configs.loaders import load_asset_spec
from configs.schema import (
    BaseAssetSpec,
    ManipulatorSpec,
    SatelliteManipulatorSpec,
    SatelliteSpec,
)

RegisteredAssetSpec = SatelliteSpec | ManipulatorSpec | SatelliteManipulatorSpec

_CONFIG_DIR = Path(__file__).resolve().parent / "assets"


class AssetRegistry:
    def __init__(self) -> None:
        self._specs: dict[str, RegisteredAssetSpec] = {}

    def register(self, spec: RegisteredAssetSpec) -> None:
        if spec.asset_id in self._specs:
            raise ValueError(f"Asset '{spec.asset_id}' is already registered")
        self._specs[spec.asset_id] = spec

    def __contains__(self, asset_id: object) -> bool:
        return asset_id in self._specs

    def load(
        self,
        path: str | Path,
        *,
        component_paths: dict[str, Path] | None = None,
    ) -> RegisteredAssetSpec:
        spec = load_asset_spec(path, component_paths=component_paths)
        self.register(spec)
        return spec

    def get(self, asset_id: str) -> RegisteredAssetSpec:
        try:
            return self._specs[asset_id]
        except KeyError as exc:
            raise KeyError(
                f"Unknown YAML asset '{asset_id}'. Available: {self.list_ids()}"
            ) from exc

    def list_ids(self, kind: str | None = None) -> list[str]:
        return sorted(
            asset_id
            for asset_id, spec in self._specs.items()
            if kind is None or spec.kind == kind
        )

    def load_directory(self, directory: str | Path) -> None:
        directory = Path(directory).resolve()
        paths = [
            path
            for path in sorted(directory.rglob("*.yaml"))
            if not any(
                part.startswith("_")
                for part in path.relative_to(directory).parts
            )
        ]
        component_paths: dict[str, Path] = {}
        for path in paths:
            with path.open("r", encoding="utf-8") as stream:
                raw = yaml.safe_load(stream)
            if not isinstance(raw, dict):
                raise ValueError(f"{path} must contain a YAML mapping")
            asset_id = str(raw.get("id", "")).strip()
            if not asset_id:
                raise ValueError(f"{path} must define a non-empty asset id")
            if asset_id in component_paths:
                raise ValueError(
                    f"Duplicate asset id {asset_id!r} in {component_paths[asset_id]} "
                    f"and {path}"
                )
            component_paths[asset_id] = path
        for path in paths:
            self.load(path, component_paths=component_paths)


def build_genesis_morph(spec: BaseAssetSpec, **overrides):
    """Build a Genesis URDF morph, optionally overriding YAML morph fields."""
    import genesis as gs

    omit_fine_collision = bool(overrides.pop("_omit_fine_collision", False))
    morph = spec.morph
    kwargs = {
        "file": spec.urdf.as_posix(),
        "pos": morph.pose.position,
        "euler": morph.pose.euler,
        "scale": morph.scale,
        "fixed": morph.fixed,
        "collision": morph.collision,
        "visualization": morph.visualization,
    }
    if morph.merge_fixed_links is not None:
        kwargs["merge_fixed_links"] = bool(morph.merge_fixed_links)
    if morph.links_to_keep:
        kwargs["links_to_keep"] = list(morph.links_to_keep)
    if morph.convexify and not omit_fine_collision:
        kwargs["convexify"] = True
    if morph.decompose_object_error_threshold is not None and not omit_fine_collision:
        kwargs["decompose_object_error_threshold"] = float(
            morph.decompose_object_error_threshold
        )
    if morph.coacd and not omit_fine_collision:
        kwargs["coacd_options"] = gs.options.CoacdOptions(**dict(morph.coacd))
    kwargs.update(overrides)
    return gs.morphs.URDF(**kwargs)


def build_genesis_asset(spec: BaseAssetSpec, **morph_overrides) -> dict:
    """Build the Genesis entity arguments lazily, after ``gs.init()``."""
    import genesis as gs

    return {
        "morph": build_genesis_morph(spec, **morph_overrides),
        "material": gs.materials.Rigid(
            gravity_compensation=spec.material.gravity_compensation
        ),
    }


registry = AssetRegistry()
registry.load_directory(_CONFIG_DIR)


def get_spec(asset_id: str) -> RegisteredAssetSpec:
    return registry.get(asset_id)


def get_legacy_params(asset_id: str) -> dict:
    return deepcopy(registry.get(asset_id).to_legacy_params())


def get_legacy_pid(asset_id: str) -> dict:
    spec = registry.get(asset_id)
    return deepcopy(spec.pid.to_legacy_dict(spec.asset_id))


def get_legacy_camera(asset_id: str) -> dict:
    return deepcopy(registry.get(asset_id).camera.to_legacy_dict())


def get_registered_asset(asset_id: str) -> dict:
    return build_genesis_asset(registry.get(asset_id))


def make_asset_morph(asset_id: str, **overrides):
    return build_genesis_morph(registry.get(asset_id), **overrides)


def make_starlink_morph(
    *,
    collision: bool = True,
    visualization: bool = True,
    keep_cylinder_link: bool = True,
    fine_cylinder_collision: bool = True,
    **overrides,
):
    """Backward-compatible Starlink morph builder backed by its YAML spec."""
    if not keep_cylinder_link:
        overrides["links_to_keep"] = []
    if not (keep_cylinder_link and fine_cylinder_collision):
        overrides["_omit_fine_collision"] = True
    return make_asset_morph(
        "starlink",
        collision=bool(collision),
        visualization=bool(visualization),
        **overrides,
    )


def registered_ids() -> Iterable[str]:
    return registry.list_ids()
