from __future__ import annotations

from ..perception import AnalyticSectorPerceptionBackend, PointCloudSonarConfig
from ..scenarios import PlanarScenarioCase
from .planar_avoidance import PlanarAvoidanceEnv, PlanarEnvConfig


def build_analytic_perception(config: PlanarEnvConfig) -> AnalyticSectorPerceptionBackend:
    """Build the analytic 13-sector backend from a planar environment config."""

    if not isinstance(config, PlanarEnvConfig):
        raise TypeError("config must be a PlanarEnvConfig")
    sonar_config = PointCloudSonarConfig(
        ray_count=config.sensor_ray_count,
        field_of_view_rad=config.sensor_field_of_view_rad,
        min_range_m=config.sensor_min_range_m,
        max_range_m=config.sensor_max_range_m,
        vertical_half_width_m=config.sensor_vertical_half_width_m,
        origin_body_m=config.sensor_origin_body_m,
        validity_s=config.sensor_validity_s,
        sensor_id="analytic-13sector",
        observation_source=config.analytic_observation_source,
        perception_mode=config.analytic_perception_mode,
    )
    return AnalyticSectorPerceptionBackend(config=sonar_config)


def make_analytic_planar_env(
    config: PlanarEnvConfig,
    *,
    scenarios: tuple[PlanarScenarioCase, ...] | None = None,
) -> PlanarAvoidanceEnv:
    """Build a single-environment planar env with analytic 13-sector perception."""

    return PlanarAvoidanceEnv(
        config,
        scenarios=scenarios,
        perception=build_analytic_perception(config),
    )


__all__ = ["build_analytic_perception", "make_analytic_planar_env"]
