"""Schema constants for Goal182 privileged critic observations."""

from __future__ import annotations

PC_CRITIC_PRIV_DIM = 30
PC_CRITIC_PRIV_BLOCK_NAMES = (
    "route_truth",
    "actuator",
    "disturbance",
    "sensor_safety",
)
PC_CRITIC_PRIV_GROUP_SLICES = {
    "route_truth": (0, 6),
    "actuator": (6, 16),
    "disturbance": (16, 22),
    "sensor_safety": (22, 30),
}
PC_CRITIC_PRIV_FEATURE_NAMES = (
    "route_cross_track_m",
    "route_vertical_error_m",
    "route_tangent_heading_error_rad",
    "route_tangent_pitch_error_rad",
    "route_curvature_1_per_m",
    "route_lookahead_heading_delta_rad",
    "actuator_rudder_norm",
    "actuator_stern_norm",
    "actuator_rpm_norm",
    "requested_action_rudder_norm",
    "requested_action_stern_norm",
    "requested_action_rpm_norm",
    "actuator_tracking_residual_rudder_norm",
    "actuator_tracking_residual_stern_norm",
    "actuator_tracking_residual_rpm_norm",
    "material_saturation_ema",
    "true_current_ned_x_norm",
    "true_current_ned_y_norm",
    "true_current_ned_z_norm",
    "current_innovation_body_x_norm",
    "current_innovation_body_y_norm",
    "current_innovation_body_z_norm",
    "usbl_age_norm",
    "dvl_bottom_age_norm",
    "usbl_valid",
    "dvl_bottom_valid",
    "nav_innovation_ned_x_norm",
    "nav_innovation_ned_y_norm",
    "nav_innovation_ned_z_norm",
    "safety_margin_norm",
)


__all__ = [
    "PC_CRITIC_PRIV_BLOCK_NAMES",
    "PC_CRITIC_PRIV_DIM",
    "PC_CRITIC_PRIV_FEATURE_NAMES",
    "PC_CRITIC_PRIV_GROUP_SLICES",
]
