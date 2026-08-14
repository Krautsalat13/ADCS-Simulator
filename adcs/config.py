from dataclasses import dataclass
import numpy as np
import yaml


# Spacecraft
@dataclass
class SpacecraftConfig:
    mass_kg: float
    envelope_m: np.ndarray
    inertia_tensor_kgm2: np.ndarray


# Actuators
@dataclass
class ResidualDipoleConfig:
    magnitude_Am2: float

@dataclass
class MagnetorquersConfig:
    max_dipole_per_axis_Am2: float

@dataclass
class ReactionWheelsConfig:
    count: int
    max_torque_per_axis_mNm: float
    momentum_storage_per_axis_mNms: float
    wheel_inertia_kgm2: float

@dataclass
class ActuatorsConfig:
    residual_dipole: ResidualDipoleConfig
    magnetorquers: MagnetorquersConfig
    reaction_wheels: ReactionWheelsConfig

# Sensors
@dataclass
class GyroConfig:
    angle_random_walk_deg_sqrt_h: float
    bias_instability_deg_h: float
    misalignment_deg: np.ndarray
    rate_hz: float

@dataclass
class MagnetometerConfig:
    noise_nT: float
    bias_nT: float
    rate_hz: float

@dataclass
class SunSensorConfig:
    faces: int
    accuracy_deg: float
    eclipse_output: bool

@dataclass
class SensorConfig:
    gyro: GyroConfig
    magnetometer: MagnetometerConfig
    sun_sensor: SunSensorConfig

# Orbit
@dataclass
class OrbitConfig:
    altitude_km: float
    inclination_deg: float
    type: str
    tle_catalog_number: int | None
    tle_epoch: str | None


# Mission Configuration
@dataclass
class MissionConfig:
    spacecraft: SpacecraftConfig
    actuators: ActuatorsConfig
    sensors: SensorConfig
    orbit: OrbitConfig

def load_mission_config(path):
    with open(path) as f:
        raw = yaml.safe_load(f)

    spacecraft = SpacecraftConfig(
        mass_kg=raw["spacecraft"]["mass_kg"],
        envelope_m=np.array(raw["spacecraft"]["envelope_m"]),
        inertia_tensor_kgm2=np.array(raw["spacecraft"]["inertia_tensor_kgm2"])
    )

    residual_dipole = ResidualDipoleConfig(**raw["actuators"]["residual_dipole"])
    magnetorquers = MagnetorquersConfig(**raw["actuators"]["magnetorquers"])
    reaction_wheels = ReactionWheelsConfig(**raw["actuators"]["reaction_wheels"])

    actuators = ActuatorsConfig(
        residual_dipole=residual_dipole,
        magnetorquers=magnetorquers,
        reaction_wheels=reaction_wheels
    )

    gyro = GyroConfig(
        angle_random_walk_deg_sqrt_h=raw["sensors"]["gyro"]["angle_random_walk_deg_sqrt_h"],
        bias_instability_deg_h=raw["sensors"]["gyro"]["bias_instability_deg_h"],
        misalignment_deg=np.array(raw["sensors"]["gyro"]["misalignment_deg"]),
        rate_hz=raw["sensors"]["gyro"]["rate_hz"]
    )
    magnetometer = MagnetometerConfig(**raw["sensors"]["magnetometer"])
    sun_sensor = SunSensorConfig(**raw["sensors"]["sun_sensor"])

    sensors = SensorConfig(
        gyro=gyro,
        magnetometer=magnetometer,
        sun_sensor=sun_sensor
    )

    orbit = OrbitConfig(**raw["orbit"])
    return MissionConfig(spacecraft=spacecraft, actuators=actuators, sensors=sensors, orbit=orbit)