from typing import Literal

from pydantic import BaseModel, Field


class SensorReading(BaseModel):
    product_type: Literal["L", "M", "H"]
    air_temperature: float = Field(gt=0, le=500)
    process_temperature: float = Field(gt=0, le=500)
    rotational_speed: float = Field(gt=0, le=10000)
    torque: float = Field(ge=0, le=1000)
    tool_wear: float = Field(ge=0, le=1000)


class MachineAssessmentRequest(SensorReading):
    asset_id: str = Field(
        default="manual-entry",
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9_.:-]+$",
    )


class RulReading(BaseModel):
    operational_setting_1: float = Field(ge=-100, le=100)
    operational_setting_2: float = Field(ge=-100, le=100)
    operational_setting_3: float = Field(ge=-100, le=100)
    sensor_1: float = Field(ge=-100000, le=100000)
    sensor_2: float = Field(ge=-100000, le=100000)
    sensor_3: float = Field(ge=-100000, le=100000)
    sensor_4: float = Field(ge=-100000, le=100000)
    sensor_5: float = Field(ge=-100000, le=100000)
    sensor_6: float = Field(ge=-100000, le=100000)
    sensor_7: float = Field(ge=-100000, le=100000)
    sensor_8: float = Field(ge=-100000, le=100000)
    sensor_9: float = Field(ge=-100000, le=100000)
    sensor_10: float = Field(ge=-100000, le=100000)
    sensor_11: float = Field(ge=-100000, le=100000)
    sensor_12: float = Field(ge=-100000, le=100000)
    sensor_13: float = Field(ge=-100000, le=100000)
    sensor_14: float = Field(ge=-100000, le=100000)
    sensor_15: float = Field(ge=-100000, le=100000)
    sensor_16: float = Field(ge=-100000, le=100000)
    sensor_17: float = Field(ge=-100000, le=100000)
    sensor_18: float = Field(ge=-100000, le=100000)
    sensor_19: float = Field(ge=-100000, le=100000)
    sensor_20: float = Field(ge=-100000, le=100000)
    sensor_21: float = Field(ge=-100000, le=100000)


class StreamStartRequest(BaseModel):
    interval_seconds: float = Field(default=2.0, ge=0.5, le=30)
