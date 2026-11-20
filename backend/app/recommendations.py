from __future__ import annotations

import math
from typing import Any


DISCLAIMER = (
    "Benchmark annotation only. These labels restate published AI4I failure-mode "
    "rules, the uncalibrated classifier score, and the Isolation Forest flag. "
    "They are not maintenance work orders and must not be used to operate equipment."
)
RUL_DISCLAIMER = (
    "Benchmark annotation only. This band is a demo label on a capped FD001 "
    "regression estimate. It is not a maintenance work order and must not be "
    "used to operate equipment."
)

# Thresholds published with the AI4I 2020 dataset's failure-mode definitions.
HDF_TEMPERATURE_GAP_K = 8.6
HDF_SPEED_RPM = 1380
POWER_LOW_W = 3500
POWER_HIGH_W = 9000
TOOL_WEAR_BAND = (200.0, 240.0)
OSF_LIMITS_MIN_NM = {"L": 11_000.0, "M": 12_000.0, "H": 13_000.0}
RUL_SHORT_CYCLES = 20.0
RUL_REVIEW_CYCLES = 50.0


def mechanical_power_watts(torque_nm: float, rotational_speed_rpm: float) -> float:
    """Power as defined for AI4I: torque (N·m) times rotational speed (rad/s)."""
    return torque_nm * rotational_speed_rpm * 2.0 * math.pi / 60.0


def _action(code: str, title: str, detail: str) -> dict[str, str]:
    return {"code": code, "title": title, "detail": detail}


def build_recommendation(
    reading: dict[str, Any],
    prediction: dict[str, Any],
    anomaly: dict[str, Any],
    zscore: dict[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        air = float(reading["air_temperature"])
        process = float(reading["process_temperature"])
        speed = float(reading["rotational_speed"])
        torque = float(reading["torque"])
        tool_wear = float(reading["tool_wear"])
        failure_score = float(prediction["failure_score"])
        threshold_assumption = float(anomaly["threshold_assumption"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Recommendation inputs contain missing or invalid values.") from error

    numeric_values = {
        "air_temperature": air,
        "process_temperature": process,
        "rotational_speed": speed,
        "torque": torque,
        "tool_wear": tool_wear,
        "failure_score": failure_score,
        "threshold_assumption": threshold_assumption,
    }
    if zscore is not None:
        for key in ("max_absolute_zscore", "threshold"):
            if key in zscore:
                numeric_values[key] = float(zscore[key])
    if not all(math.isfinite(value) for value in numeric_values.values()):
        raise ValueError("Recommendation inputs must contain finite numeric values.")

    product_type = str(reading["product_type"])
    if product_type not in OSF_LIMITS_MIN_NM:
        raise ValueError("Recommendation inputs contain an unsupported product type.")
    gap = process - air
    power = mechanical_power_watts(torque, speed)
    overstrain = tool_wear * torque
    overstrain_limit = OSF_LIMITS_MIN_NM[product_type]

    actions: list[dict[str, str]] = []
    reasons: list[dict[str, str]] = []
    if TOOL_WEAR_BAND[0] <= tool_wear <= TOOL_WEAR_BAND[1]:
        actions.append(
            _action(
                "twf",
                "Tool wear is inside the AI4I replacement band",
                (
                    f"Tool wear is {tool_wear:.1f} min. The published tool-wear "
                    "failure mode is assigned inside 200–240 min."
                ),
            )
        )
        reasons.append({
            "code": "twf",
            "message": f"Tool wear is {tool_wear:.1f} min, inside the AI4I 200–240 min band.",
        })
    if gap < HDF_TEMPERATURE_GAP_K and speed < HDF_SPEED_RPM:
        actions.append(
            _action(
                "hdf",
                "Heat-dissipation pattern matches an AI4I failure mode",
                (
                    f"Process-to-air gap is {gap:.2f} K at {speed:.0f} rpm. "
                    "The published condition is a gap below 8.6 K and speed below 1380 rpm."
                ),
            )
        )
        reasons.append({
            "code": "hdf",
            "message": f"Temperature gap is {gap:.2f} K at {speed:.0f} rpm.",
        })
    if power < POWER_LOW_W or power > POWER_HIGH_W:
        bound = "below 3500 W" if power < POWER_LOW_W else "above 9000 W"
        actions.append(
            _action(
                "pwf",
                "Calculated process power matches an AI4I failure mode",
                (
                    f"Torque and speed imply {power:.0f} W, which is {bound}. "
                    "AI4I defines power as torque times rotational speed in rad/s."
                ),
            )
        )
        reasons.append({
            "code": "pwf",
            "message": f"Calculated process power is {power:.0f} W.",
        })
    if overstrain > overstrain_limit:
        actions.append(
            _action(
                "osf",
                "Tool-wear and torque product matches an AI4I overstrain mode",
                (
                    f"Tool wear times torque is {overstrain:.0f} min·Nm for product "
                    f"type {product_type}. The published limit for this type is "
                    f"{overstrain_limit:.0f} min·Nm."
                ),
            )
        )
        reasons.append({
            "code": "osf",
            "message": (
                f"Tool wear × torque is {overstrain:.0f} min·Nm; "
                f"product type {product_type} limit is {overstrain_limit:.0f} min·Nm."
            ),
        })
    if prediction["priority"] in {"High", "Critical"} or prediction["predicted_failure"]:
        actions.append(
            _action(
                "model_score",
                "Classifier score is above a demo review band",
                (
                    f"The selected model returned an uncalibrated class-1 score of "
                    f"{float(prediction['failure_score']):.3f} "
                    f"({prediction['priority']} priority)."
                ),
            )
        )
        reasons.append({
            "code": "model_score",
            "message": (
                f"Selected classifier score is {failure_score:.3f} "
                f"with {prediction['priority']} priority."
            ),
        })
    if anomaly["is_anomaly"]:
        actions.append(
            _action(
                "anomaly",
                "Isolation Forest marked this reading as an outlier",
                (
                    "The reading is outside the inlier region of the model trained "
                    f"with a {float(anomaly['threshold_assumption']):.0%} contamination assumption."
                ),
            )
        )
        reasons.append({
            "code": "isolation_forest",
            "message": (
                "Isolation Forest flagged the reading as an outlier; "
                f"threshold assumption is {threshold_assumption:.0%}."
            ),
        })
    zscore_flags = zscore["exceeded_features"] if zscore is not None else []
    if zscore is not None and zscore["is_anomaly"]:
        actions.append(
            _action(
                "zscore",
                "Z-score statistical detection exceeded the baseline threshold",
                (
                    f"The feature(s) {', '.join(zscore_flags)} exceeded the "
                    f"{float(zscore['threshold']):.1f} absolute Z-score threshold."
                ),
            )
        )
        reasons.append({
            "code": "zscore",
            "message": (
                f"{len(zscore_flags)} sensor feature(s) exceeded the "
                f"{float(zscore['threshold']):.1f} Z-score threshold: "
                f"{', '.join(zscore_flags)}."
            ),
        })
    if not actions:
        actions.append(
            _action(
                "routine",
                "Continue routine monitoring",
                "No published AI4I failure-mode rule, elevated classifier score, or anomaly flag fired.",
            )
        )
        reasons.append({
            "code": "routine",
            "message": "No published AI4I rule, predicted-failure signal, or anomaly flag fired.",
        })

    mode_hits = sum(action["code"] in {"twf", "hdf", "pwf", "osf"} for action in actions)
    predicted_failure = bool(prediction["predicted_failure"])
    isolation_flag = bool(anomaly["is_anomaly"])
    zscore_flag = bool(zscore["is_anomaly"]) if zscore is not None else False
    if predicted_failure and (isolation_flag or zscore_flag):
        severity = "critical"
    elif predicted_failure or prediction["priority"] == "Critical":
        severity = "urgent"
    elif isolation_flag and zscore_flag:
        severity = "urgent"
    elif mode_hits >= 2:
        severity = "urgent"
    elif isolation_flag or zscore_flag or mode_hits == 1:
        severity = "warning"
    elif prediction["priority"] in {"High", "Moderate"}:
        severity = "warning"
    else:
        severity = "routine"

    return {
        "severity": severity,
        "actions": actions,
        "reasons": reasons,
        "disclaimer": DISCLAIMER,
        "basis": (
            "AI4I 2020 published failure-mode rules, uncalibrated class-1 score, "
            "and Isolation Forest flag"
        ),
    }


def build_rul_recommendation(predicted_rul: float) -> dict[str, Any]:
    if predicted_rul < RUL_SHORT_CYCLES:
        severity = "critical"
        action = _action(
            "rul_short",
            "Remaining-life estimate is inside the short demo band",
            (
                f"Estimated remaining life is {predicted_rul:.1f} cycles, below "
                f"{RUL_SHORT_CYCLES:.0f} cycles."
            ),
        )
    elif predicted_rul < RUL_REVIEW_CYCLES:
        severity = "watch"
        action = _action(
            "rul_review",
            "Remaining-life estimate is inside the review demo band",
            (
                f"Estimated remaining life is {predicted_rul:.1f} cycles, below "
                f"{RUL_REVIEW_CYCLES:.0f} cycles."
            ),
        )
    else:
        severity = "routine"
        action = _action(
            "rul_routine",
            "Continue routine monitoring",
            (
                f"Estimated remaining life is {predicted_rul:.1f} cycles, at or above "
                f"{RUL_REVIEW_CYCLES:.0f} cycles."
            ),
        )
    return {
        "severity": severity,
        "actions": [action],
        "disclaimer": RUL_DISCLAIMER,
        "basis": "Capped FD001 regression estimate and fixed demo cycle bands",
    }
