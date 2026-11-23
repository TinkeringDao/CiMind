"""Circadian Melanopic Equivalent Daylight Illuminance (m-EDI) & Home Assistant lighting curves."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CircadianTarget:
    phase: str
    target_kelvin: int
    target_brightness_pct: int
    recommended_m_edi_lux: float
    measured_m_edi_lux: float
    boost_required: bool


def compute_melanopic_edi(photopic_lux: float, cct_kelvin: int) -> float:
    """Estimate CIE S 026 Melanopic Equivalent Daylight Illuminance (m-EDI) from photopic lux & CCT.

    Uses a bounded piecewise linear approximation of the melanopic Daylight (D65) efficacy ratio
    (M/P ratio ~0.45 at 2200K warm amber up to ~1.00 at 6500K D65 daylight).
    """
    if photopic_lux < 0:
        raise ValueError("photopic_lux must be non-negative")
    if cct_kelvin < 1500 or cct_kelvin > 10000:
        raise ValueError("cct_kelvin must be between 1500 and 10000")

    clamped_k = max(2000, min(6500, cct_kelvin))
    mp_ratio = 0.42 + (clamped_k - 2000) * (0.58 / 4500.0)
    return round(photopic_lux * mp_ratio, 2)


def compute_circadian_target(
    *,
    hour_local: int,
    measured_photopic_lux: float,
    measured_cct_kelvin: int = 4000,
    bed_exit_motion: bool = False,
) -> CircadianTarget:
    """Compute dementia-friendly circadian lighting target for daytime and nocturnal phases."""
    if not 0 <= hour_local <= 23:
        raise ValueError("hour_local must be in 0..23")

    measured_medi = compute_melanopic_edi(measured_photopic_lux, measured_cct_kelvin)

    if 7 <= hour_local < 13:
        phase = "morning_anchor"
        target_k = 5500
        target_bri = 90
        rec_medi = 250.0
    elif 13 <= hour_local < 17:
        phase = "afternoon_sustain"
        target_k = 4200
        target_bri = 75
        rec_medi = 150.0
    elif 17 <= hour_local < 21:
        phase = "sundowning_deescalation"
        target_k = 2700
        target_bri = 45
        rec_medi = 40.0
    else:
        phase = "nocturnal_pathway" if bed_exit_motion else "nocturnal_sleep"
        target_k = 2000
        target_bri = 18 if bed_exit_motion else 0
        rec_medi = 5.0 if bed_exit_motion else 0.0

    boost_required = measured_medi < rec_medi and rec_medi >= 150.0
    return CircadianTarget(
        phase=phase,
        target_kelvin=target_k,
        target_brightness_pct=target_bri,
        recommended_m_edi_lux=rec_medi,
        measured_m_edi_lux=measured_medi,
        boost_required=boost_required,
    )
