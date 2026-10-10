"""License safety gate: never publish gated/restrictive models by accident."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LicenseVerdict:
    allowed: bool
    license: str | None
    reason: str


def check_publish_allowed(license_name: str | None, gated: bool, explicit_consent: bool) -> LicenseVerdict:
    """Opt-in publishing rule.

    - Gated models: always refused (cannot verify the user accepted terms).
    - Restrictive-looking licenses: need explicit `--i-accept-license` consent.
    - Permissive/unknown: allowed, but unknown is flagged in the reason.
    """
    from quantiv.analyzer.model_analyzer import _license_flag

    if gated:
        return LicenseVerdict(False, license_name, "model is gated on the Hub — publishing refused")
    if _license_flag(license_name, []):
        if not explicit_consent:
            return LicenseVerdict(
                False,
                license_name,
                f"license '{license_name}' looks restrictive — "
                "re-run with --i-accept-license to confirm publishing rights",
            )
        return LicenseVerdict(True, license_name, "restrictive license explicitly accepted by user")
    if not license_name:
        return LicenseVerdict(True, None, "no license detected — allowed with warning recorded")
    return LicenseVerdict(True, license_name, "permissive license")
