"""Typed failures. Exit codes are mapped in the CLI."""

from __future__ import annotations


class RomanRecovererError(Exception):
    """Base error. ``exit_code`` is what the CLI returns."""

    exit_code = 1


class DataError(RomanRecovererError):
    """Invalid catalogues, coordinates, fluxes, or configuration."""

    exit_code = 2


class DetectorError(RomanRecovererError):
    """The external detector process failed or returned unusable output."""

    exit_code = 3


class DependencyError(RomanRecovererError):
    """A science dependency required for the requested operation is absent."""

    exit_code = 4

    def __init__(self, message: str, *, status: str = "SKIPPED_DEPENDENCY") -> None:
        super().__init__(message)
        self.status = status
