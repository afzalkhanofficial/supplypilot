"""
CmdStanPy path safety patch for Prophet.

Prophet's internal model_from_json / model_from_dict deserialization logic
attempts to set CmdStan's path using an internal wheel path that may not exist
on the local filesystem (especially on Windows). This module applies a monkey-patch
to cmdstanpy.set_cmdstan_path to silently swallow invalid path ValueErrors.
"""

import cmdstanpy

_real_set_cmdstan_path = cmdstanpy.set_cmdstan_path


def _safe_set_cmdstan_path(path: str) -> None:
    try:
        _real_set_cmdstan_path(path)
    except ValueError:
        pass


def apply_patch() -> None:
    """Apply the cmdstanpy path safety patch if not already applied."""
    cmdstanpy.set_cmdstan_path = _safe_set_cmdstan_path


# Apply immediately upon module import
apply_patch()
