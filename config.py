# Backward-compat shim — the canonical Config now lives in parseiq.config.
# Existing code that does `import config` or `from config import Config` still works.
from parseiq.config import Config  # noqa: F401
