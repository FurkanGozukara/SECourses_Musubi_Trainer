"""empty"""

from . import toml_compat as _toml_compat

# toml 0.10 writes a string holding a backslash + "x" (C:\Users\xavier\...) as invalid TOML: fix that
# before any module saves a preset, a dataset TOML or a runtime TOML.
_toml_compat.install()
