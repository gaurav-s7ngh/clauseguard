import sys
from pathlib import Path

# Automatically ensure both project root (for 'ml') and backend dir (for 'app') are in sys.path
_current_file = Path(__file__).resolve()
_app_dir = _current_file.parent
_backend_dir = _app_dir.parent
_project_root = _backend_dir.parent

if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

