"""测试环境公共配置。

当前项目里的 app 模块使用的是脚本式导入：
- from config import ...
- from prompts import ...

为了让 pytest 在项目根目录运行时也能找到这些模块，
这里把 app/ 目录加入 sys.path。
"""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = PROJECT_ROOT / "app"

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

