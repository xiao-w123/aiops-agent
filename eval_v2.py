"""eval_v2.py：已合并到 eval_v3.py，本文件仅作兼容转发。

此前的 eval_v2.py 与 eval_v3.py 大段重复，只差"是否含证据校验"。
现在两种配置统一由 eval_v3.py 的 --configs 参数控制，避免同一逻辑维护两份：

    python eval_v3.py --configs basic      # 等价于旧的 eval_v2.py
    python eval_v3.py --configs verified   # 等价于旧的 eval_v3.py
"""

import sys

from eval_v3 import main

if __name__ == "__main__":
    print("提示：eval_v2.py 已合并到 eval_v3.py，"
          "请使用 `python eval_v3.py --configs basic` 运行无证据校验的基线。\n")
    sys.exit(main())
