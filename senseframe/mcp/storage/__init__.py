"""持久化层（下阶段实现，SenseFrame 暂不引入 SQLite）。

后续阶段如需持久化将实现：
- db.py：transaction() + retry + drain check
- dao.py：Data Access Objects
- migrations.py：Schema 迁移

本阶段为空骨架，AST 守卫测试验证 tools/ 不 import storage.dao / storage.db。
"""
