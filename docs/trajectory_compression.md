# 实验轨迹定期压缩

`scripts/compress_trajectories.py` 使用标准库，无新增依赖。配置为
`config/maintenance/trajectory_compression.json`：每次扫描结束后等待 600 秒，
压缩级别 3，文件至少 300 秒未修改。单进程低 CPU 优先级执行，不使用 GPU。

只处理 `results/` 中 `inter_<编号>.pkl`：所在运行目录有有效的
`summary.json`，目录名中的 `_pid<编号>` 对应进程已退出，而且文件足够旧。
摘要在 runner 的轨迹写入之前产生，因此不能仅凭摘要判断写入结束。
没有摘要、没有可识别 PID、PID 仍存在或符号链接的文件均跳过。
不会压缩请求、决策、配置、指标或服务日志；未完成/中断且缺少摘要的轨迹保留。

输出为同目录 `inter_<编号>.pkl.gz`。压缩先写临时文件，解压后的 SHA256
及字节数与原文件一致、原文件未变化且运行进程仍已退出时，原子发布归档，
持久化审计后删除原 `.pkl`。已有归档只有在内容一致时才允许移除原文件；
冲突或校验失败保留原件并记录错误。单实例文件锁避免重复启动。

```bash
# 预览，扫描后退出，不修改轨迹
uv run --locked python scripts/compress_trajectories.py --dry_run

# 一次压缩
uv run --locked python scripts/compress_trajectories.py --once

# 前台定期运行
uv run --locked python scripts/compress_trajectories.py
```

状态、PID、累计体积及扫描结果：`results/trajectory_compression/status.json`。
每个文件的 SHA256、压缩前后字节数与估算释放磁盘块数：`files.jsonl`。
错误：`errors.jsonl`。重启后状态累计从零开始，文件审计追加保留。
发送 SIGTERM 可在当前文件安全处理完成后退出；进程不会跨服务器重启自动恢复。
强制终止可能遗留隐藏 `.tmp` 文件；确认没有压缩进程后再检查这些临时文件。

当前实验汇总不读取 `.pkl`，压缩不改变实验配置、决策或指标。
分析归档时使用 `gzip.open(path, 'rb')` 再传入 `pickle.load`，或先解压恢复。
仅对可信的本项目轨迹使用 pickle 读取。
