# 实验配置

配置按模型组织为 `config/<模型名>/<实验名>.json`，使用 `run_jevlight.py` 的参数名作为 JSON 字段，不带 `--`。布尔值使用 JSON 的 `true` / `false`；未知字段、错误类型和非法枚举会被拒绝。API key 只从环境变量或临时命令行参数提供，不允许写入配置。

模型配置按目录管理；MaxPressure配置只供后续独立任务，模型实验队列不读取或依赖基线。所有模型请求超时为120秒。

```bash
# 只检查参数，不启动 SUMO、不请求 API
uv run python run_jevlight.py --config config/jev/jinan_smoke.json --print_config

# 运行短 mock 实验，仍然需要 SUMO
uv sync --locked --extra sumo
uv run --extra sumo python run_jevlight.py --config config/jev/jinan_smoke.json

# 命令行显式覆盖配置
uv run python run_jevlight.py --config config/jev/jinan.json --count 900 --seed 2
```

优先级为命令行、配置文件、runner 默认值。保持原有无配置命令行用法；新建正式实验应使用配置文件。`--no-mock_jev`、`--no-eightphase` 等可以关闭布尔选项，拥堵题仅使用 `speculative` 字段，默认 `false`；`--speculative` 开启，`--speculative false` 关闭，CLI覆盖配置。

配置目录名用于结果的模型分组；模型服务具体 ID 使用 `jev_model`，本地服务地址使用 `jev_base_url` 和 `jev_transport: "local"`。增加模型时创建对应目录和配置，不需要复制 runner。

本地原生 batch 验证入口使用 `jev_batch_size`：默认 `0` 保持逐路口请求，正整数表示每次原生 forward 的路口状态上限。控制器将同一轮全部非空路口的独立请求交给 `/v1/systemone/batch`，保留各自的完整状态、Choice及显式开启时的Noul、候选顺序和问题 ID；空路口继续使用原有本地规则。优先提交完整时间步（最大随路网规模确定），显存不足时由服务分级拆块；服务 `serving.toml` 的 `batch_size` 决定允许的上限，容量适配器依据实际显存探索并缓存可用容量。这不会改为全网状态打包。

作者原生batch由对应模型和硬件上的实测决定。默认3次预热、10次测量，慢或超预算的方案保留诊断并继续准备，完成部署验收后再收集正式成绩；不设batch概率差门槛。检查相位、完整输入/映射、真实SUMO动作/指标和实际提速，保留显存余量。启用及成绩只写results与实际配置，不在文档记录状态。

`cojevlight` 固定 `packaging=per_intersection`（传入 `network` 报错），每个活跃路口单独请求，只包含自身和直接上下游邻居，不递归扩展邻居。只使用最新当前状态（`incoming-side-trajectory-lanes-v6`）：使用实际观测车辆转移学习直接上下游车道关系、邻居路口当前信息、当前相位及黄灯开销；不输入历史状态、过去动作或累计流向证据，不分快慢思考，不做提前推理；两个模式默认只发送相位题；显式开启 `speculative` 时才在同一次请求中追加拥堵题，结果仅记录，不直接控制信号。兼容字段 `cojev_history_steps` 固定为 0，非零值报错；两个模式默认 `speculative=false`；JSON使用布尔值，不再接受 `no_speculative`。`jevlight` 仍使用局部输入。方向按来车方位定义（如 NT 北→南、ET 东→西）。正式比较前需冻结当前版本，并用实际 tokenizer 核验完整当前状态请求长度；不得截断状态或候选。
