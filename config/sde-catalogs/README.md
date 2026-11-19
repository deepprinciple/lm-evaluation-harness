# SDE Codex model catalogs

这两份文件是本轮 question-level 实验的固定配置，配套 Codex CLI **0.162.1**。读者无需访问作者的模型缓存或历史结果目录。

## 来源与修改

- `astra.json`：从既有实验材料保存的 `gpt-6-astra` 模型条目快照；2026-10-12 核查时，它与本机 Codex `models_cache.json` 中的官方条目逐字段一致。此文件按原始字节保留，未改写。
- `flash.json`：沿用既有 Flash catalog 的模型 ID、context、effort、tool_mode 和能力字段；删除旧的 `base_instructions`，将 `model_messages` 替换为上述 Astra 快照中的同名对象。整个 `model_messages` 对象相同，包含 `instructions_template` 及其关联模板。它是实验接入配置，不是 DeepSeek 官方发布的 catalog。
- 旧 Flash catalog 的最初生成脚本尚未定位到；本次来源文件与 2026-10-04 实验使用的 Flash catalog 一致。能力字段被保留并校验，不将这些声明当作服务端已验证能力。

来源材料位置、原始文件校验值、修改清单、目标文件校验值和公共 `model_messages` 的校验值保存在 `manifest.json`。来源材料位置用于追溯，运行不依赖该目录。JSON 对象的校验使用 `json.dumps(value, sort_keys=True, ensure_ascii=False).encode()`；文件校验使用原始字节 SHA-256。

## 使用与复现边界

```bash
export SDE_CATALOG_DIR="$PWD/config/sde-catalogs"
```

一键脚本默认已使用本目录。若之前设置过指向历史材料的 `SDE_CATALOG_DIR`，需改为此目录或 `unset SDE_CATALOG_DIR`。

两组使用相同的基础指令模板，但模型能力字段保留各自配置。因此工具 schema、上下文容量和模型执行行为仍可能不同；不能声称完整请求逐字相同。模型名称仍分别为 `deepseek-flash` 和 `gpt-6-astra`，没有切换 Flash 的实际服务模型。

此模板以 Codex 官方 GPT-6 模型描述为来源，其中的自我描述保留原文；这属于 harness 指令，不是实际模型身份的证明。

修改 catalog 会改变实验身份。新的 Flash 实验应使用新 `--output`，不能 resume 到使用旧模板的 Flash 结果里。历史 catalog 和原始输出保留。

## 验证

```bash
python -m pytest -q tests/test_sde_catalogs.py
```

测试验证模型 ID、max effort、文件校验值、公共模板一致性，以及 Flash 其余能力字段没有改动。真实模型与原生工具是否可用仍需单独冒烟；catalog 加载通过不等于模型答题成功。
