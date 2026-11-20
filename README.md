# render-bloom-pyramid

## 题面

给夜景加泛光后，亮部边缘全是方块、整体还过曝：阈值只是硬切、knee 内没有过渡，亮部边界直接出现方块；2×2 均值降采样不处理奇数尺寸，最后一列被丢掉；放大退回最近邻，斜边和边缘全是锯齿；金字塔层数完全不校验，levels=0 时直接除零，层数过深还会堆出重复的 1×1 层；组合时把尺寸不一致的层直接相加，越界或错位；权重不归一，叠加结果整体过曝；strength 不钳制，超过 1 后高光溢出；validate_pyramid 不检查尺寸严格递减，结构错了也报通过；bloom_batch 让多张图共用同一座金字塔，图与图之间互相串味；全黑图不短路，浮点误差累积出非零底噪。bloom/pyramid.py 是核心，tests/test_pyramid.py 15个红。要修的是：实现 knee 内的软阈值、奇数尺寸 2×2 均值降采样、双线性放大、层数与可降采样深度校验、跨尺寸分层合成、权重归一、strength 钳制、金字塔结构校验、批量隔离与原子回滚、全黑图短路。

## 运行

在仓库根目录执行：

```bash
python -m pytest tests -q
```

初始失败环境预期恰好为 `15 failed`、`0 error`，且没有 collection、import 或 syntax error：

```text
15 failed
```

## 字段表

| 字段 | 取值 |
| --- | --- |
| 所属领域 | 游戏、图形与多媒体 |
| 子方向 | 图形渲染 |
| 任务类型 | Bugfix |
| 语言/框架 | Python 3, pytest |
| 环境可复现等级 | 无外部依赖 |

## 约束

- 仅使用 Python 标准库 `math` 与 `pytest`；离线、无随机数。
- 图像统一使用行主序的二维 `list[list[float]]` 灰度图，像素值为线性光、范围 `[0.0, 1.0]`。
- `bloom/pyramid.py` 中的 8 个函数均为 `NotImplementedError` 桩，需按 docstring 契约实现。
- `tests/` 不可修改；数值容差为绝对误差 `1e-6`。
