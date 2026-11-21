# audio-spectrogram-stft

给音频编辑器加频谱视图后，正弦波变成一片模糊、末帧还总丢：窗函数不做周期修正，频谱泄漏；分帧不补零，末帧丢弃；帧数公式用 len//hop，多算一帧；FFT 只算一半 bin 且未乘 2，幅度差 6 dB；dB 转换用 10log10 而不是 20log10；下限不钳制，log(0) 产生 -inf；mel 滤波器边界重合，增益翻倍；mel 覆盖超过奈奎斯特，越界；批量共用频谱缓冲，轨道之间串；hop=0 不校验，死循环。stft/spectrogram.py 是核心，tests/test_spectrogram.py 15个红。要修的是：按周期窗修正泄漏，补齐末帧并修正帧数，恢复单边 FFT 的幅度，改用 20log10 并钳制 dB 下限，修正 mel 三角滤波器的边界与奈奎斯特上限，隔离批量缓冲并校验 hop。

## 运行方式

在仓库根目录执行：

```bash
python -m pytest tests -q
```

初始环境预期只有失败，且恰好为：

```text
15 failed
```

不允许出现 collection、import、syntax 或 fixture 错误。

## 字段表

| 字段 | 取值 |
| --- | --- |
| 所属领域 | 游戏、图形与多媒体 |
| 子方向 | 音视频处理 |
| 任务类型 | Bugfix |
| 语言/框架 | Python 3, pytest |
| 环境可复现等级 | 无外部依赖 |

## 约束

- 仅使用 Python 标准库（math、cmath）与 pytest；离线运行，不使用随机数，不 sleep。
- `tests/test_spectrogram.py` 是评分用例，不可修改。
- 音频统一使用 `list[float]`；核心实现位于 `stft/spectrogram.py`。
