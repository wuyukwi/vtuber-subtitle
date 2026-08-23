# VTuber Subtitle

一个纯命令行的日本 VTuber 直播录像字幕工具。它会从视频中提取音频，使用本地 Whisper 识别日语语音和时间轴，再调用 LLM 翻译成中文，最后生成 Aegisub 可以直接打开的 `.ass` 字幕文件。

支持以下模式：

- 日文原文字幕
- 日文 + 中文双语字幕
- OpenCode Go、OpenAI-compatible API、Gemini
- YAML 或 JSON 自定义术语表
- 自动缓存识别和翻译结果，支持中断后继续

## 工作流程

```text
视频文件
  -> FFmpeg 提取单声道 16 kHz 音频
  -> faster-whisper 识别日语和时间轴
  -> 按批次调用 LLM 翻译
  -> 合并日文与中文
  -> 导出双语 ASS 字幕
```

语音识别完全在本地完成。只有翻译文本会发送给所选择的 LLM API，不会上传原始视频或音频。

## 环境要求

- Windows、macOS 或 Linux
- Python 3.11 或更高版本
- FFmpeg，并且 `ffmpeg` 在系统 `PATH` 中
- 使用 GPU 时，需要兼容的 CUDA 环境；没有 GPU 也可以使用 CPU
- 翻译阶段需要 OpenCode Go、OpenAI、DeepSeek 或 Gemini 的 API Key

## 安装

Windows 使用 Scoop 安装 Python 和 FFmpeg：

```powershell
scoop install python
```

如果 Scoop 的 FFmpeg 清单无法下载，可以使用 WinGet：

```powershell
winget install --id Gyan.FFmpeg.Shared --exact
```

重新打开 PowerShell 后安装项目：

```powershell
cd C:\Users\Administrator\vtuber-subtitle
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

如果 PowerShell 禁止执行激活脚本，可以不激活环境，直接使用：

```powershell
.venv\Scripts\python.exe -m vtuber_subtitle.cli --help
```

## API 配置

复制配置模板：

```powershell
Copy-Item .env.example .env
notepad .env
```

`.env.example` 只是模板，程序实际读取的是项目根目录的 `.env`。不要把真实 API Key 写入 `.env.example`，也不要提交 `.env`。

### OpenCode Go

在 [OpenCode 控制台](https://opencode.ai/auth) 获取 Go API Key：

```env
OPENCODE_API_KEY=你的OpenCode_Go_API_Key
OPENCODE_BASE_URL=https://opencode.ai/zen/go/v1
OPENCODE_MODEL=gpt-5.6-luna
```

OpenCode Go 支持的模型列表可能变化，常用模型包括：

```text
gpt-5.6-luna

### OpenAI-compatible API

```env
LLM_API_KEY=你的API_Key
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini
```

DeepSeek 可以这样配置：

```env
LLM_API_KEY=你的DeepSeek_API_Key
LLM_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-chat
```

### Gemini

```env
GEMINI_API_KEY=你的Gemini_API_Key
GEMINI_MODEL=gemini-2.0-flash
```

## 基本用法

### 生成 OpenCode Go 双语字幕

```powershell
cd C:\Users\Administrator\vtuber-subtitle
.venv\Scripts\Activate.ps1

vtuber-subtitle input.mp4 `
  --output output.ass `
  --provider opencode-go `
  --model gpt-5.6-luna
```

也可以使用模块方式运行：

```powershell
.venv\Scripts\python.exe -m vtuber_subtitle.cli input.mp4 -o output.ass --provider opencode-go --model gpt-5.6-luna
```

默认 `--subtitle-mode` 是 `bilingual`，会为每个时间片生成日文和中文两条 ASS 对话。

### 只生成日文字幕

不需要配置 LLM API Key：

```powershell
vtuber-subtitle input.mp4 `
  --output japanese.ass `
  --subtitle-mode japanese
```

旧参数 `--skip-translation` 仍然可用。

### 按原视频时间处理指定片段

可以直接指定原视频的开始和结束时间。程序只提取并识别该区间，但输出的 ASS 时间轴仍然使用原视频的绝对时间，不会从零开始：

```powershell
vtuber-subtitle input.mp4 `
  -o part.ass `
  --start-time 00:20:00 `
  --end-time 00:25:00 `
  --provider opencode-go `
  --model gpt-5.6-luna
```

生成的字幕时间轴会落在 `00:20:00` 到 `00:25:00`，可以直接加载回完整视频。也支持秒数或 `MM:SS` 格式，例如 `--start-time 1200 --end-time 1500`。

如果已经用 FFmpeg 截取了片段，片段本身的时间轴从零开始，此时不使用 `--start-time` 即可。

### 使用人工 ASS 作为样式模板

可以导入已有的人工校轴 ASS，保留它的 `[Script Info]`、分辨率、字体、颜色、描边、位置和所有 Style 定义，只替换字幕事件：

```powershell
vtuber-subtitle input.mp4 `
  -o output.ass `
  --ass-template "人工校轴.ass" `
  --japanese-style "鹿乃                  ——1080p日文（上）" `
  --chinese-style "鹿乃                  ——1080p中文（下）" `
  --provider opencode-go `
  --model gpt-5.6-luna
```

模板中的 Style 名称必须完全匹配，包括空格、括号和全角字符。可以在 Aegisub 的样式管理器中复制名称。若不指定 `--ass-template`，程序使用内置的日文上方、中文下方样式。

## Glossary 术语表

术语表用于固定人名、昵称、组织名和专有名词的译法。

### YAML 映射格式

```yaml
ぺこら: 佩克拉
先輩: 前辈
星街すいせい: 星街彗星
鹿乃まほろ: 鹿乃まほろ
```

运行时传入：

```powershell
vtuber-subtitle input.mp4 -o output.ass --glossary glossary.yaml
```

### JSON 数组格式

```json
[
  {"source": "ぺこら", "translation": "佩克拉", "note": "昵称"},
  {"source": "先輩", "translation": "前辈"}
]
```

术语表会随每个翻译批次发送给模型。模型被要求只返回带有片段 ID 的 JSON 翻译结果，程序会检查 ID 是否完整，避免字幕错位。

翻译准则：人名、VTuber 名称、团体名、社团名、作品名等专有名词，如果 Glossary 没有明确对应译文，默认保留日文原文，不自行音译、意译或编造译名。只有在 Glossary 中明确指定译文时，才使用指定译法。例如：

```yaml
鹿乃まほろ: 鹿乃まほろ
星街すいせい: 星街彗星
```

## 命令行参数

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `video` | 输入视频或音频路径 | 必填 |
| `-o`, `--output` | 输出 ASS 路径 | 必填 |
| `--subtitle-mode` | `bilingual` 或 `japanese` | `bilingual` |
| `--provider` | `openai`、`deepseek`、`gemini`、`opencode-go` | `openai` |
| `--model` | LLM 模型 ID | 由 Provider 决定 |
| `--base-url` | OpenAI-compatible API 地址 | 由 Provider 决定 |
| `--glossary` | YAML/JSON 术语表路径 | 无 |
| `--asr-model` | Whisper 模型 | `large-v3` |
| `--device` | `auto`、`cpu`、`cuda` | `auto` |
| `--compute-type` | Whisper 计算类型 | `auto` |
| `--enable-vad` | 启用静音检测；可能减少漏轴但可能过滤短句 | 关闭 |
| `--start-time` | 原视频起始时间 | 无 |
| `--end-time` | 原视频结束时间 | 无 |
| `--ass-template` | ASS 样式模板路径 | 无 |
| `--japanese-style` | 模板中的日文 Style 名称 | `Japanese` |
| `--chinese-style` | 模板中的中文 Style 名称 | `Chinese` |
| `--batch-size` | 每次发送给 LLM 的片段数 | `20` |
| `--temperature` | 普通 Chat Completions 的温度 | `0.2` |
| `--work-dir` | 指定缓存目录 | 视频旁隐藏目录 |
| `--skip-translation` | 只导出日文字幕 | 关闭 |

查看完整帮助：

```powershell
vtuber-subtitle --help
```

## 缓存和断点续跑

默认缓存目录位于输入视频旁边；使用时间范围或 `--work-dir` 时应为每个任务使用独立目录：

```text
.<视频文件名>.vtuber-subtitle/
├─ audio.wav
├─ segments.json
└─ translated.json
```

指定时间范围时，音频缓存文件名会包含起止时间，避免把完整视频音频误当成片段音频。

- `audio.wav`：FFmpeg 提取的音频
- `segments.json`：日语识别结果和时间轴
- `translated.json`：完整翻译结果

如果任务中断，再次运行相同视频会复用已经完成的识别和翻译结果。缓存文件中可能包含视频台词，请根据需要保留或删除。

## ASS 字幕格式

生成的文件包含标准的：

- `[Script Info]`
- `[V4+ Styles]`
- `[Events]`

日文和中文使用独立 Style。默认日文显示在画面上方，中文显示在下方，生成后可以在 Aegisub 中继续调整字体、字号、颜色、位置和时间轴。

## 性能建议

- 默认使用 `large-v3`，优先保证日语识别准确率和不漏轴
- VAD 默认关闭，因为它可能误删短句、语气词和低音量说话
- GPU 显存不足：使用 `--asr-model medium --device cpu --compute-type int8`
- 有 NVIDIA GPU：使用 `--device cuda --compute-type float16`
- 环境噪声很大且静音较多时，再尝试 `--enable-vad`
- 先用 5 分钟片段测试，再处理整场直播
- 长视频建议保留缓存目录，避免重复消耗 API 额度
- 翻译速度和额度消耗主要取决于片段数量、`--batch-size` 和所选模型

## 测试

```powershell
cd C:\Users\Administrator\vtuber-subtitle
.venv\Scripts\python.exe -m pytest -q
```

当前测试覆盖：

- ASS 时间格式转换
- ASS 特殊字符转义
- Glossary JSON 加载
- Glossary 格式化

## 常见问题

### `OPENCODE_API_KEY` 缺失

确认文件名是 `.env`，而不是 `.env.example`，并且命令是在项目目录中运行。API Key 变量名必须是：

```env
OPENCODE_API_KEY=...
```

### OpenCode 返回 403

确认使用的是 OpenCode 控制台生成的 Key，而不是 OpenAI 或 DeepSeek Key。确认地址为：

```text
https://opencode.ai/zen/go/v1
```

### FFmpeg 不在 PATH

执行：

```powershell
ffmpeg -version
```

如果找不到命令，重新打开终端，或将 FFmpeg 的 `bin` 目录加入系统 PATH。

### Whisper 下载很慢

首次使用会从 Hugging Face 下载 `large-v3` 模型，模型较大且 CPU 推理会很慢。模型下载完成后会被本地缓存。若只做快速测试，可以显式使用 `--asr-model medium`，但这会牺牲一部分识别准确率。

### PowerShell 显示乱码

项目已将 CLI 输出设置为 UTF-8。若路径仍显示异常，使用新版 Windows Terminal 或重新打开 PowerShell。

## 安全注意事项

- 不要在聊天、截图、日志或 Git 提交中公开 API Key
- `.env` 已加入 `.gitignore`
- 如果 Key 意外泄露，应立即在对应服务控制台撤销并重新生成
- `.ass` 和缓存文件可能包含完整直播台词，请注意文件分享范围
