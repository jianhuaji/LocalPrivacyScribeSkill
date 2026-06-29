---
name: local-privacy-scribe
description: 本地高密会议多模态督办官 - 接收本地录音文件(.wav)和会议白板截图(.png)，利用本地离线工具（OpenVINO加速的ASR和OCR）提取内容，产出结构化纪要并自动生成本地 .ics 日程待办文件。所有计算均在本地完成，无需云端API调用。
---

# LocalPrivacyScribe Skill

## 概述

LocalPrivacyScribe 是一个完全本地化的高保密会议多模态内容提取与纪要生成 Skill。它利用 OpenVINO 加速的 ASR（语音识别）和 OCR（光学字符识别）引擎，从本地录音文件和会议白板截图中提取文本内容，进行信息交叉验证，最终产出结构化会议纪要并自动生成日程待办文件。

## 核心特性

- **完全本地化**：所有计算均在本地完成，无需任何云端 API 调用，保障数据隐私
- **多模态输入**：同时处理音频（.wav）和图像（.png/.jpg）两种模态
- **OpenVINO 加速**：支持 NPU/GPU/CPU 异构设备调度
- **结构化输出**：生成 Markdown 纪要 + ICS 日程文件 + JSON 结构化数据
- **信息交叉验证**：音频转录与白板 OCR 内容互相印证，提高准确性
- **待办事项自动提取**：从会议内容中自动识别并提取行动项

## 技术架构

```
用户输入 (音频.wav + 图片.png)
        │
        ▼
┌─────────────────────────────────────────┐
│          MultimodalExtractor              │
│  ┌──────────────┐  ┌──────────────┐     │
│  │ ASR Engine    │  │ OCR Engine   │     │
│  │ (SenseVoice)  │  │ (RapidOCR)   │     │
│  │ OpenVINO 加速  │  │ OpenVINO 加速 │     │
│  │ NPU/GPU/CPU   │  │ NPU/GPU/CPU  │     │
│  └──────┬───────┘  └──────┬───────┘     │
└─────────┼──────────────────┼─────────────┘
          │                  │
          ▼                  ▼
    音频转录文本          OCR识别文本
          │                  │
          ▼                  ▼
    ┌──────────────────────────┐
    │    信息交叉验证 &          │
    │    会议纪要生成             │
    │  - 关键要点提取            │
    │  - 决策事项识别            │
    │  - 待办事项提取            │
    └──────────┬───────────────┘
               │
               ▼
    ┌──────────────────────────┐
    │      结构化输出            │
    │  - 📄 Markdown 纪要       │
    │  - 📅 ICS 日程文件        │
    │  - 📊 JSON 结构化数据     │
    └──────────────────────────┘
```

## 调用方式

### 参数 Schema

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `audio_path` | string | 是 | 本地 .wav 文件路径（16kHz 推荐，单声道） |
| `image_path` | string | 是 | 本地 .png/.jpg 文件路径 |
| `meeting_title` | string | 否 | 会议标题（默认自动生成） |
| `participants` | array | 否 | 参会人员列表 |
| `language` | string | 否 | 音频语言代码（默认 "auto"） |

### 返回结果

```json
{
  "status": "success|error",
  "audio_text": "音频转录内容",
  "image_text": "OCR识别内容",
  "audio_confidence": 0.95,
  "image_confidence": 0.90,
  "meeting_summary": {
    "title": "会议标题",
    "key_points": ["要点1", "要点2"],
    "decisions": ["决策1"],
    "action_items": [
      {
        "description": "完成报告",
        "assignee": "张三",
        "deadline": "2024-12-31",
        "priority": "high"
      }
    ]
  },
  "output_files": {
    "markdown": "/path/to/meeting.md",
    "ics": "/path/to/meeting.ics",
    "json": "/path/to/meeting.json"
  }
}
```

## 安装与使用

### 环境要求

- Python 3.8+
- pip

### 安装依赖

```bash
pip install -r requirements.txt
```

### 基本使用

```python
from local_privacy_scribe import LocalPrivacyScribeSkill

# 初始化 Skill
skill = LocalPrivacyScribeSkill(
    asr_device="CPU",    # 可选: NPU, GPU, CPU
    ocr_device="CPU",    # 可选: NPU, GPU, CPU
    output_dir="./output"
)

# 执行多模态提取
result = skill._local_call(
    audio_path="/path/to/meeting.wav",
    image_path="/path/to/whiteboard.png",
    meeting_title="Q4 产品规划会议",
    participants=["张三", "李四"]
)

# 处理结果
if result["status"] == "success":
    print(f"音频文本: {result['audio_text']}")
    print(f"白板文本: {result['image_text']}")
    print(f"待办事项: {len(result['meeting_summary']['action_items'])} 条")
```

### 运行测试

```bash
# 运行单元测试
python3 -m unittest local_privacy_scribe.test_skill_benchmark -v

# 运行完整验证
python3 local_privacy_scribe/validate_skill.py
```

## 文件结构

```
.
├── .github/skills/LocalPrivacyScribe/
│   └── SKILL.md                    # Skill 定义文件（本文件）
├── local_privacy_scribe/
│   ├── __init__.py                 # 包入口
│   ├── skill.py                    # Skill 核心实现
│   ├── openvino_backends.py        # OpenVINO ASR/OCR 引擎封装
│   ├── utils.py                    # 工具函数（纪要生成、文件导出）
│   ├── configuration.json          # 配置文件
│   ├── test_skill_benchmark.py     # 单元测试套件 (33 tests)
│   ├── validate_skill.py           # 本地验证脚本 (40 tests)
│   └── requirements.txt            # 依赖声明
├── .gitignore
└── README.md