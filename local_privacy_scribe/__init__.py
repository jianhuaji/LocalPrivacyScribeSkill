"""
LocalPrivacyScribe - 本地高密会议多模态督办官

本模块提供本地化的会议录音转录和会议白板 OCR 识别能力，
结合端侧 LLM 进行信息交叉验证，最终产出结构化会议纪要并自动生成本地日程待办文件。

技术栈：
- ASR: SenseVoice 通过 OpenVINO 加速
- OCR: RapidOCR 通过 OpenVINO 加速
- LLM: Ollama + Qwen3.6-35B-A3B (本地推理)
- 输出: Markdown 纪要 + .ics 日程文件

所有计算均在本地完成，无需云端 API 调用。
"""

__version__ = "1.0.0"
__author__ = "ModelScope Agent Ecosystem"
__description__ = "本地高密会议多模态督办官 - 支持录音转录、OCR识别、纪要生成和日程导出"

from .skill import LocalPrivacyScribeSkill

__all__ = ["LocalPrivacyScribeSkill"]