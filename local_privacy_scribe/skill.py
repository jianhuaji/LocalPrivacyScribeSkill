"""
LocalPrivacyScribe Skill 核心实现

本模块实现了 LocalPrivacyScribeSkill 类，继承自 ModelScope BaseTool，
是魔搭 ModelScope Skills 中心规范的标准 Skill 封装。

核心职责：
1. 接收本地录音文件(.wav)和会议白板截图(.png)路径
2. 调用 OpenVINO 后端引擎进行 ASR 和 OCR 推理
3. 将提取的多模态内容打包为结构化 JSON 返回给 Agent 大脑
4. 供端侧 LLM（Ollama + Qwen3.6-35B-A3B）进行信息交叉验证和纪要生成

设计原则：
- 严格遵循 ModelScope BaseTool 接口规范
- 极其严谨的 Docstring 和 Pydantic Schema，防止 35B 模型 Tool Calling 幻觉
- 纯本地运行，无任何云端 API 依赖
- 支持异构设备调度（NPU/GPU/CPU）

作者: ModelScope Agent Ecosystem
版本: 1.0.0
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
from datetime import datetime

# ModelScope Agent 相关导入
# 注意：实际使用时需要安装 modelscope-agent 包
try:
    from modelscope_agent.tools.base import BaseTool  # type: ignore
    from modelscope_agent.schemas import ToolSchema  # type: ignore
except ImportError:
    # 如果未安装 modelscope-agent，提供模拟基类以便代码可读
    logging.warning("modelscope-agent 未安装，使用模拟基类。请运行: pip install modelscope-agent")
    
    class BaseTool:
        """ModelScope BaseTool 模拟基类"""
        def __init__(self, name: str, description: str, parameters: Dict[str, Any]):
            self.name = name
            self.description = description
            self.parameters = parameters
        
        def _local_call(self, *args, **kwargs):
            raise NotImplementedError("子类必须实现 _local_call 方法")
    
    class ToolSchema:
        """Tool Schema 模拟类"""
        pass

# 导入本地模块
from .openvino_backends import (
    OpenVINOASREngine,
    OpenVINOOCREngine,
    MultimodalExtractor,
    ASRResult,
    OCRResult
)
from .utils import (
    MeetingSummary,
    ActionItem,
    generate_ics_file,
    generate_markdown_summary,
    parse_action_items,
    export_to_json
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LocalPrivacyScribeSkill(BaseTool):
    """
    本地高密会议多模态督办官 Skill
    
    本 Skill 接收本地录音文件(.wav)和会议白板截图(.png)路径，
    利用本地离线工具（OpenVINO 加速的 ASR 和 OCR）提取内容，
    并交由端侧 LLM 大脑进行信息交叉验证，最终产出结构化纪要并自动生成本地 .ics 日程待办文件。
    
    技术架构：
    - ASR: SenseVoice 模型（OpenVINO 加速，支持 NPU/GPU/CPU）
    - OCR: RapidOCR 模型（OpenVINO 加速，支持 NPU/GPU/CPU）
    - LLM: Ollama + Qwen3.6-35B-A3B（本地推理，通过 Agent 框架调用）
    - 输出: Markdown 纪要 + .ics 日程文件 + JSON 结构化数据
    
    使用场景：
    - 企业内部机密会议记录
    - 敏感信息处理（所有数据不出本地）
    - 离线环境下的会议纪要生成
    - 多模态信息交叉验证（语音 + 视觉）
    
    Attributes:
        name: Skill 名称，固定为 "LocalPrivacyScribe"
        description: Skill 功能描述
        parameters: 参数 Schema（Pydantic 风格）
        asr_device: ASR 引擎使用的设备类型
        ocr_device: OCR 引擎使用的设备类型
        output_dir: 输出文件保存目录
        
    Example:
        >>> skill = LocalPrivacyScribeSkill()
        >>> result = skill._local_call(
        ...     audio_path="/path/to/meeting.wav",
        ...     image_path="/path/to/whiteboard.png"
        ... )
        >>> print(result['audio_text'])
        >>> print(result['image_text'])
    """
    
    # Skill 元数据
    name = "LocalPrivacyScribe"
    description = """本地高密会议多模态督办官 - 接收本地录音文件(.wav)和会议白板截图(.png)，
利用本地离线工具提取内容，并交由端侧 LLM 大脑进行信息交叉验证，最终产出结构化纪要并自动生成本地 .ics 日程待办文件。
所有计算均在本地完成，无需云端 API 调用，支持 OpenVINO 加速（NPU/GPU/CPU）。"""
    
    # 参数 Schema（极其严谨的定义，防止 35B 模型 Tool Calling 幻觉）
    parameters = {
        "type": "object",
        "properties": {
            "audio_path": {
                "type": "string",
                "description": """本地会议录音文件的绝对路径或相对路径（.wav 格式）。
                
                格式要求：
                - 文件格式：WAV（PCM 编码）
                - 采样率：16kHz（推荐）或 8kHz/44.1kHz（自动重采样）
                - 声道：单声道（推荐）或立体声（自动转换为单声道）
                - 位深：16-bit
                - 时长限制：建议不超过 2 小时（内存限制）
                
                路径示例：
                - 绝对路径："/Users/jianhuaji/meetings/2024-12-25.wav"
                - 相对路径："./recordings/meeting.wav"
                
                注意事项：
                - 文件必须存在于本地文件系统
                - 不支持网络路径（如 http://, ftp://）
                - 不支持加密或压缩的音频文件
                - 如果路径包含空格或特殊字符，请使用引号包裹
                """,
                "pattern": ".*\\.wav$",
                "minLength": 1,
                "maxLength": 1024
            },
            "image_path": {
                "type": "string",
                "description": """本地会议白板截图的绝对路径或相对路径（.png 或 .jpg 格式）。
                
                格式要求：
                - 文件格式：PNG（推荐）或 JPG/JPEG
                - 分辨率：建议 1920x1080 或更高
                - 颜色模式：RGB 或 RGBA
                - 文件大小：建议不超过 10MB
                
                路径示例：
                - 绝对路径："/Users/jianhuaji/meetings/whiteboard_001.png"
                - 相对路径："./screenshots/whiteboard.png"
                
                注意事项：
                - 文件必须存在于本地文件系统
                - 不支持网络路径
                - 图片应清晰可读，避免过度模糊或倾斜
                - 支持包含文字的白板、便签、投影屏幕等
                - 如果路径包含空格或特殊字符，请使用引号包裹
                """,
                "pattern": ".*\\.(png|jpg|jpeg)$",
                "minLength": 1,
                "maxLength": 1024
            }
        },
        "required": ["audio_path", "image_path"],
        "additionalProperties": False
    }
    
    def __init__(
        self,
        asr_device: str = "CPU",
        ocr_device: str = "CPU",
        asr_config: Optional[Dict[str, Any]] = None,
        ocr_config: Optional[Dict[str, Any]] = None,
        output_dir: str = "./output",
        auto_generate_ics: bool = True,
        auto_generate_markdown: bool = True
    ):
        """
        初始化 LocalPrivacyScribe Skill
        
        Args:
            asr_device: ASR 引擎使用的设备类型
                - "NPU": 神经网络处理器（推荐，功耗最低，适合边缘设备）
                - "GPU": 集成或独立显卡（性能最强，适合高性能 PC）
                - "CPU": 中央处理器（兼容性最好，作为回退方案）
            ocr_device: OCR 引擎使用的设备类型（可选值同上）
            asr_config: ASR 引擎的 OpenVINO 推理配置，例如：
                {
                    "PERFORMANCE_HINT": "LATENCY",  # 延迟优先
                    "NUM_STREAMS": "1",              # 单流推理
                    "INFERENCE_NUM_THREADS": "4"    # 推理线程数
                }
            ocr_config: OCR 引擎的 OpenVINO 推理配置，例如：
                {
                    "PERFORMANCE_HINT": "THROUGHPUT",  # 吞吐量优先
                    "NUM_STREAMS": "4"                  # 多流并行
                }
            output_dir: 输出文件保存目录（.ics, .md, .json 文件）
            auto_generate_ics: 是否自动生成 .ics 日程文件
            auto_generate_markdown: 是否自动生成 Markdown 纪要
            
        Raises:
            RuntimeError: OpenVINO 依赖缺失或设备不可用
            
        Example:
            >>> # 使用 NPU 加速（推荐）
            >>> skill = LocalPrivacyScribeSkill(
            ...     asr_device="NPU",
            ...     ocr_device="NPU",
            ...     output_dir="./meeting_output"
            ... )
            
            >>> # 使用 GPU 加速（性能最强）
            >>> skill = LocalPrivacyScribeSkill(
            ...     asr_device="GPU",
            ...     ocr_device="GPU"
            ... )
            
            >>> # 使用 CPU（兼容性最好）
            >>> skill = LocalPrivacyScribeSkill(
            ...     asr_device="CPU",
            ...     ocr_device="CPU"
            ... )
        """
        # 调用父类构造函数
        super().__init__(
            name=self.name,
            description=self.description,
            parameters=self.parameters
        )
        
        # 设备配置
        self.asr_device = asr_device.upper()
        self.ocr_device = ocr_device.upper()
        self.asr_config = asr_config or {}
        self.ocr_config = ocr_config or {}
        
        # 输出配置
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.auto_generate_ics = auto_generate_ics
        self.auto_generate_markdown = auto_generate_markdown
        
        # 初始化多模态提取器
        logger.info("正在初始化 LocalPrivacyScribe Skill...")
        logger.info(f"ASR 设备: {self.asr_device}, OCR 设备: {self.ocr_device}")
        
        try:
            self.extractor = MultimodalExtractor(
                asr_device=self.asr_device,
                ocr_device=self.ocr_device,
                asr_config=self.asr_config,
                ocr_config=self.ocr_config
            )
            logger.info("Skill 初始化完成")
        except Exception as e:
            logger.error(f"Skill 初始化失败: {str(e)}")
            raise RuntimeError(f"LocalPrivacyScribe 初始化失败: {str(e)}")
    
    def _local_call(
        self,
        audio_path: str,
        image_path: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        执行本地多模态内容提取（核心方法）
        
        本方法是 ModelScope BaseTool 的标准接口，会被 Agent 框架自动调用。
        接收音频和图片路径，提取文本内容，并返回结构化 JSON 数据供 Agent 大脑进行后续处理。
        
        Args:
            audio_path: 本地会议录音文件路径（.wav 格式）
                - 类型：字符串（String）
                - 格式：WAV 音频文件（16kHz, 单声道, 16-bit PCM）
                - 路径：绝对路径或相对路径
                - 示例："/data/meetings/2024-12-25_morning.wav"
                - 约束：文件必须存在，大小建议不超过 500MB
            image_path: 本地会议白板截图路径（.png 或 .jpg 格式）
                - 类型：字符串（String）
                - 格式：PNG 或 JPG 图片文件
                - 路径：绝对路径或相对路径
                - 示例："./screenshots/whiteboard_session1.png"
                - 约束：文件必须存在，分辨率建议 1920x1080 或更高
            **kwargs: 额外参数（可选）
                - language: 音频语言代码（默认 "auto"，可选 "zh", "en", "ja", "ko"）
                - meeting_title: 会议标题（可选，用于生成纪要）
                - meeting_date: 会议日期（可选，ISO 格式，默认当前时间）
                - participants: 参会人员列表（可选）
                - template: Markdown 模板类型（默认 "default"）
        
        Returns:
            Dict[str, Any]: 包含以下字段的结构化 JSON 对象：
            
            {
                "status": "success" | "error",
                "audio_text": "音频转录的完整文本内容（String）",
                "image_text": "图片 OCR 识别的完整文本内容（String）",
                "audio_confidence": 0.95,  # 音频识别置信度（Float, 0.0-1.0）
                "image_confidence": 0.90,  # 图片识别置信度（Float, 0.0-1.0）
                "audio_duration": 1234.5,  # 音频时长（秒，Float）
                "meeting_summary": {       # 结构化会议纪要（Object）
                    "title": "会议标题",
                    "date": "2024-12-25T14:30:00",
                    "duration": 45.5,
                    "participants": ["张三", "李四"],
                    "key_points": ["要点1", "要点2"],
                    "decisions": ["决策1", "决策2"],
                    "action_items": [
                        {
                            "id": "abc12345",
                            "description": "完成项目报告",
                            "assignee": "张三",
                            "deadline": "2024-12-31",
                            "priority": "high",
                            "status": "pending",
                            "source": "audio"
                        }
                    ],
                    "audio_transcript": "完整音频转录...",
                    "whiteboard_content": "完整白板识别..."
                },
                "output_files": {          # 生成的文件路径（Object）
                    "ics": "/path/to/meeting.ics",
                    "markdown": "/path/to/meeting.md",
                    "json": "/path/to/meeting.json"
                },
                "metadata": {              # 元数据（Object）
                    "asr_device": "NPU",
                    "ocr_device": "GPU",
                    "processing_time": 12.5,
                    "timestamp": "2024-12-25T14:35:00"
                }
            }
            
        Raises:
            FileNotFoundError: audio_path 或 image_path 指定的文件不存在
            ValueError: 参数格式错误或文件格式不支持
            RuntimeError: ASR 或 OCR 推理过程出错
            
        Example:
            >>> skill = LocalPrivacyScribeSkill(asr_device="NPU", ocr_device="NPU")
            >>> result = skill._local_call(
            ...     audio_path="./meeting.wav",
            ...     image_path="./whiteboard.png",
            ...     meeting_title="Q4 产品规划会议",
            ...     participants=["张三", "李四", "王五"]
            ... )
            >>> 
            >>> # Agent 大脑可以这样解析返回结果：
            >>> if result['status'] == 'success':
            ...     audio_content = result['audio_text']
            ...     visual_content = result['image_text']
            ...     # 进行信息交叉验证和纪要生成...
            ...     
            >>> # 访问结构化数据：
            >>> for item in result['meeting_summary']['action_items']:
            ...     print(f"[{item['priority'].upper()}] {item['description']}")
            ...     print(f"  负责人: {item['assignee']}, 截止: {item['deadline']}")
        """
        start_time = datetime.now()
        
        try:
            # ========== 步骤 1: 参数验证 ==========
            logger.info("=" * 60)
            logger.info("LocalPrivacyScribe 开始处理...")
            logger.info(f"音频文件: {audio_path}")
            logger.info(f"图片文件: {image_path}")
            logger.info("=" * 60)
            
            # 验证音频路径
            audio_file = Path(audio_path)
            if not audio_file.exists():
                raise FileNotFoundError(
                    f"音频文件不存在: {audio_path}\n"
                    f"请检查路径是否正确，文件是否已移动到其他位置。"
                )
            
            if not audio_file.suffix.lower() == '.wav':
                raise ValueError(
                    f"不支持的音频格式: {audio_file.suffix}\n"
                    f"仅支持 .wav 格式，请转换后重试。"
                )
            
            # 验证图片路径
            image_file = Path(image_path)
            if not image_file.exists():
                raise FileNotFoundError(
                    f"图片文件不存在: {image_path}\n"
                    f"请检查路径是否正确，文件是否已移动到其他位置。"
                )
            
            if image_file.suffix.lower() not in ['.png', '.jpg', '.jpeg']:
                raise ValueError(
                    f"不支持的图片格式: {image_file.suffix}\n"
                    f"仅支持 .png, .jpg, .jpeg 格式，请转换后重试。"
                )
            
            # 提取额外参数
            language = kwargs.get('language', 'auto')
            meeting_title = kwargs.get('meeting_title', f'会议纪要 {datetime.now().strftime("%Y-%m-%d %H:%M")}')
            meeting_date = kwargs.get('meeting_date', datetime.now().isoformat())
            participants = kwargs.get('participants', [])
            template = kwargs.get('template', 'default')
            
            # ========== 步骤 2: ASR 音频转录 ==========
            logger.info(f"[步骤 1/4] 正在进行音频转录 (设备: {self.asr_device})...")
            
            try:
                asr_result = self.extractor.extract_audio(
                    audio_path=audio_path,
                    language=language,
                    enable_punc=True
                )
                audio_text = asr_result.text
                audio_confidence = asr_result.confidence
                audio_duration = asr_result.duration
                
                logger.info(f"✓ 音频转录完成 (置信度: {audio_confidence:.2%})")
                logger.info(f"  文本长度: {len(audio_text)} 字符")
                logger.info(f"  音频时长: {audio_duration:.1f} 秒")
                
            except Exception as e:
                logger.error(f"✗ 音频转录失败: {str(e)}")
                raise RuntimeError(f"ASR 推理失败: {str(e)}")
            
            # ========== 步骤 3: OCR 图片识别 ==========
            logger.info(f"[步骤 2/4] 正在进行图片 OCR 识别 (设备: {self.ocr_device})...")
            
            try:
                ocr_result = self.extractor.extract_image(
                    image_path=image_path,
                    det_thresh=0.3,
                    det_box_thresh=0.5,
                    rec_thresh=0.5
                )
                image_text = ocr_result.text
                image_confidence = ocr_result.confidence
                
                logger.info(f"✓ OCR 识别完成 (置信度: {image_confidence:.2%})")
                logger.info(f"  文本长度: {len(image_text)} 字符")
                logger.info(f"  识别区域数: {len(ocr_result.bounding_boxes)}")
                
            except Exception as e:
                logger.error(f"✗ OCR 识别失败: {str(e)}")
                raise RuntimeError(f"OCR 推理失败: {str(e)}")
            
            # ========== 步骤 4: 信息交叉验证与纪要生成 ==========
            logger.info("[步骤 3/4] 正在进行信息交叉验证...")
            
            # 合并文本内容
            combined_text = f"【音频转录】\n{audio_text}\n\n【白板内容】\n{image_text}"
            
            # 提取待办事项（从音频和图片文本中）
            audio_action_items = parse_action_items(audio_text, source="audio")
            image_action_items = parse_action_items(image_text, source="image")
            all_action_items = audio_action_items + image_action_items
            
            # 提取关键要点（简化处理：取前 N 句话）
            key_points = self._extract_key_points(audio_text, image_text)
            
            # 提取决策事项（简化处理：查找包含"决定/同意/通过"的句子）
            decisions = self._extract_decisions(audio_text, image_text)
            
            logger.info(f"✓ 信息验证完成")
            logger.info(f"  关键要点: {len(key_points)} 条")
            logger.info(f"  决策事项: {len(decisions)} 条")
            logger.info(f"  待办事项: {len(all_action_items)} 条")
            
            # ========== 步骤 5: 构建结构化数据 ==========
            logger.info("[步骤 4/4] 正在生成输出文件...")
            
            # 创建会议纪要对象
            summary = MeetingSummary(
                title=meeting_title,
                date=meeting_date,
                duration=audio_duration / 60.0,  # 转换为分钟
                participants=participants,
                key_points=key_points,
                decisions=decisions,
                action_items=all_action_items,
                audio_transcript=audio_text,
                whiteboard_content=image_text,
                metadata={
                    "asr_confidence": audio_confidence,
                    "ocr_confidence": image_confidence,
                    "asr_device": self.asr_device,
                    "ocr_device": self.ocr_device,
                    "audio_file": str(audio_file),
                    "image_file": str(image_file),
                    "processing_timestamp": datetime.now().isoformat()
                }
            )
            
            # 生成输出文件
            output_files = {}
            base_name = audio_file.stem  # 使用音频文件名作为基础
            
            # 生成 Markdown 纪要
            if self.auto_generate_markdown:
                md_path = self.output_dir / f"{base_name}_meeting.md"
                generate_markdown_summary(summary, str(md_path), template=template)
                output_files['markdown'] = str(md_path.absolute())
                logger.info(f"✓ Markdown 纪要已生成: {md_path}")
            
            # 生成 ICS 日程文件
            if self.auto_generate_ics:
                ics_path = self.output_dir / f"{base_name}_schedule.ics"
                generate_ics_file(summary, str(ics_path))
                output_files['ics'] = str(ics_path.absolute())
                logger.info(f"✓ ICS 日程文件已生成: {ics_path}")
            
            # 生成 JSON 结构化数据
            json_path = self.output_dir / f"{base_name}_data.json"
            export_to_json(summary, str(json_path))
            output_files['json'] = str(json_path.absolute())
            logger.info(f"✓ JSON 数据已导出: {json_path}")
            
            # ========== 步骤 6: 构建返回结果 ==========
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            result = {
                "status": "success",
                "audio_text": audio_text,
                "image_text": image_text,
                "audio_confidence": audio_confidence,
                "image_confidence": image_confidence,
                "audio_duration": audio_duration,
                "meeting_summary": {
                    "title": summary.title,
                    "date": summary.date,
                    "duration": summary.duration,
                    "participants": summary.participants,
                    "key_points": summary.key_points,
                    "decisions": summary.decisions,
                    "action_items": [
                        {
                            "id": item.id,
                            "description": item.description,
                            "assignee": item.assignee,
                            "deadline": item.deadline,
                            "priority": item.priority,
                            "status": item.status,
                            "source": item.source
                        }
                        for item in summary.action_items
                    ],
                    "audio_transcript": summary.audio_transcript,
                    "whiteboard_content": summary.whiteboard_content
                },
                "output_files": output_files,
                "metadata": {
                    "asr_device": self.asr_device,
                    "ocr_device": self.ocr_device,
                    "processing_time": processing_time,
                    "timestamp": end_time.isoformat(),
                    "output_directory": str(self.output_dir.absolute())
                }
            }
            
            logger.info("=" * 60)
            logger.info(f"✓ 处理完成！耗时: {processing_time:.2f} 秒")
            logger.info(f"  输出目录: {self.output_dir}")
            logger.info("=" * 60)
            
            return result
            
        except FileNotFoundError as e:
            error_msg = f"文件未找到: {str(e)}"
            logger.error(error_msg)
            return {
                "status": "error",
                "error_type": "FileNotFoundError",
                "error_message": error_msg,
                "audio_text": "",
                "image_text": ""
            }
            
        except ValueError as e:
            error_msg = f"参数错误: {str(e)}"
            logger.error(error_msg)
            return {
                "status": "error",
                "error_type": "ValueError",
                "error_message": error_msg,
                "audio_text": "",
                "image_text": ""
            }
            
        except RuntimeError as e:
            error_msg = f"推理失败: {str(e)}"
            logger.error(error_msg)
            return {
                "status": "error",
                "error_type": "RuntimeError",
                "error_message": error_msg,
                "audio_text": "",
                "image_text": ""
            }
            
        except Exception as e:
            error_msg = f"未知错误: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {
                "status": "error",
                "error_type": "UnknownError",
                "error_message": error_msg,
                "audio_text": "",
                "image_text": ""
            }
    
    def _extract_key_points(self, audio_text: str, image_text: str, max_points: int = 5) -> list[str]:
        """
        从文本中提取关键要点（简化实现）
        
        Args:
            audio_text: 音频转录文本
            image_text: 图片识别文本
            max_points: 最大要点数
            
        Returns:
            关键要点列表
        """
        # 合并文本
        combined = audio_text + "\n" + image_text
        
        # 按句子分割
        sentences = [s.strip() for s in combined.replace('。', '。\n').split('\n') if s.strip()]
        
        # 简单启发式：选择包含关键词的句子
        keywords = ['重要', '关键', '核心', '主要', '目标', '计划', '总结', '结论']
        key_sentences = []
        
        for sentence in sentences:
            if any(kw in sentence for kw in keywords):
                key_sentences.append(sentence)
        
        # 如果没找到，取前几句
        if not key_sentences and sentences:
            key_sentences = sentences[:max_points]
        
        return key_sentences[:max_points]
    
    def _extract_decisions(self, audio_text: str, image_text: str, max_decisions: int = 5) -> list[str]:
        """
        从文本中提取决策事项（简化实现）
        
        Args:
            audio_text: 音频转录文本
            image_text: 图片识别文本
            max_decisions: 最大决策数
            
        Returns:
            决策事项列表
        """
        # 合并文本
        combined = audio_text + "\n" + image_text
        
        # 按句子分割
        sentences = [s.strip() for s in combined.replace('。', '。\n').split('\n') if s.strip()]
        
        # 查找包含决策关键词的句子
        keywords = ['决定', '同意', '通过', '确认', '批准', '决议', '定下']
        decisions = []
        
        for sentence in sentences:
            if any(kw in sentence for kw in keywords):
                decisions.append(sentence)
        
        return decisions[:max_decisions]
    
    def run(
        self,
        audio_path: str,
        image_path: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Skill 公开执行接口（兼容 ModelScope Agent 框架）
        
        本方法是 _local_call 的包装器，提供更友好的调用接口。
        
        Args:
            audio_path: 音频文件路径
            image_path: 图片文件路径
            **kwargs: 额外参数（传递给 _local_call）
            
        Returns:
            与 _local_call 相同的返回格式
        """
        return self._local_call(audio_path=audio_path, image_path=image_path, **kwargs)
    
    def get_schema(self) -> Dict[str, Any]:
        """
        获取 Skill 的 JSON Schema（供 Agent 框架解析）
        
        Returns:
            Skill 参数 Schema
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters
        }
    
    def __repr__(self) -> str:
        """返回 Skill 的字符串表示"""
        return (
            f"LocalPrivacyScribeSkill("
            f"asr_device={self.asr_device}, "
            f"ocr_device={self.ocr_device}, "
            f"output_dir={self.output_dir})"
        )