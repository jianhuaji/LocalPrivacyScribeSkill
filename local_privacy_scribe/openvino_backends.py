"""
OpenVINO 后端引擎封装模块

本模块封装了基于 OpenVINO 推理框架的 ASR（自动语音识别）和 OCR（光学字符识别）引擎。
严格遵循参赛铁律：优先使用 OpenVINO 加速，支持异构设备调度（NPU/GPU/CPU）。

技术细节：
- ASR: SenseVoice 模型（通过 OpenVINO IR 格式加载）
- OCR: RapidOCR 模型（通过 OpenVINO 加速）
- 设备支持: NPU (神经网络处理器), GPU (集成/独立显卡), CPU (回退方案)

作者: ModelScope Agent Ecosystem
版本: 1.0.0
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Union
from dataclasses import dataclass

import cv2
import numpy as np

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ASRResult:
    """ASR 识别结果数据结构"""
    text: str
    confidence: float
    language: str
    duration: float


@dataclass
class OCRResult:
    """OCR 识别结果数据结构"""
    text: str
    confidence: float
    bounding_boxes: list
    image_path: str


class OpenVINOASREngine:
    """
    基于 OpenVINO 的 SenseVoice ASR 引擎封装
    
    该引擎负责将音频文件（.wav）转换为文本，支持多语言识别。
    严格使用 OpenVINO 推理框架，支持 NPU/GPU/CPU 异构设备调度。
    
    Attributes:
        model_path: OpenVINO IR 模型路径（xml 文件）
        device: 推理设备，可选 "NPU", "GPU", "CPU"
        config: OpenVINO 推理配置字典
        
    Example:
        >>> engine = OpenVINOASREngine(device="NPU")
        >>> result = engine.transcribe("meeting.wav")
        >>> print(result.text)
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "CPU",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        初始化 OpenVINO ASR 引擎
        
        Args:
            model_path: OpenVINO IR 模型路径（.xml 文件）。如果为 None，则使用默认模型路径
            device: 推理设备类型
                - "NPU": 神经网络处理器（推荐，功耗最低）
                - "GPU": 集成或独立显卡（性能最强）
                - "CPU": 中央处理器（兼容性最好，作为回退方案）
            config: OpenVINO 推理配置，例如：
                {
                    "PERFORMANCE_HINT": "LATENCY",
                    "NUM_STREAMS": "1",
                    "INFERENCE_NUM_THREADS": "4"
                }
        """
        self.device = device.upper()
        self.config = config or {}
        self.model_path = model_path
        self._model = None
        self._compiled_model = None
        self._model_loaded = False
        
        # 验证设备支持
        self._validate_device()
        
        # 尝试加载模型（支持懒加载，模型不存在时不阻塞初始化）
        try:
            self._load_model()
            self._model_loaded = True
            logger.info(f"OpenVINO ASR 引擎初始化完成，使用设备: {self.device}")
        except (FileNotFoundError, RuntimeError) as e:
            logger.warning(f"ASR 模型暂未加载，将使用懒加载模式: {e}")
            logger.info(f"OpenVINO ASR 引擎初始化完成（懒加载模式），使用设备: {self.device}")
    
    def _validate_device(self) -> None:
        """验证目标设备是否可用"""
        try:
            from openvino.runtime import Core  # type: ignore
            
            core = Core()
            available_devices = core.available_devices
            
            if self.device not in available_devices:
                logger.warning(
                    f"设备 {self.device} 不可用，可用设备: {available_devices}。"
                    f"自动回退到 CPU 模式。"
                )
                self.device = "CPU"
                
        except ImportError:
            logger.error("OpenVINO 未安装，请运行: pip install openvino")
            raise RuntimeError("OpenVINO 依赖缺失")
    
    def _load_model(self) -> None:
        """加载 OpenVINO IR 格式的 SenseVoice 模型"""
        try:
            from openvino.runtime import Core  # type: ignore
            
            core = Core()
            
            # 模型路径解析
            if self.model_path is None:
                # 使用 ModelScope 预训练模型路径
                self.model_path = self._get_default_model_path()
            
            model_xml = Path(self.model_path)
            if not model_xml.exists():
                raise FileNotFoundError(f"模型文件不存在: {model_xml}")
            
            # 读取模型
            logger.info(f"正在加载 ASR 模型: {model_xml}")
            self._model = core.read_model(model=model_xml)
            
            # 编译模型到指定设备
            logger.info(f"正在编译模型到设备: {self.device}")
            self._compiled_model = core.compile_model(
                model=self._model,
                device_name=self.device,
                config=self.config
            )
            
            # 获取输入输出节点
            self._input_layer = self._compiled_model.input(0)
            self._output_layer = self._compiled_model.output(0)
            
            logger.info("ASR 模型加载成功")
            
        except Exception as e:
            logger.error(f"模型加载失败: {str(e)}")
            raise RuntimeError(f"ASR 模型加载失败: {str(e)}")
    
    def _get_default_model_path(self) -> str:
        """
        获取默认的 SenseVoice 模型路径
        
        Returns:
            模型 XML 文件的完整路径
        """
        # 优先检查本地缓存
        cache_dir = Path.home() / ".cache" / "modelscope" / "LocalPrivacyScribe"
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = cache_dir / "sensevoice_openvino.xml"
        
        if not model_path.exists():
            logger.warning(
                f"未找到本地模型: {model_path}\n"
                "请从 ModelScope 下载 SenseVoice 模型并转换为 OpenVINO IR 格式。\n"
                "参考: https://modelscope.cn/models/iic/SenseVoiceSmall/files"
            )
        
        return str(model_path)
    
    def transcribe(
        self,
        audio_path: str,
        language: str = "auto",
        enable_punc: bool = True
    ) -> ASRResult:
        """
        转录音频文件为文本
        
        Args:
            audio_path: 音频文件路径（.wav 格式，16kHz 采样率，单声道）
            language: 语言代码，可选 "auto", "zh", "en", "ja", "ko" 等
            enable_punc: 是否启用标点符号预测
            
        Returns:
            ASRResult: 包含文本、置信度、语言和时长的识别结果
            
        Raises:
            FileNotFoundError: 音频文件不存在
            RuntimeError: 推理过程出错
        """
        # 验证输入文件
        audio_file = Path(audio_path)
        if not audio_file.exists():
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")
        
        try:
            import soundfile as sf  # type: ignore
            import librosa  # type: ignore
            
            # 读取音频文件
            logger.info(f"正在读取音频: {audio_path}")
            audio_data, sample_rate = sf.read(audio_path)
            
            # 重采样到 16kHz（SenseVoice 要求）
            if sample_rate != 16000:
                logger.info(f"重采样: {sample_rate}Hz -> 16000Hz")
                audio_data = librosa.resample(
                    audio_data,
                    orig_sr=sample_rate,
                    target_sr=16000
                )
            
            # 转换为单声道
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            duration = len(audio_data) / 16000.0
            
            # 预处理：提取特征（这里简化处理，实际应根据模型要求提取 Fbank 等特征）
            input_tensor = self._preprocess_audio(audio_data)
            
            # 推理
            logger.info(f"正在使用 {self.device} 进行 ASR 推理...")
            result = self._compiled_model([input_tensor])[self._output_layer]
            
            # 后处理：解码为文本
            text, confidence = self._postprocess_output(result)
            
            logger.info(f"ASR 完成: {text[:50]}...")
            
            return ASRResult(
                text=text,
                confidence=confidence,
                language=language,
                duration=duration
            )
            
        except Exception as e:
            logger.error(f"音频转录失败: {str(e)}")
            raise RuntimeError(f"ASR 推理失败: {str(e)}")
    
    def _preprocess_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """
        音频预处理：提取模型所需的特征
        
        Args:
            audio_data: 原始音频数据（numpy 数组）
            
        Returns:
            预处理后的输入张量
        """
        # 归一化
        audio_data = audio_data.astype(np.float32)
        if np.max(np.abs(audio_data)) > 0:
            audio_data = audio_data / np.max(np.abs(audio_data))
        
        # 添加 batch 和 channel 维度
        input_tensor = np.expand_dims(audio_data, axis=0)
        input_tensor = np.expand_dims(input_tensor, axis=0)
        
        return input_tensor
    
    def _postprocess_output(self, output: np.ndarray) -> tuple[str, float]:
        """
        解码模型输出为文本
        
        Args:
            output: 模型原始输出
            
        Returns:
            (文本, 置信度) 元组
        """
        # 简化处理：实际应根据模型的 tokenizer 进行解码
        # 这里返回占位符，实际部署时需要集成 SenseVoice 的 tokenizer
        text = "[ASR 输出需要集成 SenseVoice Tokenizer]"
        confidence = 0.95
        
        return text, confidence


class OpenVINOOCREngine:
    """
    基于 OpenVINO 的 RapidOCR 引擎封装
    
    该引擎负责从会议白板截图中提取文本信息，支持多语言 OCR。
    使用 OpenVINO 加速推理，支持 NPU/GPU/CPU 异构设备调度。
    
    Attributes:
        model_path: OpenVINO IR 模型目录路径
        device: 推理设备，可选 "NPU", "GPU", "CPU"
        config: OpenVINO 推理配置
        
    Example:
        >>> engine = OpenVINOOCREngine(device="GPU")
        >>> result = engine.recognize("whiteboard.png")
        >>> print(result.text)
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "CPU",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        初始化 OpenVINO OCR 引擎
        
        Args:
            model_path: OpenVINO IR 模型目录路径。如果为 None，则使用默认模型路径
            device: 推理设备类型
                - "NPU": 神经网络处理器（推荐，功耗最低）
                - "GPU": 集成或独立显卡（性能最强）
                - "CPU": 中央处理器（兼容性最好，作为回退方案）
            config: OpenVINO 推理配置，例如：
                {
                    "PERFORMANCE_HINT": "THROUGHPUT",
                    "NUM_STREAMS": "4"
                }
        """
        self.device = device.upper()
        self.config = config or {}
        self.model_path = model_path
        self._det_model = None
        self._rec_model = None
        self._det_compiled = None
        self._rec_compiled = None
        self._model_loaded = False
        
        # 验证设备支持
        self._validate_device()
        
        # 尝试加载模型（支持懒加载，模型不存在时不阻塞初始化）
        try:
            self._load_models()
            self._model_loaded = True
            logger.info(f"OpenVINO OCR 引擎初始化完成，使用设备: {self.device}")
        except (FileNotFoundError, RuntimeError) as e:
            logger.warning(f"OCR 模型暂未加载，将使用懒加载模式: {e}")
            logger.info(f"OpenVINO OCR 引擎初始化完成（懒加载模式），使用设备: {self.device}")
    
    def _validate_device(self) -> None:
        """验证目标设备是否可用"""
        try:
            from openvino.runtime import Core  # type: ignore
            
            core = Core()
            available_devices = core.available_devices
            
            if self.device not in available_devices:
                logger.warning(
                    f"设备 {self.device} 不可用，可用设备: {available_devices}。"
                    f"自动回退到 CPU 模式。"
                )
                self.device = "CPU"
                
        except ImportError:
            logger.error("OpenVINO 未安装，请运行: pip install openvino")
            raise RuntimeError("OpenVINO 依赖缺失")
    
    def _load_models(self) -> None:
        """加载 OpenVINO IR 格式的 RapidOCR 检测和识别模型"""
        try:
            from openvino.runtime import Core  # type: ignore
            
            core = Core()
            
            # 模型路径解析
            if self.model_path is None:
                self.model_path = self._get_default_model_path()
            
            model_dir = Path(self.model_path)
            if not model_dir.exists():
                raise FileNotFoundError(f"模型目录不存在: {model_dir}")
            
            # 加载检测模型（Detection Model）
            det_xml = model_dir / "det.xml"
            det_bin = model_dir / "det.bin"
            
            if not det_xml.exists() or not det_bin.exists():
                raise FileNotFoundError(f"检测模型文件缺失: {det_xml} 或 {det_bin}")
            
            logger.info(f"正在加载 OCR 检测模型: {det_xml}")
            det_model = core.read_model(model=str(det_xml))
            self._det_compiled = core.compile_model(
                model=det_model,
                device_name=self.device,
                config=self.config
            )
            
            # 加载识别模型（Recognition Model）
            rec_xml = model_dir / "rec.xml"
            rec_bin = model_dir / "rec.bin"
            
            if not rec_xml.exists() or not rec_bin.exists():
                raise FileNotFoundError(f"识别模型文件缺失: {rec_xml} 或 {rec_bin}")
            
            logger.info(f"正在加载 OCR 识别模型: {rec_xml}")
            rec_model = core.read_model(model=str(rec_xml))
            self._rec_compiled = core.compile_model(
                model=rec_model,
                device_name=self.device,
                config=self.config
            )
            
            # 获取输入输出节点
            self._det_input = self._det_compiled.input(0)
            self._det_output = self._det_compiled.output(0)
            self._rec_input = self._rec_compiled.input(0)
            self._rec_output = self._rec_compiled.output(0)
            
            logger.info("OCR 模型加载成功")
            
        except Exception as e:
            logger.error(f"OCR 模型加载失败: {str(e)}")
            raise RuntimeError(f"OCR 模型加载失败: {str(e)}")
    
    def _get_default_model_path(self) -> str:
        """
        获取默认的 RapidOCR 模型路径
        
        Returns:
            模型目录的完整路径
        """
        cache_dir = Path.home() / ".cache" / "modelscope" / "LocalPrivacyScribe" / "rapidocr"
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        return str(cache_dir)
    
    def recognize(
        self,
        image_path: str,
        det_thresh: float = 0.3,
        det_box_thresh: float = 0.5,
        rec_thresh: float = 0.5
    ) -> OCRResult:
        """
        识别图片中的文本
        
        Args:
            image_path: 图片文件路径（.png, .jpg 等格式）
            det_thresh: 检测模型置信度阈值（0.0-1.0）
            det_box_thresh: 检测框阈值（0.0-1.0）
            rec_thresh: 识别模型置信度阈值（0.0-1.0）
            
        Returns:
            OCRResult: 包含文本、置信度、边界框和图片路径的识别结果
            
        Raises:
            FileNotFoundError: 图片文件不存在
            RuntimeError: 推理过程出错
        """
        # 验证输入文件
        image_file = Path(image_path)
        if not image_file.exists():
            raise FileNotFoundError(f"图片文件不存在: {image_path}")
        
        # 读取图片
        logger.info(f"正在读取图片: {image_path}")
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图片文件: {image_path}")
        
        # 文本检测
        logger.info(f"正在使用 {self.device} 进行 OCR 推理...")
        det_boxes = self._detect_text(image, det_thresh, det_box_thresh)
        
        # 文本识别
        texts = []
        confidences = []
        all_boxes = []
        
        for box in det_boxes:
            text, conf = self._recognize_text(image, box, rec_thresh)
            texts.append(text)
            confidences.append(conf)
            all_boxes.append(box.tolist())
        
        # 合并结果
        full_text = "\n".join(texts)
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        logger.info(f"OCR 完成，识别到 {len(texts)} 个文本区域")
        
        return OCRResult(
            text=full_text,
            confidence=avg_confidence,
            bounding_boxes=all_boxes,
            image_path=image_path
        )
    
    def _detect_text(
        self,
        image: np.ndarray,
        thresh: float,
        box_thresh: float
    ) -> np.ndarray:
        """
        检测图片中的文本区域
        
        Args:
            image: 输入图片（BGR 格式）
            thresh: 置信度阈值
            box_thresh: 检测框阈值
            
        Returns:
            检测到的文本框坐标数组
        """
        # 预处理
        input_tensor = self._preprocess_image(image)
        
        # 推理
        result = self._det_compiled([input_tensor])[self._det_output]
        
        # 后处理
        boxes = self._postprocess_detection(result, thresh, box_thresh)
        
        return boxes
    
    def _recognize_text(
        self,
        image: np.ndarray,
        box: np.ndarray,
        thresh: float
    ) -> tuple[str, float]:
        """
        识别单个文本框中的文本
        
        Args:
            image: 输入图片
            box: 文本框坐标
            thresh: 置信度阈值
            
        Returns:
            (文本, 置信度) 元组
        """
        # 裁剪文本框区域
        cropped = self._crop_text_region(image, box)
        
        # 预处理
        input_tensor = self._preprocess_image(cropped)
        
        # 推理
        result = self._rec_compiled([input_tensor])[self._rec_output]
        
        # 后处理
        text, confidence = self._postprocess_recognition(result, thresh)
        
        return text, confidence
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        图像预处理：调整大小、归一化、添加维度
        
        Args:
            image: 原始图像
            
        Returns:
            预处理后的张量
        """
        # 调整大小为模型输入尺寸（通常是 640x640 或类似）
        target_size = (640, 640)
        resized = cv2.resize(image, target_size)
        
        # 转换为 RGB（OpenCV 默认 BGR）
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # 归一化到 [0, 1]
        normalized = rgb.astype(np.float32) / 255.0
        
        # 转换为 CHW 格式并添加 batch 维度
        chw = np.transpose(normalized, (2, 0, 1))
        batched = np.expand_dims(chw, axis=0)
        
        return batched
    
    def _postprocess_detection(
        self,
        output: np.ndarray,
        thresh: float,
        box_thresh: float
    ) -> np.ndarray:
        """
        后处理检测结果：解码边界框
        
        Args:
            output: 模型原始输出
            thresh: 置信度阈值
            box_thresh: 检测框阈值
            
        Returns:
            过滤后的文本框坐标数组
        """
        # 简化处理：实际应根据模型的解码逻辑
        # 这里返回占位符
        return np.array([])
    
    def _postprocess_recognition(
        self,
        output: np.ndarray,
        thresh: float
    ) -> tuple[str, float]:
        """
        后处理识别结果：解码文本
        
        Args:
            output: 模型原始输出
            thresh: 置信度阈值
            
        Returns:
            (文本, 置信度) 元组
        """
        # 简化处理：实际应根据模型的 tokenizer 解码
        text = "[OCR 输出需要集成 RapidOCR Decoder]"
        confidence = 0.90
        
        return text, confidence
    
    def _crop_text_region(self, image: np.ndarray, box: np.ndarray) -> np.ndarray:
        """
        根据边界框裁剪文本区域
        
        Args:
            image: 原始图像
            box: 边界框坐标（4个点）
            
        Returns:
            裁剪后的图像区域
        """
        # 转换为整数坐标
        box = box.astype(np.int32)
        
        # 计算裁剪区域
        x_min = np.min(box[:, 0])
        y_min = np.min(box[:, 1])
        x_max = np.max(box[:, 0])
        y_max = np.max(box[:, 1])
        
        # 裁剪
        cropped = image[y_min:y_max, x_min:x_max]
        
        return cropped


class MultimodalExtractor:
    """
    多模态内容提取器
    
    统一封装 ASR 和 OCR 引擎，提供便捷的多模态内容提取接口。
    
    Example:
        >>> extractor = MultimodalExtractor(device="NPU")
        >>> audio_text = extractor.extract_audio("meeting.wav")
        >>> image_text = extractor.extract_image("whiteboard.png")
    """
    
    def __init__(
        self,
        asr_device: str = "CPU",
        ocr_device: str = "CPU",
        asr_config: Optional[Dict[str, Any]] = None,
        ocr_config: Optional[Dict[str, Any]] = None
    ):
        """
        初始化多模态提取器
        
        Args:
            asr_device: ASR 引擎使用的设备（"NPU", "GPU", "CPU"）
            ocr_device: OCR 引擎使用的设备（"NPU", "GPU", "CPU"）
            asr_config: ASR 引擎的 OpenVINO 配置
            ocr_config: OCR 引擎的 OpenVINO 配置
        """
        self.asr_engine = OpenVINOASREngine(
            device=asr_device,
            config=asr_config
        )
        self.ocr_engine = OpenVINOOCREngine(
            device=ocr_device,
            config=ocr_config
        )
        
        logger.info("多模态提取器初始化完成")
    
    def extract_audio(self, audio_path: str, **kwargs) -> ASRResult:
        """
        从音频文件中提取文本
        
        Args:
            audio_path: 音频文件路径
            **kwargs: 传递给 ASR 引擎的额外参数
            
        Returns:
            ASRResult: 音频识别结果
        """
        return self.asr_engine.transcribe(audio_path, **kwargs)
    
    def extract_image(self, image_path: str, **kwargs) -> OCRResult:
        """
        从图片文件中提取文本
        
        Args:
            image_path: 图片文件路径
            **kwargs: 传递给 OCR 引擎的额外参数
            
        Returns:
            OCRResult: 图片识别结果
        """
        return self.ocr_engine.recognize(image_path, **kwargs)
    
    def extract_all(
        self,
        audio_path: str,
        image_path: str
    ) -> tuple[ASRResult, OCRResult]:
        """
        同时提取音频和图片内容
        
        Args:
            audio_path: 音频文件路径
            image_path: 图片文件路径
            
        Returns:
            (ASRResult, OCRResult) 元组
        """
        audio_result = self.extract_audio(audio_path)
        image_result = self.extract_image(image_path)
        
        return audio_result, image_result