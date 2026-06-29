#!/usr/bin/env python3
"""
LocalPrivacyScribe Skill 基准测试脚本

本脚本用于赛事模拟 Ollama + Qwen3.6-35B-A3B 调用的本地自动化测试。
验证 Skill 的各个功能模块，确保符合 ModelScope Skills 中心规范。

测试覆盖：
1. Skill 初始化和配置
2. 参数 Schema 验证
3. ASR 音频转录（模拟）
4. OCR 图片识别（模拟）
5. 多模态内容提取
6. 纪要生成和文件导出
7. 错误处理
8. 设备兼容性

作者: ModelScope Agent Ecosystem
版本: 1.0.0
"""

import os
import sys
import json
import time
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from typing import Dict, Any

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 导入被测模块
from local_privacy_scribe import LocalPrivacyScribeSkill
from local_privacy_scribe.openvino_backends import (
    OpenVINOASREngine,
    OpenVINOOCREngine,
    MultimodalExtractor,
    ASRResult,
    OCRResult
)
from local_privacy_scribe.utils import (
    MeetingSummary,
    ActionItem,
    generate_ics_file,
    generate_markdown_summary,
    parse_action_items,
    export_to_json
)


class TestLocalPrivacyScribeSkillInit(unittest.TestCase):
    """测试 Skill 初始化"""
    
    def test_default_initialization(self):
        """测试默认初始化（CPU 设备）"""
        skill = LocalPrivacyScribeSkill()
        
        self.assertEqual(skill.name, "LocalPrivacyScribe")
        self.assertEqual(skill.asr_device, "CPU")
        self.assertEqual(skill.ocr_device, "CPU")
        self.assertTrue(skill.auto_generate_ics)
        self.assertTrue(skill.auto_generate_markdown)
        self.assertIsNotNone(skill.extractor)
    
    def test_npu_initialization(self):
        """测试 NPU 设备初始化"""
        skill = LocalPrivacyScribeSkill(
            asr_device="NPU",
            ocr_device="NPU",
            output_dir="./test_output_npu"
        )
        
        self.assertEqual(skill.asr_device, "NPU")
        self.assertEqual(skill.ocr_device, "NPU")
    
    def test_gpu_initialization(self):
        """测试 GPU 设备初始化"""
        skill = LocalPrivacyScribeSkill(
            asr_device="GPU",
            ocr_device="GPU",
            output_dir="./test_output_gpu"
        )
        
        self.assertEqual(skill.asr_device, "GPU")
        self.assertEqual(skill.ocr_device, "GPU")
    
    def test_custom_config_initialization(self):
        """测试自定义配置初始化"""
        asr_config = {"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": "1"}
        ocr_config = {"PERFORMANCE_HINT": "THROUGHPUT", "NUM_STREAMS": "4"}
        
        skill = LocalPrivacyScribeSkill(
            asr_device="CPU",
            ocr_device="CPU",
            asr_config=asr_config,
            ocr_config=ocr_config,
            output_dir="./test_output_custom",
            auto_generate_ics=False,
            auto_generate_markdown=False
        )
        
        self.assertEqual(skill.asr_config, asr_config)
        self.assertEqual(skill.ocr_config, ocr_config)
        self.assertFalse(skill.auto_generate_ics)
        self.assertFalse(skill.auto_generate_markdown)
    
    def test_output_dir_creation(self):
        """测试输出目录自动创建"""
        test_dir = "./test_output_auto_created"
        
        # 清理可能存在的目录
        import shutil
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
        
        skill = LocalPrivacyScribeSkill(output_dir=test_dir)
        self.assertTrue(os.path.exists(test_dir))
        
        # 清理
        shutil.rmtree(test_dir)


class TestSkillSchema(unittest.TestCase):
    """测试 Skill Schema 定义"""
    
    def setUp(self):
        """测试前准备"""
        self.skill = LocalPrivacyScribeSkill()
    
    def test_schema_structure(self):
        """测试 Schema 结构完整性"""
        schema = self.skill.get_schema()
        
        self.assertIn("name", schema)
        self.assertIn("description", schema)
        self.assertIn("parameters", schema)
    
    def test_schema_parameters(self):
        """测试参数 Schema 定义"""
        schema = self.skill.get_schema()
        params = schema["parameters"]
        
        self.assertEqual(params["type"], "object")
        self.assertIn("properties", params)
        self.assertIn("required", params)
        
        # 验证必需参数
        self.assertIn("audio_path", params["required"])
        self.assertIn("image_path", params["required"])
    
    def test_audio_path_parameter(self):
        """测试 audio_path 参数定义"""
        schema = self.skill.get_schema()
        audio_param = schema["parameters"]["properties"]["audio_path"]
        
        self.assertEqual(audio_param["type"], "string")
        self.assertIn("pattern", audio_param)
        self.assertTrue(audio_param["pattern"].endswith("\\.wav$"))
        self.assertIn("minLength", audio_param)
        self.assertIn("maxLength", audio_param)
        self.assertGreater(audio_param["maxLength"], 0)
    
    def test_image_path_parameter(self):
        """测试 image_path 参数定义"""
        schema = self.skill.get_schema()
        image_param = schema["parameters"]["properties"]["image_path"]
        
        self.assertEqual(image_param["type"], "string")
        self.assertIn("pattern", image_param)
        self.assertIn("png", image_param["pattern"])
        self.assertIn("jpg", image_param["pattern"])
        self.assertIn("minLength", image_param)
        self.assertIn("maxLength", image_param)


class TestSkillLocalCall(unittest.TestCase):
    """测试 Skill _local_call 方法"""
    
    def setUp(self):
        """测试前准备"""
        self.skill = LocalPrivacyScribeSkill(output_dir="./test_output")
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """测试后清理"""
        import shutil
        if os.path.exists("./test_output"):
            shutil.rmtree("./test_output")
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_local_call_missing_audio_file(self):
        """测试缺失音频文件的错误处理"""
        fake_image = os.path.join(self.temp_dir, "fake.png")
        
        # 创建一个假图片文件
        Path(fake_image).touch()
        
        result = self.skill._local_call(
            audio_path="/nonexistent/meeting.wav",
            image_path=fake_image
        )
        
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error_type"], "FileNotFoundError")
        self.assertIn("audio_text", result)
        self.assertIn("image_text", result)
    
    def test_local_call_missing_image_file(self):
        """测试缺失图片文件的错误处理"""
        fake_audio = os.path.join(self.temp_dir, "fake.wav")
        
        # 创建一个假音频文件
        Path(fake_audio).touch()
        
        result = self.skill._local_call(
            audio_path=fake_audio,
            image_path="/nonexistent/whiteboard.png"
        )
        
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error_type"], "FileNotFoundError")
    
    def test_local_call_invalid_audio_format(self):
        """测试无效音频格式"""
        fake_audio = os.path.join(self.temp_dir, "fake.mp3")
        fake_image = os.path.join(self.temp_dir, "fake.png")
        
        Path(fake_audio).touch()
        Path(fake_image).touch()
        
        result = self.skill._local_call(
            audio_path=fake_audio,
            image_path=fake_image
        )
        
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error_type"], "ValueError")
    
    def test_local_call_invalid_image_format(self):
        """测试无效图片格式"""
        fake_audio = os.path.join(self.temp_dir, "fake.wav")
        fake_image = os.path.join(self.temp_dir, "fake.bmp")
        
        Path(fake_audio).touch()
        Path(fake_image).touch()
        
        result = self.skill._local_call(
            audio_path=fake_audio,
            image_path=fake_image
        )
        
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error_type"], "ValueError")


class TestOpenVINOBackends(unittest.TestCase):
    """测试 OpenVINO 后端引擎"""
    
    def test_asr_engine_device_validation(self):
        """测试 ASR 引擎设备验证"""
        # 测试设备名称标准化
        with patch('openvino.runtime.Core') as mock_core:
            mock_instance = MagicMock()
            mock_instance.available_devices = ["CPU", "GPU"]
            mock_core.return_value = mock_instance
            
            # 测试小写设备名转换
            engine = OpenVINOASREngine(device="cpu")
            self.assertEqual(engine.device, "CPU")
    
    def test_ocr_engine_device_validation(self):
        """测试 OCR 引擎设备验证"""
        with patch('openvino.runtime.Core') as mock_core:
            mock_instance = MagicMock()
            mock_instance.available_devices = ["CPU", "NPU"]
            mock_core.return_value = mock_instance
            
            engine = OpenVINOOCREngine(device="npu")
            self.assertEqual(engine.device, "NPU")
    
    def test_asr_result_dataclass(self):
        """测试 ASR 结果数据结构"""
        result = ASRResult(
            text="测试文本",
            confidence=0.95,
            language="zh",
            duration=123.4
        )
        
        self.assertEqual(result.text, "测试文本")
        self.assertEqual(result.confidence, 0.95)
        self.assertEqual(result.language, "zh")
        self.assertEqual(result.duration, 123.4)
    
    def test_ocr_result_dataclass(self):
        """测试 OCR 结果数据结构"""
        result = OCRResult(
            text="识别文本",
            confidence=0.90,
            bounding_boxes=[[0, 0, 100, 50]],
            image_path="/path/to/image.png"
        )
        
        self.assertEqual(result.text, "识别文本")
        self.assertEqual(result.confidence, 0.90)
        self.assertEqual(len(result.bounding_boxes), 1)
        self.assertEqual(result.image_path, "/path/to/image.png")


class TestUtils(unittest.TestCase):
    """测试工具函数"""
    
    def setUp(self):
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_summary = MeetingSummary(
            title="测试会议",
            date="2024-12-25T14:30:00",
            duration=45.5,
            participants=["张三", "李四"],
            key_points=["要点1", "要点2"],
            decisions=["决策1"],
            action_items=[
                ActionItem(
                    id="test001",
                    description="完成报告",
                    assignee="张三",
                    deadline="2024-12-31",
                    priority="high",
                    status="pending",
                    source="audio"
                )
            ],
            audio_transcript="音频转录内容",
            whiteboard_content="白板内容",
            metadata={"test": "metadata"}
        )
    
    def tearDown(self):
        """测试后清理"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_generate_ics_file(self):
        """测试 ICS 文件生成"""
        ics_path = os.path.join(self.temp_dir, "test.ics")
        
        result_path = generate_ics_file(
            self.test_summary,
            ics_path,
            organizer="Test Organizer"
        )
        
        self.assertTrue(os.path.exists(ics_path))
        self.assertTrue(ics_path.endswith(".ics"))
        
        # 验证 ICS 内容
        with open(ics_path, 'r', encoding='utf-8') as f:
            content = f.read()
            self.assertIn("BEGIN:VCALENDAR", content)
            self.assertIn("END:VCALENDAR", content)
            self.assertIn("BEGIN:VEVENT", content)
            self.assertIn("测试会议", content)
    
    def test_generate_markdown_summary_default(self):
        """测试 Markdown 纪要生成（默认模板）"""
        md_path = os.path.join(self.temp_dir, "test.md")
        
        result_path = generate_markdown_summary(
            self.test_summary,
            md_path,
            template="default"
        )
        
        self.assertTrue(os.path.exists(md_path))
        
        with open(md_path, 'r', encoding='utf-8') as f:
            content = f.read()
            self.assertIn("# 测试会议", content)
            self.assertIn("## 基本信息", content)
            self.assertIn("## 关键要点", content)
            self.assertIn("## 待办事项", content)
    
    def test_generate_markdown_summary_minimal(self):
        """测试 Markdown 纪要生成（极简模板）"""
        md_path = os.path.join(self.temp_dir, "test_minimal.md")
        
        result_path = generate_markdown_summary(
            self.test_summary,
            md_path,
            template="minimal"
        )
        
        self.assertTrue(os.path.exists(md_path))
        
        with open(md_path, 'r', encoding='utf-8') as f:
            content = f.read()
            self.assertIn("# 测试会议", content)
            self.assertIn("## 要点", content)
    
    def test_generate_markdown_summary_detailed(self):
        """测试 Markdown 纪要生成（详细模板）"""
        md_path = os.path.join(self.temp_dir, "test_detailed.md")
        
        result_path = generate_markdown_summary(
            self.test_summary,
            md_path,
            template="detailed"
        )
        
        self.assertTrue(os.path.exists(md_path))
        
        with open(md_path, 'r', encoding='utf-8') as f:
            content = f.read()
            self.assertIn("# 测试会议", content)
            self.assertIn("## 元数据", content)
            self.assertIn("```json", content)
    
    def test_parse_action_items(self):
        """测试待办事项提取"""
        text = """
        张三需要在下周五前完成项目报告。
        李四应该跟进客户反馈。
        紧急：王五必须立即修复生产环境 bug。
        TODO: 更新项目文档
        """
        
        items = parse_action_items(text, source="audio")
        
        self.assertGreater(len(items), 0)
        
        # 验证提取的待办事项
        descriptions = [item.description for item in items]
        self.assertTrue(any("报告" in desc for desc in descriptions))
    
    def test_export_to_json(self):
        """测试 JSON 导出"""
        json_path = os.path.join(self.temp_dir, "test.json")
        
        result_path = export_to_json(self.test_summary, json_path)
        
        self.assertTrue(os.path.exists(json_path))
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.assertEqual(data["title"], "测试会议")
            self.assertEqual(data["duration"], 45.5)
            self.assertIn("action_items", data)
            self.assertEqual(len(data["action_items"]), 1)


class TestOllamaQwenIntegration(unittest.TestCase):
    """测试 Ollama + Qwen3.6-35B-A3B 集成（模拟）"""
    
    def setUp(self):
        """测试前准备"""
        self.skill = LocalPrivacyScribeSkill(output_dir="./test_ollama_output")
    
    def tearDown(self):
        """测试后清理"""
        import shutil
        if os.path.exists("./test_ollama_output"):
            shutil.rmtree("./test_ollama_output")
    
    def test_skill_schema_for_llm_parsing(self):
        """测试 Schema 对 LLM 的友好性"""
        schema = self.skill.get_schema()
        
        # 验证 Schema 结构清晰，易于 LLM 解析
        self.assertIn("name", schema)
        self.assertIn("description", schema)
        self.assertIn("parameters", schema)
        self.assertIn("type", schema["parameters"])
        self.assertIn("properties", schema["parameters"])
        
        # 验证每个参数都有明确的类型和描述
        for param_name, param_def in schema["parameters"]["properties"].items():
            self.assertIn("type", param_def, f"参数 {param_name} 缺少 type 定义")
            self.assertIn("description", param_def, f"参数 {param_name} 缺少 description 定义")
    
    def test_docstring_completeness(self):
        """测试 Docstring 完整性（供 LLM 解析）"""
        # 检查 _local_call 方法的 Docstring
        docstring = self.skill._local_call.__doc__
        
        self.assertIsNotNone(docstring)
        self.assertIn("Args:", docstring)
        self.assertIn("Returns:", docstring)
        self.assertIn("audio_path", docstring)
        self.assertIn("image_path", docstring)
        self.assertIn("Raises:", docstring)
        self.assertIn("Example:", docstring)
    
    def test_parameter_types_clarity(self):
        """测试参数类型定义的清晰性"""
        schema = self.skill.get_schema()
        audio_param = schema["parameters"]["properties"]["audio_path"]
        image_param = schema["parameters"]["properties"]["image_path"]
        
        # 验证类型定义
        self.assertEqual(audio_param["type"], "string")
        self.assertEqual(image_param["type"], "string")
        
        # 验证格式约束
        self.assertIn("pattern", audio_param)
        self.assertIn("pattern", image_param)
        
        # 验证长度约束
        self.assertIn("minLength", audio_param)
        self.assertIn("maxLength", audio_param)
    
    @patch('local_privacy_scribe.skill.MultimodalExtractor')
    def test_mocked_multimodal_extraction(self, mock_extractor_class):
        """测试模拟的多模态提取"""
        # 创建模拟的 ASR 和 OCR 结果
        mock_asr_result = ASRResult(
            text="会议录音转录文本",
            confidence=0.95,
            language="zh",
            duration=120.5
        )
        
        mock_ocr_result = OCRResult(
            text="白板识别文本",
            confidence=0.90,
            bounding_boxes=[[0, 0, 100, 50]],
            image_path="/fake/path.png"
        )
        
        # 配置模拟对象
        mock_extractor = MagicMock()
        mock_extractor.extract_audio.return_value = mock_asr_result
        mock_extractor.extract_image.return_value = mock_ocr_result
        mock_extractor_class.return_value = mock_extractor
        
        # 创建临时文件
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as audio_file:
            audio_path = audio_file.name
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as image_file:
            image_path = image_file.name
        
        try:
            # 执行测试
            skill = LocalPrivacyScribeSkill(output_dir="./test_mock_output")
            skill.extractor = mock_extractor
            
            result = skill._local_call(
                audio_path=audio_path,
                image_path=image_path,
                meeting_title="模拟测试会议"
            )
            
            # 验证结果
            self.assertEqual(result["status"], "success")
            self.assertEqual(result["audio_text"], "会议录音转录文本")
            self.assertEqual(result["image_text"], "白板识别文本")
            self.assertEqual(result["audio_confidence"], 0.95)
            self.assertEqual(result["image_confidence"], 0.90)
            self.assertIn("meeting_summary", result)
            self.assertIn("output_files", result)
            self.assertIn("metadata", result)
            
        finally:
            # 清理临时文件
            if os.path.exists(audio_path):
                os.unlink(audio_path)
            if os.path.exists(image_path):
                os.unlink(image_path)
            import shutil
            if os.path.exists("./test_mock_output"):
                shutil.rmtree("./test_mock_output")


class TestPerformanceBenchmark(unittest.TestCase):
    """性能基准测试"""
    
    def setUp(self):
        """测试前准备"""
        self.skill = LocalPrivacyScribeSkill(output_dir="./test_benchmark_output")
    
    def tearDown(self):
        """测试后清理"""
        import shutil
        if os.path.exists("./test_benchmark_output"):
            shutil.rmtree("./test_benchmark_output")
    
    def test_skill_initialization_time(self):
        """测试 Skill 初始化时间"""
        start_time = time.time()
        
        skill = LocalPrivacyScribeSkill(
            asr_device="CPU",
            ocr_device="CPU"
        )
        
        init_time = time.time() - start_time
        
        # 初始化时间应该在合理范围内（< 5 秒）
        self.assertLess(init_time, 5.0)
        print(f"\n✓ Skill 初始化耗时: {init_time:.3f} 秒")
    
    def test_schema_generation_time(self):
        """测试 Schema 生成时间"""
        start_time = time.time()
        
        schema = self.skill.get_schema()
        
        schema_time = time.time() - start_time
        
        # Schema 生成应该非常快（< 0.1 秒）
        self.assertLess(schema_time, 0.1)
        print(f"\n✓ Schema 生成耗时: {schema_time:.3f} 秒")
    
    def test_error_response_time(self):
        """测试错误响应时间"""
        start_time = time.time()
        
        result = self.skill._local_call(
            audio_path="/nonexistent/file.wav",
            image_path="/nonexistent/image.png"
        )
        
        response_time = time.time() - start_time
        
        # 错误响应应该快速（< 1 秒）
        self.assertLess(response_time, 1.0)
        self.assertEqual(result["status"], "error")
        print(f"\n✓ 错误响应耗时: {response_time:.3f} 秒")


class TestEdgeCases(unittest.TestCase):
    """边界情况测试"""
    
    def setUp(self):
        """测试前准备"""
        self.skill = LocalPrivacyScribeSkill(output_dir="./test_edge_cases")
    
    def tearDown(self):
        """测试后清理"""
        import shutil
        if os.path.exists("./test_edge_cases"):
            shutil.rmtree("./test_edge_cases")
    
    def test_empty_file_paths(self):
        """测试空路径"""
        result = self.skill._local_call(
            audio_path="",
            image_path=""
        )
        
        self.assertEqual(result["status"], "error")
    
    def test_very_long_path(self):
        """测试超长路径"""
        long_path = "a" * 2000 + ".wav"
        result = self.skill._local_call(
            audio_path=long_path,
            image_path="/fake/image.png"
        )
        
        self.assertEqual(result["status"], "error")
    
    def test_special_characters_in_path(self):
        """测试路径中的特殊字符"""
        special_path = "./test 文件/会议录音 (2024).wav"
        result = self.skill._local_call(
            audio_path=special_path,
            image_path="/fake/image.png"
        )
        
        self.assertEqual(result["status"], "error")


def run_benchmark_suite():
    """运行完整的基准测试套件"""
    print("=" * 70)
    print("LocalPrivacyScribe Skill 基准测试套件")
    print("=" * 70)
    print()
    
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加测试类
    suite.addTests(loader.loadTestsFromTestCase(TestLocalPrivacyScribeSkillInit))
    suite.addTests(loader.loadTestsFromTestCase(TestSkillSchema))
    suite.addTests(loader.loadTestsFromTestCase(TestSkillLocalCall))
    suite.addTests(loader.loadTestsFromTestCase(TestOpenVINOBackends))
    suite.addTests(loader.loadTestsFromTestCase(TestUtils))
    suite.addTests(loader.loadTestsFromTestCase(TestOllamaQwenIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestPerformanceBenchmark))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 输出总结
    print()
    print("=" * 70)
    print("测试总结")
    print("=" * 70)
    print(f"总测试数: {result.testsRun}")
    print(f"成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
    print()
    
    if result.wasSuccessful():
        print("✓ 所有测试通过！Skill 符合 ModelScope Skills 中心规范。")
        return 0
    else:
        print("✗ 部分测试失败，请检查上述错误信息。")
        return 1


def simulate_ollama_qwen_call():
    """模拟 Ollama + Qwen3.6-35B-A3B 调用"""
    print()
    print("=" * 70)
    print("模拟 Ollama + Qwen3.6-35B-A3B 调用场景")
    print("=" * 70)
    print()
    
    # 模拟 Agent 框架解析 Skill Schema
    print("[步骤 1] Agent 框架解析 Skill Schema...")
    skill = LocalPrivacyScribeSkill()
    schema = skill.get_schema()
    
    print(f"  ✓ Skill 名称: {schema['name']}")
    print(f"  ✓ 必需参数: {list(schema['parameters']['required'])}")
    print(f"  ✓ 可选参数: {list(schema['parameters']['properties'].keys())}")
    print()
    
    # 模拟 LLM 生成 Tool Call
    print("[步骤 2] Qwen3.6-35B 生成 Tool Call...")
    simulated_tool_call = {
        "name": "LocalPrivacyScribe",
        "arguments": {
            "audio_path": "./meeting.wav",
            "image_path": "./whiteboard.png",
            "meeting_title": "Q4 产品规划会议",
            "participants": ["张三", "李四", "王五"],
            "language": "zh"
        }
    }
    
    print(f"  ✓ 工具名称: {simulated_tool_call['name']}")
    print(f"  ✓ 调用参数: {json.dumps(simulated_tool_call['arguments'], ensure_ascii=False, indent=4)}")
    print()
    
    # 验证参数符合 Schema
    print("[步骤 3] 验证参数符合 Schema...")
    args = simulated_tool_call['arguments']
    
    # 验证必需参数
    assert 'audio_path' in args, "缺少 audio_path 参数"
    assert 'image_path' in args, "缺少 image_path 参数"
    
    # 验证参数类型
    assert isinstance(args['audio_path'], str), "audio_path 必须是字符串"
    assert isinstance(args['image_path'], str), "image_path 必须是字符串"
    
    # 验证文件格式
    assert args['audio_path'].endswith('.wav'), "audio_path 必须以 .wav 结尾"
    assert args['image_path'].endswith(('.png', '.jpg', '.jpeg')), "image_path 必须是图片格式"
    
    print("  ✓ 参数验证通过")
    print()
    
    # 模拟执行 Skill
    print("[步骤 4] 执行 Skill（使用模拟数据）...")
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as audio_file:
        audio_path = audio_file.name
    
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as image_file:
        image_path = image_file.name
    
    try:
        # 使用 patch 模拟 ASR/OCR
        with patch('local_privacy_scribe.skill.MultimodalExtractor') as mock_class:
            mock_extractor = MagicMock()
            mock_extractor.extract_audio.return_value = ASRResult(
                text="模拟的会议录音转录内容",
                confidence=0.95,
                language="zh",
                duration=1800.0
            )
            mock_extractor.extract_image.return_value = OCRResult(
                text="模拟的白板识别内容",
                confidence=0.90,
                bounding_boxes=[],
                image_path=image_path
            )
            mock_class.return_value = mock_extractor
            
            skill = LocalPrivacyScribeSkill(output_dir="./test_ollama_simulation")
            skill.extractor = mock_extractor
            
            result = skill._local_call(**args)
            
            print(f"  ✓ 执行状态: {result['status']}")
            print(f"  ✓ 音频文本长度: {len(result['audio_text'])} 字符")
            print(f"  ✓ 图片文本长度: {len(result['image_text'])} 字符")
            print(f"  ✓ 输出文件: {result['output_files']}")
            print()
            
            # 验证返回结构
            print("[步骤 5] 验证返回结构...")
            required_fields = [
                'status', 'audio_text', 'image_text', 'audio_confidence',
                'image_confidence', 'audio_duration', 'meeting_summary',
                'output_files', 'metadata'
            ]
            
            for field in required_fields:
                assert field in result, f"缺少必需字段: {field}"
                print(f"  ✓ {field}: {type(result[field]).__name__}")
            
            print()
            print("=" * 70)
            print("✓ Ollama + Qwen3.6-35B-A3B 集成测试通过！")
            print("=" * 70)
            
    finally:
        # 清理
        if os.path.exists(audio_path):
            os.unlink(audio_path)
        if os.path.exists(image_path):
            os.unlink(image_path)
        import shutil
        if os.path.exists("./test_ollama_simulation"):
            shutil.rmtree("./test_ollama_simulation")


if __name__ == "__main__":
    # 运行基准测试套件
    exit_code = run_benchmark_suite()
    
    # 运行 Ollama 集成模拟
    simulate_ollama_qwen_call()
    
    sys.exit(exit_code)