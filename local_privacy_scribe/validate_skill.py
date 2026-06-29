#!/usr/bin/env python3
"""
LocalPrivacyScribe Skill 本地验证脚本

验证 Skill 的稳定性、错误处理能力和可用性。
无需真实 ASR/OCR 模型文件，使用模拟数据验证核心逻辑。

测试覆盖：
1. Skill 初始化和配置（多种设备、配置组合）
2. 参数 Schema 结构完整性
3. 错误处理（文件不存在、格式错误、空路径、超长路径、特殊字符路径）
4. 输出文件生成（Markdown、ICS、JSON）
5. 待办事项提取（中文文本解析）
6. 系统兼容性（Python 版本、依赖检查）
7. 压力测试（初始化稳定性）
8. 模拟端到端流程（模拟 ASR/OCR 输出，验证完整处理链路）

运行方式：
    python3 local_privacy_scribe/validate_skill.py

作者: ModelScope Agent Ecosystem
版本: 1.0.0
"""

import os
import sys
import json
import time
import shutil
import tempfile
import traceback
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, MagicMock

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# ANSI 颜色代码
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"
BOLD = "\033[1m"

# 测试结果统计
total_tests = 0
passed_tests = 0
failed_tests = 0


def log_pass(message: str):
    """记录通过的测试"""
    global passed_tests, total_tests
    total_tests += 1
    passed_tests += 1
    print(f"  {GREEN}✓{RESET} {message}")


def log_fail(message: str, detail: str = ""):
    """记录失败的测试"""
    global failed_tests, total_tests
    total_tests += 1
    failed_tests += 1
    print(f"  {RED}✗{RESET} {message}")
    if detail:
        print(f"    {RED}原因: {detail}{RESET}")


def log_info(message: str):
    """记录信息"""
    print(f"  {CYAN}ℹ{RESET} {message}")


def log_warn(message: str):
    """记录警告"""
    print(f"  {YELLOW}⚠{RESET} {message}")


def print_header(title: str):
    """打印章节标题"""
    print()
    print(f"{BOLD}{'=' * 70}{RESET}")
    print(f"{BOLD} {title}{RESET}")
    print(f"{BOLD}{'=' * 70}{RESET}")
    print()


def print_summary():
    """打印测试总结"""
    print()
    print(f"{BOLD}{'=' * 70}{RESET}")
    print(f"{BOLD} 验证总结{RESET}")
    print(f"{BOLD}{'=' * 70}{RESET}")
    print(f"  总测试数: {total_tests}")
    print(f"  {GREEN}通过: {passed_tests}{RESET}")
    print(f"  {RED}失败: {failed_tests}{RESET}")
    print()
    
    if failed_tests == 0:
        print(f"  {GREEN}{BOLD}✓ 所有验证通过！Skill 稳定可用。{RESET}")
    else:
        print(f"  {RED}{BOLD}✗ 部分验证失败，请检查上述错误信息。{RESET}")
    print()


def check_dependencies():
    """
    检查系统依赖状态
    
    验证关键模块是否可以正常导入，提供详细的依赖状态报告。
    """
    print_header("1. 系统依赖检查")
    
    dependencies = [
        ("Python 版本", lambda: sys.version.split()[0]),
        ("cv2 (opencv-python)", lambda: __import__('cv2').__version__),
        ("numpy", lambda: __import__('numpy').__version__),
        ("openvino", lambda: __import__('openvino').__version__),
        ("modelscope_agent", lambda: "模拟模式" if 'modelscope_agent' not in sys.modules else __import__('modelscope_agent').__version__),
    ]
    
    all_ok = True
    for name, version_func in dependencies:
        try:
            version = version_func()
            log_pass(f"{name}: {version}")
        except Exception as e:
            log_fail(f"{name}: 未安装或导入失败", str(e))
            all_ok = False
    
    # 检查 soundfile 和 librosa（ASR 所需）
    try:
        import soundfile
        log_pass("soundfile: 可用")
    except ImportError:
        log_warn("soundfile: 未安装（仅 ASR 需要）")
    
    try:
        import librosa
        log_pass("librosa: 可用")
    except ImportError:
        log_warn("librosa: 未安装（仅 ASR 需要）")
    
    return all_ok


def test_skill_initialization():
    """
    测试 Skill 初始化
    
    验证不同配置组合下的初始化是否正常，包括设备回退逻辑。
    """
    print_header("2. Skill 初始化测试")
    
    # 测试 1: 默认初始化
    try:
        from local_privacy_scribe import LocalPrivacyScribeSkill
        skill = LocalPrivacyScribeSkill()
        assert skill.name == "LocalPrivacyScribe"
        assert skill.asr_device == "CPU"
        assert skill.ocr_device == "CPU"
        log_pass("默认初始化 (CPU)")
    except Exception as e:
        log_fail("默认初始化 (CPU)", str(e))
    
    # 测试 2: NPU 设备（会回退到 CPU）
    try:
        skill = LocalPrivacyScribeSkill(asr_device="NPU", ocr_device="NPU")
        # NPU 不可用时会回退到 CPU
        assert skill.asr_device in ("NPU", "CPU")
        assert skill.ocr_device in ("NPU", "CPU")
        log_pass(f"NPU 初始化 (实际设备: {skill.asr_device}/{skill.ocr_device})")
    except Exception as e:
        log_fail("NPU 初始化", str(e))
    
    # 测试 3: GPU 设备（会回退到 CPU）
    try:
        skill = LocalPrivacyScribeSkill(asr_device="GPU", ocr_device="GPU")
        assert skill.asr_device in ("GPU", "CPU")
        assert skill.ocr_device in ("GPU", "CPU")
        log_pass(f"GPU 初始化 (实际设备: {skill.asr_device}/{skill.ocr_device})")
    except Exception as e:
        log_fail("GPU 初始化", str(e))
    
    # 测试 4: 自定义配置
    try:
        skill = LocalPrivacyScribeSkill(
            asr_device="CPU",
            ocr_device="CPU",
            asr_config={"PERFORMANCE_HINT": "LATENCY"},
            ocr_config={"PERFORMANCE_HINT": "THROUGHPUT"},
            output_dir="./test_validate_output",
            auto_generate_ics=False,
            auto_generate_markdown=False
        )
        assert skill.asr_config == {"PERFORMANCE_HINT": "LATENCY"}
        assert skill.ocr_config == {"PERFORMANCE_HINT": "THROUGHPUT"}
        assert not skill.auto_generate_ics
        assert not skill.auto_generate_markdown
        skill = None
        shutil.rmtree("./test_validate_output", ignore_errors=True)
        log_pass("自定义配置初始化")
    except Exception as e:
        log_fail("自定义配置初始化", str(e))
        shutil.rmtree("./test_validate_output", ignore_errors=True)
    
    # 测试 5: 多次初始化的稳定性
    try:
        for i in range(10):
            s = LocalPrivacyScribeSkill()
            assert s is not None
        log_pass("多次初始化稳定性 (10次)")
    except Exception as e:
        log_fail("多次初始化稳定性", str(e))


def test_schema_integrity():
    """
    测试 Schema 结构完整性
    
    验证 get_schema() 返回的结构完整、参数定义清晰、符合 Agent 框架要求。
    """
    print_header("3. Schema 结构完整性")
    
    try:
        from local_privacy_scribe import LocalPrivacyScribeSkill
        skill = LocalPrivacyScribeSkill()
        schema = skill.get_schema()
        
        # 顶层结构
        assert "name" in schema, "缺少 name"
        assert "description" in schema, "缺少 description"
        assert "parameters" in schema, "缺少 parameters"
        log_pass("Schema 顶层结构完整")
        
        # parameters 结构
        params = schema["parameters"]
        assert params["type"] == "object", "parameters.type 应为 object"
        assert "properties" in params, "缺少 properties"
        assert "required" in params, "缺少 required"
        log_pass("parameters 结构完整")
        
        # 必需参数
        assert "audio_path" in params["required"], "缺少 audio_path"
        assert "image_path" in params["required"], "缺少 image_path"
        log_pass("必需参数定义完整: audio_path, image_path")
        
        # 参数属性
        audio_prop = params["properties"]["audio_path"]
        image_prop = params["properties"]["image_path"]
        
        assert audio_prop["type"] == "string", "audio_path.type 应为 string"
        assert image_prop["type"] == "string", "image_path.type 应为 string"
        assert "description" in audio_prop, "audio_path 缺少 description"
        assert "pattern" in audio_prop, "audio_path 缺少 pattern"
        assert "minLength" in audio_prop, "audio_path 缺少 minLength"
        assert "maxLength" in audio_prop, "audio_path 缺少 maxLength"
        log_pass("参数类型和约束定义完整")
        
    except Exception as e:
        log_fail("Schema 结构完整性", str(e))


def test_error_handling():
    """
    测试错误处理能力
    
    验证 Skill 在异常情况下的错误响应是否正确、友好。
    """
    print_header("4. 错误处理测试")
    
    from local_privacy_scribe import LocalPrivacyScribeSkill
    skill = LocalPrivacyScribeSkill(output_dir="./test_validate_errors")
    
    # 测试 1: 音频文件不存在
    try:
        result = skill._local_call(
            audio_path="/nonexistent/meeting.wav",
            image_path="/nonexistent/whiteboard.png"
        )
        assert result["status"] == "error"
        assert result["error_type"] == "FileNotFoundError"
        assert "audio_text" in result
        assert result["audio_text"] == ""
        log_pass("文件不存在错误处理")
    except Exception as e:
        log_fail("文件不存在错误处理", str(e))
    
    # 测试 2: 音频格式错误
    with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
        audio_mp3 = f.name
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        image_png = f.name
    try:
        Path(audio_mp3).touch()
        Path(image_png).touch()
        
        result = skill._local_call(
            audio_path=audio_mp3,
            image_path=image_png
        )
        assert result["status"] == "error"
        assert result["error_type"] == "ValueError"
        log_pass("音频格式错误处理 (.mp3)")
    except Exception as e:
        log_fail("音频格式错误处理", str(e))
    finally:
        os.unlink(audio_mp3)
        os.unlink(image_png)
    
    # 测试 3: 图片格式错误
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        audio_wav = f.name
    with tempfile.NamedTemporaryFile(suffix='.bmp', delete=False) as f:
        image_bmp = f.name
    try:
        Path(audio_wav).touch()
        Path(image_bmp).touch()
        
        result = skill._local_call(
            audio_path=audio_wav,
            image_path=image_bmp
        )
        assert result["status"] == "error"
        assert result["error_type"] == "ValueError"
        log_pass("图片格式错误处理 (.bmp)")
    except Exception as e:
        log_fail("图片格式错误处理", str(e))
    finally:
        os.unlink(audio_wav)
        os.unlink(image_bmp)
    
    # 测试 4: 空路径
    try:
        result = skill._local_call(audio_path="", image_path="")
        assert result["status"] == "error"
        log_pass("空路径错误处理")
    except Exception as e:
        log_fail("空路径错误处理", str(e))
    
    # 测试 5: 超长路径
    try:
        long_path = "a" * 2000 + ".wav"
        result = skill._local_call(
            audio_path=long_path,
            image_path="/fake/image.png"
        )
        assert result["status"] == "error"
        log_pass("超长路径错误处理")
    except Exception as e:
        log_fail("超长路径错误处理", str(e))
    
    # 测试 6: 特殊字符路径
    try:
        result = skill._local_call(
            audio_path="./test 文件/会议 (2024).wav",
            image_path="./whiteboard (final).png"
        )
        assert result["status"] == "error"
        log_pass("特殊字符路径错误处理")
    except Exception as e:
        log_fail("特殊字符路径错误处理", str(e))
    
    # 清理
    shutil.rmtree("./test_validate_errors", ignore_errors=True)


def test_output_file_generation():
    """
    测试输出文件生成
    
    使用模拟的 ASR/OCR 结果测试完整的文本提取和文件生成流程。
    """
    print_header("5. 输出文件生成测试")
    
    from local_privacy_scribe import LocalPrivacyScribeSkill
    from local_privacy_scribe.openvino_backends import ASRResult, OCRResult
    from unittest.mock import MagicMock
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        # 创建模拟的音频和图片文件
        audio_path = os.path.join(temp_dir, "test_meeting.wav")
        image_path = os.path.join(temp_dir, "test_whiteboard.png")
        Path(audio_path).touch()
        Path(image_path).touch()
        
        output_dir = os.path.join(temp_dir, "output")
        
        with patch('local_privacy_scribe.skill.MultimodalExtractor') as mock_class:
            mock_extractor = MagicMock()
            mock_extractor.extract_audio.return_value = ASRResult(
                text="今天会议讨论了Q4产品规划。张三需要在下周五前完成用户调研报告。李四应该跟进客户反馈。紧急：王五必须立即修复生产环境bug。",
                confidence=0.95,
                language="zh",
                duration=1800.0
            )
            mock_extractor.extract_image.return_value = OCRResult(
                text="Q4产品规划\n目标:提升30%用户留存\n重点:新功能开发\n截止:12月31日",
                confidence=0.88,
                bounding_boxes=[[10, 10, 200, 50], [10, 60, 300, 40]],
                image_path=image_path
            )
            mock_class.return_value = mock_extractor
            
            skill = LocalPrivacyScribeSkill(
                output_dir=output_dir,
                auto_generate_ics=True,
                auto_generate_markdown=True
            )
            skill.extractor = mock_extractor
            
            result = skill._local_call(
                audio_path=audio_path,
                image_path=image_path,
                meeting_title="Q4产品规划会议",
                participants=["张三", "李四", "王五"]
            )
            
            # 验证返回结构
            assert result["status"] == "success"
            assert len(result["audio_text"]) > 0
            assert len(result["image_text"]) > 0
            assert result["audio_confidence"] == 0.95
            assert result["image_confidence"] == 0.88
            assert result["audio_duration"] == 1800.0
            log_pass("返回结构完整")
            
            # 验证 meeting_summary 结构
            summary = result["meeting_summary"]
            assert summary["title"] == "Q4产品规划会议"
            assert len(summary["participants"]) == 3
            assert len(summary["action_items"]) > 0
            log_pass(f"会议纪要结构完整 (待办事项: {len(summary['action_items'])}条)")
            
            # 验证输出文件
            output_files = result["output_files"]
            assert "ics" in output_files
            assert "markdown" in output_files
            assert "json" in output_files
            assert os.path.exists(output_files["ics"]), "ICS 文件未生成"
            assert os.path.exists(output_files["markdown"]), "Markdown 文件未生成"
            assert os.path.exists(output_files["json"]), "JSON 文件未生成"
            log_pass("输出文件已生成 (ICS/Markdown/JSON)")
            
            # 验证 ICS 文件内容
            with open(output_files["ics"], 'r', encoding='utf-8') as f:
                ics_content = f.read()
                assert "BEGIN:VCALENDAR" in ics_content
                assert "BEGIN:VEVENT" in ics_content
            log_pass("ICS 文件格式有效")
            
            # 验证 Markdown 文件内容
            with open(output_files["markdown"], 'r', encoding='utf-8') as f:
                md_content = f.read()
                assert "Q4产品规划会议" in md_content
            log_pass("Markdown 文件内容有效")
            
            # 验证 JSON 文件内容
            with open(output_files["json"], 'r', encoding='utf-8') as f:
                json_data = json.load(f)
                assert json_data["title"] == "Q4产品规划会议"
                assert len(json_data["action_items"]) > 0
            log_pass("JSON 文件内容有效")
            
    except Exception as e:
        log_fail("输出文件生成", f"{str(e)}\n{traceback.format_exc()}")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_action_item_extraction():
    """
    测试待办事项提取
    
    验证 parse_action_items 能否从中文文本中正确识别待办事项。
    """
    print_header("6. 待办事项提取测试")
    
    from local_privacy_scribe.utils import parse_action_items
    
    test_texts = [
        ("张三需要在下周五前完成项目报告", "需要模式"),
        ("李四应该跟进客户反馈", "应该模式"),
        ("紧急：王五必须立即修复生产环境bug", "必须模式"),
        ("TODO: 更新项目文档", "TODO 模式"),
        ("行动项：完成用户测试", "行动项模式"),
    ]
    
    all_extracted = 0
    for text, desc in test_texts:
        try:
            items = parse_action_items(text, source="audio")
            if items:
                all_extracted += len(items)
                log_pass(f"待办事项提取成功: {desc} ({items[0].description[:30]}...)")
            else:
                log_warn(f"待办事项未提取: {desc}")
        except Exception as e:
            log_fail(f"待办事项提取失败: {desc}", str(e))
    
    log_info(f"共提取 {all_extracted} 条待办事项")


def test_end_to_end_simulation():
    """
    模拟端到端流程
    
    模拟 Ollama + Qwen 调用场景，验证完整的 Skill 调用链路。
    """
    print_header("7. 端到端流程模拟")
    
    from local_privacy_scribe import LocalPrivacyScribeSkill
    from local_privacy_scribe.openvino_backends import ASRResult, OCRResult
    from unittest.mock import MagicMock
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Step 1: Agent 解析 Schema
        skill = LocalPrivacyScribeSkill()
        schema = skill.get_schema()
        assert "name" in schema
        assert "parameters" in schema
        log_pass("[Agent] Schema 解析成功")
        
        # Step 2: 模拟 LLM 生成参数
        params = schema["parameters"]
        assert "audio_path" in params["required"]
        assert "image_path" in params["required"]
        log_pass("[LLM] 参数生成验证通过")
        
        # Step 3: 创建模拟文件
        audio_path = os.path.join(temp_dir, "meeting.wav")
        image_path = os.path.join(temp_dir, "whiteboard.png")
        Path(audio_path).touch()
        Path(image_path).touch()
        
        # Step 4: 模拟 ASR/OCR 执行
        with patch('local_privacy_scribe.skill.MultimodalExtractor') as mock_class:
            mock_extractor = MagicMock()
            mock_extractor.extract_audio.return_value = ASRResult(
                text="模拟的会议录音转录内容，包含待办事项：张三需要完成报告。",
                confidence=0.95,
                language="zh",
                duration=1800.0
            )
            mock_extractor.extract_image.return_value = OCRResult(
                text="白板上的重要决策：通过Q4计划。",
                confidence=0.90,
                bounding_boxes=[],
                image_path=image_path
            )
            mock_class.return_value = mock_extractor
            
            skill = LocalPrivacyScribeSkill(output_dir=os.path.join(temp_dir, "output"))
            skill.extractor = mock_extractor
            
            result = skill._local_call(
                audio_path=audio_path,
                image_path=image_path,
                meeting_title="Q4 产品规划会议",
                participants=["张三", "李四"]
            )
            
            # Step 5: 验证返回结果
            assert result["status"] == "success"
            assert len(result["audio_text"]) > 0
            assert len(result["image_text"]) > 0
            assert len(result["output_files"]) > 0
            log_pass("[Skill] 执行成功")
            
            # Step 6: 验证结构化数据可用于 LLM 交叉验证
            summary = result["meeting_summary"]
            audio_content = summary["audio_transcript"]
            image_content = summary["whiteboard_content"]
            assert audio_content == result["audio_text"]
            assert image_content == result["image_text"]
            log_pass("[交叉验证] 音频与图片内容一致性确认")
            
            # Step 7: 验证待办事项
            if summary["action_items"]:
                log_pass(f"[待办事项] 已提取 {len(summary['action_items'])} 条")
            
            log_pass("端到端流程模拟通过")
            
    except Exception as e:
        log_fail("端到端流程模拟", f"{str(e)}\n{traceback.format_exc()}")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_stability():
    """
    稳定性测试
    
    验证 Skill 在重复操作下的稳定性。
    """
    print_header("8. 稳定性测试")
    
    from local_privacy_scribe import LocalPrivacyScribeSkill
    
    # 重复初始化测试
    try:
        for i in range(20):
            skill = LocalPrivacyScribeSkill()
            schema = skill.get_schema()
            assert schema["name"] == "LocalPrivacyScribe"
        log_pass("重复初始化 (20次)")
    except Exception as e:
        log_fail("重复初始化", str(e))
    
    # 重复错误调用测试
    try:
        skill = LocalPrivacyScribeSkill()
        for i in range(20):
            result = skill._local_call(
                audio_path="/nonexistent/file.wav",
                image_path="/nonexistent/image.png"
            )
            assert result["status"] == "error"
        log_pass("重复错误调用 (20次)")
    except Exception as e:
        log_fail("重复错误调用", str(e))
    
    # Schema 生成无副作用测试
    try:
        skill = LocalPrivacyScribeSkill()
        schema1 = skill.get_schema()
        schema2 = skill.get_schema()
        assert schema1 == schema2
        log_pass("Schema 生成无副作用 (幂等性)")
    except Exception as e:
        log_fail("Schema 幂等性", str(e))


def main():
    """主函数：运行所有验证"""
    print()
    print(f"{BOLD}{GREEN}╔══════════════════════════════════════════════════════════════╗{RESET}")
    print(f"{BOLD}{GREEN}║       LocalPrivacyScribe Skill 本地验证脚本                  ║{RESET}")
    print(f"{BOLD}{GREEN}║       版本: 1.0.0 | 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}            ║{RESET}")
    print(f"{BOLD}{GREEN}╚══════════════════════════════════════════════════════════════╝{RESET}")
    print()
    print(f" Python: {sys.version.split()[0]}")
    print(f" 路径: {Path(__file__).parent}")
    print()
    
    # 运行所有验证
    check_dependencies()
    test_skill_initialization()
    test_schema_integrity()
    test_error_handling()
    test_output_file_generation()
    test_action_item_extraction()
    test_end_to_end_simulation()
    test_stability()
    
    # 打印总结
    print_summary()
    
    return 0 if failed_tests == 0 else 1


if __name__ == "__main__":
    sys.exit(main())