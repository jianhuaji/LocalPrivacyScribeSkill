"""
工具函数模块

本模块提供会议纪要生成、ICS 日程文件导出、Markdown 格式化等辅助功能。
所有功能均在本地执行，无需云端 API 调用。

功能列表：
1. generate_ics_file: 生成标准 .ics 日程文件（兼容 Outlook、Google Calendar、Apple Calendar）
2. generate_markdown_summary: 生成 Markdown 格式的会议纪要
3. parse_action_items: 从会议文本中提取待办事项
4. validate_datetime: 验证和格式化日期时间

作者: ModelScope Agent Ecosystem
版本: 1.0.0
"""

import re
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import uuid

import logging

logger = logging.getLogger(__name__)


@dataclass
class ActionItem:
    """待办事项数据结构"""
    id: str
    description: str
    assignee: Optional[str]
    deadline: Optional[str]
    priority: str  # "high", "medium", "low"
    status: str  # "pending", "in_progress", "completed"
    source: str  # "audio" 或 "image"


@dataclass
class MeetingSummary:
    """会议纪要数据结构"""
    title: str
    date: str
    duration: float
    participants: List[str]
    key_points: List[str]
    decisions: List[str]
    action_items: List[ActionItem]
    audio_transcript: str
    whiteboard_content: str
    metadata: Dict[str, Any]


def generate_ics_file(
    summary: MeetingSummary,
    output_path: str,
    organizer: str = "LocalPrivacyScribe"
) -> str:
    """
    生成标准 ICS 日程文件
    
    生成的 .ics 文件兼容主流日历应用（Outlook、Google Calendar、Apple Calendar、Thunderbird 等）。
    文件格式遵循 RFC 5545 标准。
    
    Args:
        summary: 会议纪要数据对象
        output_path: 输出文件路径（.ics 后缀）
        organizer: 组织者名称
        
    Returns:
        生成的 ICS 文件完整路径
        
    Raises:
        ValueError: 输入数据无效
        IOError: 文件写入失败
        
    Example:
        >>> summary = MeetingSummary(...)
        >>> ics_path = generate_ics_file(summary, "./meeting.ics")
        >>> print(f"日程文件已生成: {ics_path}")
    """
    try:
        # 验证输入
        if not summary.title or not summary.date:
            raise ValueError("会议标题和日期不能为空")
        
        # 解析会议日期
        meeting_date = datetime.fromisoformat(summary.date)
        
        # 计算会议时间（默认 1 小时）
        start_time = meeting_date
        end_time = meeting_date + timedelta(minutes=int(summary.duration * 60) if summary.duration > 0 else 60)
        
        # 生成唯一事件 ID
        event_uid = str(uuid.uuid4())
        
        # 构建 ICS 内容
        ics_content = f"""BEGIN:VCALENDAR
VERSION:2.0
PRODID:-//LocalPrivacyScribe//CN
CALSCALE:GREGORIAN
METHOD:PUBLISH
BEGIN:VEVENT
UID:{event_uid}
DTSTART:{start_time.strftime('%Y%m%dT%H%M%S')}
DTEND:{end_time.strftime('%Y%m%dT%H%M%S')}
SUMMARY:{_escape_ics_text(summary.title)}
DESCRIPTION:{_escape_ics_text(_format_description(summary))}
LOCATION:本地会议
ORGANIZER;CN={_escape_ics_text(organizer)}:mailto:local@privacy-scribe.local
STATUS:CONFIRMED
SEQUENCE:0
BEGIN:VALARM
TRIGGER:-PT15M
ACTION:DISPLAY
DESCRIPTION:会议提醒
END:VALARM
END:VEVENT
END:VCALENDAR
"""
        
        # 写入文件
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(ics_content)
        
        logger.info(f"ICS 文件已生成: {output_file}")
        return str(output_file.absolute())
        
    except Exception as e:
        logger.error(f"生成 ICS 文件失败: {str(e)}")
        raise IOError(f"ICS 文件生成失败: {str(e)}")


def _escape_ics_text(text: str) -> str:
    """
    转义 ICS 文本中的特殊字符
    
    ICS 格式要求对特定字符进行转义：
    - 逗号、分号、反斜杠、换行符需要转义
    - 长文本需要折叠（每行 75 字符）
    
    Args:
        text: 原始文本
        
    Returns:
        转义后的 ICS 安全文本
    """
    # 转义特殊字符
    text = text.replace('\\', '\\\\')
    text = text.replace(';', '\\;')
    text = text.replace(',', '\\,')
    text = text.replace('\n', '\\n')
    
    # 折叠长行（每 75 字符换行，缩进一个空格）
    lines = []
    for i in range(0, len(text), 75):
        if i == 0:
            lines.append(text[i:i+75])
        else:
            lines.append(' ' + text[i:i+75])
    
    return ''.join(lines)


def _format_description(summary: MeetingSummary) -> str:
    """
    格式化会议描述信息
    
    Args:
        summary: 会议纪要
        
    Returns:
        格式化的描述文本
    """
    lines = [
        "=== 会议纪要 ===",
        f"日期: {summary.date}",
        f"时长: {summary.duration:.1f} 分钟",
        f"参会人: {', '.join(summary.participants) if summary.participants else '未记录'}",
        "",
        "=== 关键要点 ===",
    ]
    
    for i, point in enumerate(summary.key_points, 1):
        lines.append(f"{i}. {point}")
    
    lines.extend([
        "",
        "=== 决策事项 ===",
    ])
    
    for i, decision in enumerate(summary.decisions, 1):
        lines.append(f"{i}. {decision}")
    
    lines.extend([
        "",
        "=== 待办事项 ===",
    ])
    
    for item in summary.action_items:
        status_icon = "✓" if item.status == "completed" else "○"
        lines.append(
            f"{status_icon} [{item.priority.upper()}] {item.description}"
            f" (负责人: {item.assignee or '未指定'}, 截止: {item.deadline or '未定'})"
        )
    
    return '\n'.join(lines)


def generate_markdown_summary(
    summary: MeetingSummary,
    output_path: str,
    template: str = "default"
) -> str:
    """
    生成 Markdown 格式的会议纪要
    
    生成结构化的 Markdown 文件，包含会议的所有关键信息。
    支持多种模板样式（default, minimal, detailed）。
    
    Args:
        summary: 会议纪要数据对象
        output_path: 输出文件路径（.md 后缀）
        template: 模板类型，可选 "default", "minimal", "detailed"
        
    Returns:
        生成的 Markdown 文件完整路径
        
    Raises:
        ValueError: 输入数据无效
        IOError: 文件写入失败
        
    Example:
        >>> summary = MeetingSummary(...)
        >>> md_path = generate_markdown_summary(summary, "./meeting.md")
        >>> print(f"纪要已导出: {md_path}")
    """
    try:
        # 验证输入
        if not summary.title:
            raise ValueError("会议标题不能为空")
        
        # 选择模板
        if template == "minimal":
            content = _generate_minimal_template(summary)
        elif template == "detailed":
            content = _generate_detailed_template(summary)
        else:
            content = _generate_default_template(summary)
        
        # 写入文件
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"Markdown 纪要已生成: {output_file}")
        return str(output_file.absolute())
        
    except Exception as e:
        logger.error(f"生成 Markdown 文件失败: {str(e)}")
        raise IOError(f"Markdown 文件生成失败: {str(e)}")


def _generate_default_template(summary: MeetingSummary) -> str:
    """生成默认模板的 Markdown 内容"""
    lines = [
        f"# {summary.title}",
        "",
        "## 基本信息",
        "",
        f"- **会议日期**: {summary.date}",
        f"- **会议时长**: {summary.duration:.1f} 分钟",
        f"- **参会人员**: {', '.join(summary.participants) if summary.participants else '未记录'}",
        "",
        "## 关键要点",
        "",
    ]
    
    for i, point in enumerate(summary.key_points, 1):
        lines.append(f"{i}. {point}")
    
    lines.extend([
        "",
        "## 决策事项",
        "",
    ])
    
    for i, decision in enumerate(summary.decisions, 1):
        lines.append(f"{i}. {decision}")
    
    lines.extend([
        "",
        "## 待办事项",
        "",
        "| 优先级 | 任务 | 负责人 | 截止日期 | 状态 |",
        "|--------|------|--------|----------|------|",
    ])
    
    for item in summary.action_items:
        status_cn = {
            "pending": "待处理",
            "in_progress": "进行中",
            "completed": "已完成"
        }.get(item.status, item.status)
        
        priority_cn = {
            "high": "高",
            "medium": "中",
            "low": "低"
        }.get(item.priority, item.priority)
        
        lines.append(
            f"| {priority_cn} | {item.description} | {item.assignee or '未指定'} | "
            f"{item.deadline or '未定'} | {status_cn} |"
        )
    
    lines.extend([
        "",
        "## 原始内容",
        "",
        "### 音频转录",
        "",
        "> 以下内容由 ASR 引擎自动转录",
        "",
        summary.audio_transcript or "（无音频内容）",
        "",
        "### 白板内容",
        "",
        "> 以下内容由 OCR 引擎自动识别",
        "",
        summary.whiteboard_content or "（无白板内容）",
        "",
        "---",
        "",
        f"*本纪要由 LocalPrivacyScribe 自动生成 | 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
    ])
    
    return '\n'.join(lines)


def _generate_minimal_template(summary: MeetingSummary) -> str:
    """生成极简模板的 Markdown 内容"""
    lines = [
        f"# {summary.title}",
        f"**日期**: {summary.date} | **时长**: {summary.duration:.0f} 分钟",
        "",
        "## 要点",
    ]
    
    for point in summary.key_points:
        lines.append(f"- {point}")
    
    lines.append("")
    return '\n'.join(lines)


def _generate_detailed_template(summary: MeetingSummary) -> str:
    """生成详细模板的 Markdown 内容（包含元数据）"""
    lines = [
        f"# {summary.title}",
        "",
        "## 元数据",
        "",
        f"- **生成工具**: LocalPrivacyScribe v1.0.0",
        f"- **生成时间**: {datetime.now().isoformat()}",
        f"- **会议日期**: {summary.date}",
        f"- **会议时长**: {summary.duration:.2f} 秒",
        f"- **参会人数**: {len(summary.participants)}",
        "",
        "## 关键要点",
        "",
    ]
    
    for i, point in enumerate(summary.key_points, 1):
        lines.append(f"{i}. {point}")
    
    lines.extend([
        "",
        "## 决策事项",
        "",
    ])
    
    for i, decision in enumerate(summary.decisions, 1):
        lines.append(f"{i}. {decision}")
    
    lines.extend([
        "",
        "## 待办事项（详细）",
        "",
    ])
    
    for item in summary.action_items:
        lines.append(f"### {item.description}")
        lines.append(f"- **ID**: {item.id}")
        lines.append(f"- **负责人**: {item.assignee or '未指定'}")
        lines.append(f"- **截止日期**: {item.deadline or '未定'}")
        lines.append(f"- **优先级**: {item.priority}")
        lines.append(f"- **状态**: {item.status}")
        lines.append(f"- **来源**: {item.source}")
        lines.append("")
    
    lines.extend([
        "## 完整转录",
        "",
        "<details>",
        "<summary>音频转录内容（点击展开）</summary>",
        "",
        summary.audio_transcript or "（无音频内容）",
        "",
        "</details>",
        "",
        "<details>",
        "<summary>白板识别内容（点击展开）</summary>",
        "",
        summary.whiteboard_content or "（无白板内容）",
        "",
        "</details>",
        "",
        "## 原始 JSON 数据",
        "",
        "```json",
        json.dumps(asdict(summary), ensure_ascii=False, indent=2),
        "```",
    ])
    
    return '\n'.join(lines)


def parse_action_items(
    text: str,
    source: str = "audio",
    assignee_hint: Optional[str] = None
) -> List[ActionItem]:
    """
    从文本中提取待办事项
    
    使用正则表达式和启发式规则从会议文本中识别待办事项。
    支持识别以下模式：
    - "需要/应该/必须 + 动作"
    - "TODO/待办/行动项"
    - "截止日期 + 任务"
    
    Args:
        text: 会议文本内容
        source: 内容来源（"audio" 或 "image"）
        assignee_hint: 负责人提示（如果文本中未明确指定）
        
    Returns:
        提取的待办事项列表
        
    Example:
        >>> text = "张三需要在下周五前完成报告，李四应该跟进客户反馈"
        >>> items = parse_action_items(text, source="audio")
        >>> print(items[0].description)
    """
    action_items = []
    
    # 定义待办事项识别模式
    patterns = [
        # 模式1: "XXX 需要/应该/必须/得 + 动作"
        r'([\u4e00-\u9fa5]{2,4})\s*[需要应该必须得]\s*(.+?)(?:，|。|；|$)',
        # 模式2: "TODO/待办/行动项: XXX"
        r'(?:TODO|待办|行动项)[：:]\s*(.+?)(?:，|。|；|$)',
        # 模式3: "截止日期 + 任务"
        r'(\d{4}[-/年]\d{1,2}[-/月]\d{1,2}[日号]?前?)\s*[，,]?\s*(.+?)(?:，|。|；|$)',
        # 模式4: "XXX + 动作 + 时间"
        r'([\u4e00-\u9fa5]{2,4})\s*(.+?)(?:在|于|前)\s*(\d{4}[-/年]\d{1,2}[-/月]\d{1,2}[日号]?)\s*(?:前|完成|提交)',
    ]
    
    seen_descriptions = set()
    
    for pattern in patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        
        for match in matches:
            groups = match.groups()
            
            # 提取任务描述
            if len(groups) >= 2:
                description = groups[-1].strip() if isinstance(groups, tuple) else groups[0].strip()
                assignee = groups[0].strip() if len(groups) >= 3 and len(groups[0]) <= 4 else None
                deadline = groups[1].strip() if len(groups) >= 3 and re.match(r'\d{4}', groups[1]) else None
            else:
                description = match.group(0).strip()
                assignee = None
                deadline = None
            
            # 去重
            if description in seen_descriptions:
                continue
            
            seen_descriptions.add(description)
            
            # 推断优先级
            priority = _infer_priority(description, text)
            
            # 推断截止日期
            if not deadline:
                deadline = _infer_deadline(description, text)
            
            # 创建待办事项
            item = ActionItem(
                id=str(uuid.uuid4())[:8],
                description=description,
                assignee=assignee or assignee_hint,
                deadline=deadline,
                priority=priority,
                status="pending",
                source=source
            )
            
            action_items.append(item)
    
    logger.info(f"从 {source} 内容中提取到 {len(action_items)} 个待办事项")
    return action_items


def _infer_priority(description: str, context: str) -> str:
    """
    推断任务优先级
    
    Args:
        description: 任务描述
        context: 上下文文本
        
    Returns:
        优先级：'high', 'medium', 'low'
    """
    high_keywords = ['紧急', '重要', '尽快', '立即', 'urgent', 'critical', 'asap']
    medium_keywords = ['应该', '需要', 'should', 'need']
    
    desc_lower = description.lower()
    ctx_lower = context.lower()
    
    for keyword in high_keywords:
        if keyword in desc_lower or keyword in ctx_lower:
            return "high"
    
    for keyword in medium_keywords:
        if keyword in desc_lower or keyword in ctx_lower:
            return "medium"
    
    return "low"


def _infer_deadline(description: str, context: str) -> Optional[str]:
    """
    推断任务截止日期
    
    Args:
        description: 任务描述
        context: 上下文文本
        
    Returns:
        截止日期字符串（ISO 格式），如果无法推断则返回 None
    """
    # 日期模式匹配
    date_patterns = [
        r'(\d{4}[-/年]\d{1,2}[-/月]\d{1,2}[日号]?)',
        r'(今天|明天|后天|下周[一二三四五六日天])',
        r'(\d{1,2}[日号])',
    ]
    
    combined_text = description + " " + context
    
    for pattern in date_patterns:
        match = re.search(pattern, combined_text)
        if match:
            date_str = match.group(1)
            try:
                # 尝试解析日期
                parsed_date = _parse_chinese_date(date_str)
                return parsed_date.strftime('%Y-%m-%d')
            except:
                continue
    
    return None


def _parse_chinese_date(date_str: str) -> datetime:
    """
    解析中文日期表达式
    
    Args:
        date_str: 中文日期字符串
        
    Returns:
        datetime 对象
        
    Raises:
        ValueError: 无法解析的日期格式
    """
    today = datetime.now()
    
    # 相对日期
    relative_dates = {
        '今天': 0,
        '明天': 1,
        '后天': 2,
    }
    
    for key, days in relative_dates.items():
        if key in date_str:
            return today + timedelta(days=days)
    
    # 下周
    week_match = re.search(r'下周([一二三四五六日天])', date_str)
    if week_match:
        weekday_map = {'一': 0, '二': 1, '三': 2, '四': 3, '五': 4, '六': 5, '日': 6, '天': 6}
        target_weekday = weekday_map.get(week_match.group(1))
        if target_weekday is not None:
            days_ahead = target_weekday - today.weekday()
            if days_ahead <= 0:
                days_ahead += 7
            return today + timedelta(days=days_ahead)
    
    # 标准日期格式
    try:
        # 尝试多种格式
        for fmt in ['%Y-%m-%d', '%Y/%m/%d', '%Y年%m月%d日']:
            try:
                return datetime.strptime(re.sub(r'[年月/]', '-', date_str).rstrip('日号'), fmt)
            except:
                continue
    except:
        pass
    
    raise ValueError(f"无法解析日期: {date_str}")


def validate_datetime(
    date_str: str,
    time_str: Optional[str] = None,
    fmt: str = "%Y-%m-%d"
) -> str:
    """
    验证和格式化日期时间
    
    Args:
        date_str: 日期字符串
        time_str: 时间字符串（可选，格式 HH:MM）
        fmt: 输入日期格式
        
    Returns:
        标准化的 ISO 格式日期时间字符串
        
    Raises:
        ValueError: 日期格式无效
        
    Example:
        >>> dt = validate_datetime("2024-12-25", "14:30")
        >>> print(dt)  # 2024-12-25T14:30:00
    """
    try:
        date_obj = datetime.strptime(date_str, fmt)
        
        if time_str:
            time_obj = datetime.strptime(time_str, "%H:%M")
            date_obj = date_obj.replace(
                hour=time_obj.hour,
                minute=time_obj.minute,
                second=0
            )
        
        return date_obj.isoformat()
        
    except Exception as e:
        raise ValueError(f"日期时间格式无效: {str(e)}")


def merge_summaries(
    summaries: List[MeetingSummary],
    title: Optional[str] = None
) -> MeetingSummary:
    """
    合并多个会议纪要
    
    将多次会议的纪要合并为一个综合纪要，用于周报、月报等场景。
    
    Args:
        summaries: 会议纪要列表
        title: 合并后的标题（可选）
        
    Returns:
        合并后的会议纪要
        
    Example:
        >>> weekly = merge_summaries(meeting_list, title="本周会议汇总")
    """
    if not summaries:
        raise ValueError("至少需要一个会议纪要")
    
    # 合并基本信息
    merged_title = title or f"合并会议纪要 ({len(summaries)} 场会议)"
    all_participants = list(set(
        p for s in summaries for p in s.participants
    ))
    
    # 合并内容
    all_key_points = []
    all_decisions = []
    all_action_items = []
    all_audio = []
    all_whiteboard = []
    
    total_duration = sum(s.duration for s in summaries)
    dates = [s.date for s in summaries]
    
    for summary in summaries:
        all_key_points.extend(summary.key_points)
        all_decisions.extend(summary.decisions)
        all_action_items.extend(summary.action_items)
        all_audio.append(f"## {summary.date}\n\n{summary.audio_transcript}")
        all_whiteboard.append(f"## {summary.date}\n\n{summary.whiteboard_content}")
    
    return MeetingSummary(
        title=merged_title,
        date=dates[0],
        duration=total_duration,
        participants=all_participants,
        key_points=all_key_points,
        decisions=all_decisions,
        action_items=all_action_items,
        audio_transcript="\n\n".join(all_audio),
        whiteboard_content="\n\n".join(all_whiteboard),
        metadata={
            "merged_from": len(summaries),
            "source_dates": dates
        }
    )


def export_to_json(
    summary: MeetingSummary,
    output_path: str
) -> str:
    """
    导出会议纪要为 JSON 格式
    
    便于后续程序化处理和集成到其他系统。
    
    Args:
        summary: 会议纪要数据对象
        output_path: 输出文件路径（.json 后缀）
        
    Returns:
        生成的 JSON 文件完整路径
    """
    try:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 转换为字典
        data = {
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
            "whiteboard_content": summary.whiteboard_content,
            "metadata": summary.metadata
        }
        
        # 写入 JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"JSON 文件已生成: {output_file}")
        return str(output_file.absolute())
        
    except Exception as e:
        logger.error(f"导出 JSON 失败: {str(e)}")
        raise IOError(f"JSON 导出失败: {str(e)}")