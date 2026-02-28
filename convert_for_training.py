#!/usr/bin/env python3
"""
将道德经直译的RST文件转换为适合LLM LoRA训练的文本格式。

格式说明：
- :: 后的缩进内容是原文
- *(文字)* 是强调/注释
- 缩进的段落是字词解释
- [阅读引导] 是阅读引导部分
- [机器学习解释] 是机器学习解释部分
"""

import os
import re
import glob
from pathlib import Path


def parse_rst_file(filepath):
    """解析RST文件，提取结构化内容"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # 从文件名提取章节号和标题
    filename = os.path.basename(filepath)
    match = re.match(r'(\d+)\.(.+)\.rst', filename)
    chapter_num = match.group(1) if match else ''
    chapter_title = match.group(2) if match else ''

    # 去掉文件头的元数据
    # 移除开头的注释块
    content = re.sub(r'^\.\..*?\n\n', '', content, flags=re.DOTALL)
    # 移除 :Key: Value 格式的元数据
    content = re.sub(r'^:[A-Za-z]+:.*\n', '', content, flags=re.MULTILINE)

    # 移除章节标题行（如 "1. 道可道" 后跟星号行）
    content = re.sub(r'^\d+\..*?\n[*]+\n', '', content, flags=re.MULTILINE)

    return content, chapter_num, chapter_title


def clean_text(text):
    """清理RST格式标记，转换为纯文本"""
    # 处理斜体标记 \*(文字)*\ -> （文字）
    text = re.sub(r'\\\s*\*\(([^)]+)\)\*\s*\\', r'（\1）', text)

    # 处理简单的斜体标记
    text = re.sub(r'\*([^*]+)\*', r'\1', text)

    # 处理引用标记 [1]_ 等
    text = re.sub(r'\[(\d+)\]_', r'[注\1]', text)

    # 处理文档链接 :doc:`xxx`
    text = re.sub(r':doc:`([^`]+)`', r'《\1》', text)

    # 处理下划线链接 `xxx`_
    text = re.sub(r'`([^`]+)`_', r'\1', text)

    # 清理多余空格
    text = re.sub(r' +', ' ', text)

    # 清理转义反斜杠
    text = text.replace('\\', '')

    return text.strip()


def extract_sections(content):
    """从内容中提取所有小节"""
    sections = []

    # 按二级标题分割（标题后跟等号行）
    # 匹配: 标题\n=====\n内容
    pattern = r'([^`\n][^\n]*?)\n={3,}\n(.*?)(?=(?:[^`\n][^\n]*?\n={3,}\n)|$)'
    matches = re.findall(pattern, content, re.DOTALL)

    for title, body in matches:
        title = title.strip()
        body = body.strip()
        if title and body:
            sections.append({
                'title': title,
                'content': body
            })

    return sections


def extract_structured_content(section_content):
    """从节内容中提取结构化信息"""
    result = {
        'original': '',      # 原文
        'translation': '',   # 翻译
        'word_notes': [],    # 字词解释
        'reading_guide': '', # 阅读引导
        'ml_explanation': '' # 机器学习解释
    }

    lines = section_content.split('\n')

    # 状态标记
    in_original = False
    in_reading_guide = False
    in_ml_explanation = False
    in_footnote = False

    original_lines = []
    reading_guide_lines = []
    ml_explanation_lines = []
    translation_lines = []

    # 字词解释收集
    word_notes = {}
    current_word = None
    current_explanation = []

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # 检测原文开始
        if stripped == '::':
            in_original = True
            in_reading_guide = False
            in_ml_explanation = False
            in_footnote = False
            i += 1
            continue

        # 检测阅读引导
        if '[阅读引导]' in line:
            # 保存之前的字词解释
            if current_word and current_explanation:
                word_notes[current_word] = ' '.join(current_explanation)
                current_word = None
                current_explanation = []

            in_reading_guide = True
            in_original = False
            in_ml_explanation = False
            in_footnote = False
            i += 1
            continue

        # 检测机器学习解释
        if '[机器学习解释]' in line:
            # 保存之前的字词解释
            if current_word and current_explanation:
                word_notes[current_word] = ' '.join(current_explanation)
                current_word = None
                current_explanation = []

            in_ml_explanation = True
            in_reading_guide = False
            in_original = False
            in_footnote = False
            i += 1
            continue

        # 检测脚注开始
        if stripped.startswith('.. [') and stripped.endswith(']'):
            if current_word and current_explanation:
                word_notes[current_word] = ' '.join(current_explanation)
                current_word = None
                current_explanation = []

            in_footnote = True
            in_reading_guide = False
            in_original = False
            in_ml_explanation = False
            i += 1
            continue

        # 处理原文
        if in_original:
            if line.startswith('    ') or line == '':
                if stripped:
                    original_lines.append(stripped)
            else:
                # 原文结束，开始翻译
                in_original = False

        # 处理阅读引导
        if in_reading_guide:
            # 脚注开始时结束阅读引导
            if stripped.startswith('.. ['):
                in_reading_guide = False
                in_footnote = True
            else:
                reading_guide_lines.append(stripped)

        # 处理机器学习解释
        if in_ml_explanation:
            ml_explanation_lines.append(stripped)

        # 检测字词解释（非缩进的短词后跟缩进解释）
        # 支持8空格缩进或制表符
        if (not in_original and not in_reading_guide and not in_ml_explanation
            and not in_footnote and stripped
            and not line.startswith(' ')
            and not stripped.startswith('..')
            and len(stripped) <= 8  # 字词通常较短
            and i + 1 < len(lines)
            and (lines[i + 1].startswith('        ') or lines[i + 1].startswith('\t'))):

            # 保存之前的字词解释
            if current_word and current_explanation:
                word_notes[current_word] = ' '.join(current_explanation)

            current_word = stripped
            current_explanation = []
            i += 1
            continue

        # 收集字词解释内容
        if current_word is not None:
            if line.startswith('        ') or line.startswith('\t'):
                current_explanation.append(stripped)
            else:
                # 字词解释结束
                if current_word and current_explanation:
                    word_notes[current_word] = ' '.join(current_explanation)
                current_word = None
                current_explanation = []

        # 收集翻译文本（不在特殊状态下的非空行）
        if (not in_original and not in_reading_guide and not in_ml_explanation
            and not in_footnote and current_word is None
            and stripped and not stripped.startswith('..')
            and not line.startswith('\t')):
            # 跳过看起来像标题的行
            if not re.match(r'^\d+\.', stripped):
                translation_lines.append(stripped)

        i += 1

    # 保存最后的字词解释
    if current_word and current_explanation:
        word_notes[current_word] = ' '.join(current_explanation)

    # 组装结果
    result['original'] = '\n'.join(original_lines)
    result['translation'] = clean_text('\n'.join(translation_lines))
    result['reading_guide'] = clean_text('\n'.join(
        line for line in reading_guide_lines if line
    ))
    result['ml_explanation'] = clean_text('\n'.join(
        line for line in ml_explanation_lines if line
    ))

    # 转换字词解释为列表
    for word, explanation in word_notes.items():
        if word and explanation:
            result['word_notes'].append({
                'word': word,
                'explanation': clean_text(explanation)
            })

    return result


def format_for_training(chapter_num, section_title, structured_content):
    """将结构化内容格式化为训练文本"""
    parts = []

    # 添加章节信息
    if section_title:
        header = f"【第{chapter_num}章：{section_title}】"
    else:
        return None  # 没有标题则跳过

    # 检查是否有实质内容
    has_content = (structured_content['original'] or
                   structured_content['translation'] or
                   structured_content['reading_guide'] or
                   structured_content['ml_explanation'])

    if not has_content:
        return None

    parts.append(header)
    parts.append("")

    # 添加原文
    if structured_content['original']:
        parts.append("【原文】")
        parts.append(structured_content['original'])
        parts.append("")

    # 添加翻译
    if structured_content['translation']:
        parts.append("【直译】")
        parts.append(structured_content['translation'])
        parts.append("")

    # 添加字词解释
    if structured_content['word_notes']:
        parts.append("【字词解释】")
        for note in structured_content['word_notes']:
            parts.append(f"{note['word']}：{note['explanation']}")
        parts.append("")

    # 添加阅读引导
    if structured_content['reading_guide']:
        parts.append("【阅读引导】")
        parts.append(structured_content['reading_guide'])
        parts.append("")

    # 添加机器学习解释
    if structured_content['ml_explanation']:
        parts.append("【机器学习视角】")
        parts.append(structured_content['ml_explanation'])
        parts.append("")

    return '\n'.join(parts)


def estimate_tokens(text):
    """估算文本的token数量（中文约1.5字符/token，英文约4字符/token）"""
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    other_chars = len(text) - chinese_chars
    return int(chinese_chars / 1.5 + other_chars / 4)


def split_into_chunks(texts, target_tokens=1024):
    """将文本列表分割成接近目标token数的块"""
    chunks = []
    current_chunk = []
    current_tokens = 0

    for text in texts:
        if not text or not text.strip():
            continue

        text_tokens = estimate_tokens(text)

        # 如果单个文本就超过目标，需要单独处理
        if text_tokens > target_tokens:
            # 先保存当前块
            if current_chunk:
                chunks.append('\n\n---\n\n'.join(current_chunk))
                current_chunk = []
                current_tokens = 0

            # 将大文本按段落分割
            paragraphs = text.split('\n\n')
            temp_chunk = []
            temp_tokens = 0

            for para in paragraphs:
                para_tokens = estimate_tokens(para)
                if temp_tokens + para_tokens > target_tokens * 1.2:  # 允许20%溢出
                    if temp_chunk:
                        chunks.append('\n\n'.join(temp_chunk))
                    temp_chunk = [para]
                    temp_tokens = para_tokens
                else:
                    temp_chunk.append(para)
                    temp_tokens += para_tokens

            if temp_chunk:
                chunks.append('\n\n'.join(temp_chunk))
        else:
            if current_tokens + text_tokens > target_tokens:
                chunks.append('\n\n---\n\n'.join(current_chunk))
                current_chunk = [text]
                current_tokens = text_tokens
            else:
                current_chunk.append(text)
                current_tokens += text_tokens

    if current_chunk:
        chunks.append('\n\n---\n\n'.join(current_chunk))

    return chunks


def main():
    source_dir = Path(__file__).parent / 'source'
    output_dir = Path(__file__).parent / 'training_data'
    output_dir.mkdir(exist_ok=True)

    # 获取所有章节文件（排除index和说明文件）
    rst_files = sorted(source_dir.glob('*.rst'))
    chapter_files = [f for f in rst_files if re.match(r'\d+\.', f.name)]

    all_training_texts = []

    for filepath in chapter_files:
        print(f"处理: {filepath.name}")

        content, chapter_num, chapter_title = parse_rst_file(filepath)
        sections = extract_sections(content)

        for section in sections:
            structured = extract_structured_content(section['content'])
            training_text = format_for_training(
                chapter_num, section['title'], structured
            )
            if training_text and training_text.strip():
                all_training_texts.append(training_text)
                print(f"  - {section['title']}: 约{estimate_tokens(training_text)} tokens")

    # 分割成适合训练的块
    print(f"\n共生成 {len(all_training_texts)} 个文本片段")
    chunks = split_into_chunks(all_training_texts, target_tokens=1024)
    print(f"分割为 {len(chunks)} 个训练块")

    # 保存到文件
    for i, chunk in enumerate(chunks):
        output_file = output_dir / f'training_{i+1:03d}.txt'
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(chunk)
        print(f"保存: {output_file.name} (约 {estimate_tokens(chunk)} tokens)")

    # 同时生成一个合并的JSONL格式文件
    import json
    jsonl_file = output_dir / 'training_data.jsonl'
    with open(jsonl_file, 'w', encoding='utf-8') as f:
        for chunk in chunks:
            data = {
                'text': chunk,
                'token_estimate': estimate_tokens(chunk)
            }
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
    print(f"\nJSONL格式保存至: {jsonl_file}")

    # 生成一个纯文本合并文件
    all_text_file = output_dir / 'all_training_text.txt'
    with open(all_text_file, 'w', encoding='utf-8') as f:
        f.write('\n\n' + '='*50 + '\n\n'.join(chunks))
    print(f"纯文本合并保存至: {all_text_file}")

    # 生成统计信息
    total_tokens = sum(estimate_tokens(chunk) for chunk in chunks)
    print(f"\n统计信息:")
    print(f"  总文件数: {len(chunks)}")
    print(f"  估算总tokens: {total_tokens}")
    print(f"  平均每块tokens: {total_tokens / len(chunks):.0f}")


if __name__ == '__main__':
    main()
