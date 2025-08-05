#!/usr/bin/env python3
"""
将EndoAgent的JSONL格式数据转换为VLMEvalKit要求的TSV格式
包含base64编码的图像数据
"""

import json
import pandas as pd
import os
import base64
from pathlib import Path

def image_to_base64(image_path):
    """将图像文件转换为base64编码"""
    try:
        with open(image_path, 'rb') as image_file:
            base64_string = base64.b64encode(image_file.read()).decode('utf-8')
            return base64_string
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return None

def convert_jsonl_to_tsv(jsonl_file, output_dir, dataset_name="EndoAgentBench"):
    """
    将JSONL格式转换为VLMEvalKit的TSV格式
    
    Args:
        jsonl_file: 输入的JSONL文件路径
        output_dir: 输出目录
        dataset_name: 数据集名称
    """
    
    # 读取JSONL数据
    data = []
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    # 转换为VLMEvalKit格式
    vlm_data = []
    
    for item in data:
        # 获取图像的base64编码
        image_path = item['image_path']
        base64_image = image_to_base64(image_path)
        
        if base64_image is None:
            print(f"Skipping item {item['question_id']} due to image encoding error")
            continue
        
        # 基础字段
        vlm_entry = {
            'index': item['question_id'],
            'image': base64_image,  # base64编码的图像数据
            'image_path': image_path,  # 保留图像路径
            'question': item['question'],
            'answer': item['correct_answer'],
            'category': item['task_type'],  # classification_mcq 或 detection_bbox
            'split': 'test',
            'dataset': 'private'  # 添加dataset列
        }
        
        # 处理多选题选项
        if item['task_type'] == 'classification_mcq':
            # 分类任务：直接使用选项文本
            vlm_entry['A'] = 'normal'
            vlm_entry['B'] = 'polyp'
            vlm_entry['C'] = 'adenoma'
            vlm_entry['D'] = 'cancer'
            vlm_entry['l2-category'] = item['category']  # actual category: normal/polyp/adenoma/cancer
            
        elif item['task_type'] == 'detection_bbox':
            # 检测任务：使用边界框坐标
            choices = item['answer_choices']
            for key in ['A', 'B', 'C', 'D']:
                if key in choices:
                    bbox = choices[key]
                    vlm_entry[key] = f"[top={bbox[0]}, left={bbox[1]}, bottom={bbox[2]}, right={bbox[3]}]"
            vlm_entry['l2-category'] = item['category']
            # 添加正确边界框信息用于评估
            if 'correct_bbox' in item:
                bbox = item['correct_bbox']
                vlm_entry['correct_bbox'] = f"[{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]"
        
        vlm_data.append(vlm_entry)
    
    # 转换为DataFrame
    df = pd.DataFrame(vlm_data)
    
    # 确保列的顺序符合VLMEvalKit要求
    standard_columns = ['index', 'image', 'image_path', 'question', 'A', 'B', 'C', 'D', 'answer', 'category', 'l2-category', 'split', 'dataset']
    
    # 添加检测任务特有的列
    if 'correct_bbox' in df.columns:
        standard_columns.append('correct_bbox')
    
    # 重新排列列顺序
    df = df.reindex(columns=standard_columns, fill_value='')
    
    # 保存为TSV文件
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{dataset_name}.tsv")
    df.to_csv(output_file, sep='\t', index=False)
    
    print(f"转换完成！数据已保存到: {output_file}")
    print(f"总共转换了 {len(df)} 条记录")
    print(f"任务分布:")
    print(df['category'].value_counts())
    print(f"\n类别分布:")
    print(df['l2-category'].value_counts())
    
    return output_file

def main():
    """主函数"""
    
    # 配置路径
    jsonl_file = "/path/to/endoscopy_mcq_questions.jsonl"
    output_dir = "/path/to/EndoAgentBench"
    dataset_name = "EndoAgentBench"
    
    # 检查输入文件是否存在
    if not os.path.exists(jsonl_file):
        print(f"错误: JSONL文件不存在: {jsonl_file}")
        print("请先运行question_generation.py生成数据")
        return
    
    # 执行转换
    output_file = convert_jsonl_to_tsv(jsonl_file, output_dir, dataset_name)
    
    # 显示部分数据样例
    df = pd.read_csv(output_file, sep='\t')
    print(f"\n数据样例 (前3行):")
    print(df.head(3).to_string())

if __name__ == "__main__":
    main()
