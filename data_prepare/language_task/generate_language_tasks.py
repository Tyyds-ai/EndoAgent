import os
import random
import json
import glob
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
from PIL import Image
import cv2

load_dotenv()

# VQA任务问题模板
vqa_question_templates = [
    "Can you describe the findings visible in this endoscopy image?",
    "What can you observe in this endoscopic image?",
    "Can you provide a detailed description of what's visible in this endoscopy image?",
    "Please describe the characteristics of the tissue shown in this endoscopic image.",
    "What are the key features you can identify in this endoscopy image?",
    "Can you analyze the visual features of this endoscopy image?",
    "What abnormalities or normal findings can you identify in this image?",
    "Please provide an analysis of the findings in this endoscopy image.",
    "Can you describe the characteristics of any lesions or tissue in this endoscopy image?",
    "What diagnostic features are visible in this endoscopic visualization?"
]

# MRG任务问题模板
mrg_question_templates = [
    "Can you generate a detailed medical report for this endoscopy image?",
    "Please create a comprehensive endoscopy report based on this image.",
    "Generate a professional endoscopy examination report for this image.",
    "Please prepare a detailed endoscopy report with your assessment of this image.",
    "Create a medical report documenting the findings in this endoscopy image.",
    "Generate a clinical endoscopy report based on the visual findings in this image.",
    "Please provide a structured medical report for this endoscopic examination.",
    "Create a comprehensive diagnostic report for this endoscopy image.",
    "Generate a detailed clinical assessment report for this endoscopic image.",
    "Please write a professional medical report based on this endoscopy examination."
]

# 注意：标准答案将通过qwen_vl_plus生成，这里不需要预设答案模板

def collect_all_images_from_paths():
    """
    从指定路径收集所有图像文件
    """
    # 数据路径 (使用train目录)
    base_paths = [
        "/path/to/data/train"
    ]
    
    # 定义类别
    categories = ["normal", "polyp", "adenoma", "cancer"]
    
    # 收集每个类别的所有图像文件
    images = {category: [] for category in categories}
    
    print("开始从以下路径收集图像:")
    for path in base_paths:
        print(f"- {path}")
    print("-" * 30)

    for category in categories:
        for base_path in base_paths:
            path = os.path.join(base_path, category, "image")
            if not os.path.isdir(path):
                continue
            
            # 查找该路径下的所有图像文件
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                images[category].extend(glob.glob(os.path.join(path, ext)))

    total_images = 0
    print("\n--- 收集结果汇总 ---")
    for category, image_list in images.items():
        total_images += len(image_list)
        print(f"{category}: 找到 {len(image_list)} 张图像")
    
    print(f"总计: {total_images} 张图像")
    return images

def split_images_for_language_tasks(images):
    """
    为语言类任务分配图像：
    - 使用每个类别的所有图像
    - 一半用于VQA，一半用于MRG
    """
    task_images = {
        "vqa": {},
        "mrg": {}
    }
    
    for category, img_list in images.items():
        # 打乱图像顺序
        random.shuffle(img_list)
        
        # 一半用于VQA，一半用于MRG
        mid_point = len(img_list) // 2
        
        task_images["vqa"][category] = img_list[:mid_point]
        task_images["mrg"][category] = img_list[mid_point:]
        
        print(f"{category}: VQA任务 {len(task_images['vqa'][category])} 张，"
              f"MRG任务 {len(task_images['mrg'][category])} 张")
    
    return task_images

def generate_vqa_entry(question_id, image_path, category):
    """
    生成VQA任务条目（不包含答案，答案将由qwen_vl_plus生成）
    """
    question = random.choice(vqa_question_templates)
    
    entry = {
        "question_id": question_id,
        "task_type": "vqa",
        "image": os.path.basename(image_path),
        "image_path": image_path,
        "question": question,
        "category": category,
        "data_source": "Private"
    }
    
    return entry

def generate_mrg_entry(question_id, image_path, category):
    """
    生成MRG任务条目（不包含答案，答案将由qwen_vl_plus生成）
    """
    question = random.choice(mrg_question_templates)
    
    entry = {
        "question_id": question_id,
        "task_type": "mrg",
        "image": os.path.basename(image_path),
        "image_path": image_path,
        "question": question,
        "category": category,
        "data_source": "Private"
    }
    
    return entry

def generate_language_tasks_jsonl(output_file):
    """
    生成语言类任务的JSONL文件
    包括VQA和MRG两种任务
    """
    # 收集所有图像
    all_images = collect_all_images_from_paths()
    
    # 为语言任务分配图像
    task_images = split_images_for_language_tasks(all_images)
    
    # 生成数据条目
    entries = []
    question_id = 0
    
    # 生成VQA任务
    print("\n生成VQA任务...")
    for category, images in task_images["vqa"].items():
        print(f"处理 {category} 类别的 {len(images)} 张图像...")
        for image_path in tqdm(images, desc=f"VQA-{category}"):
            entry = generate_vqa_entry(question_id, image_path, category)
            entries.append(entry)
            question_id += 1
    
    # 生成MRG任务
    print("\n生成MRG任务...")
    for category, images in task_images["mrg"].items():
        print(f"处理 {category} 类别的 {len(images)} 张图像...")
        for image_path in tqdm(images, desc=f"MRG-{category}"):
            entry = generate_mrg_entry(question_id, image_path, category)
            entries.append(entry)
            question_id += 1
    
    # 写入JSONL文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"\n已成功生成 {len(entries)} 条语言任务记录到 {output_file}")
    
    # 统计每个任务和类别的数量
    stats = {}
    for entry in entries:
        task = entry["task_type"]
        category = entry["category"]
        
        if task not in stats:
            stats[task] = {"normal": 0, "polyp": 0, "adenoma": 0, "cancer": 0, "total": 0}
        
        stats[task][category] += 1
        stats[task]["total"] += 1
    
    # 打印统计信息
    print(f"\n=== 语言任务分布统计 ===")
    for task, counts in stats.items():
        print(f"{task.upper()}: 总计 {counts['total']} 条记录")
        for category in ["normal", "polyp", "adenoma", "cancer"]:
            if counts[category] > 0:
                print(f"  - {category}: {counts[category]} 条")
    
    return entries

def print_sample_data():
    """打印示例数据格式"""
    print("\n=== 语言任务数据格式示例 ===")
    
    # VQA任务示例
    vqa_example = {
        "question_id": 0,
        "task_type": "vqa",
        "image": "polyp_001.jpg",
        "image_path": "/path/to/polyp_001.jpg",
        "question": "Can you describe the findings visible in this endoscopy image?",
        "category": "polyp",
        "data_source": "Private"
    }
    
    # MRG任务示例
    mrg_example = {
        "question_id": 1,
        "task_type": "mrg",
        "image": "adenoma_002.jpg", 
        "image_path": "/path/to/adenoma_002.jpg",
        "question": "Can you generate a detailed medical report for this endoscopy image?",
        "category": "adenoma",
        "data_source": "Private"
    }
    
    print("VQA任务示例:")
    print(json.dumps(vqa_example, indent=2, ensure_ascii=False))
    print("\nMRG任务示例:")
    print(json.dumps(mrg_example, indent=2, ensure_ascii=False))
    print("\n注意：答案字段将由qwen_vl_plus模型生成，不在初始数据中包含")
    print("\n" + "="*50)

def main():
    """主函数"""
    # 设置随机种子确保可重复性
    random.seed(42)
    
    # 输出文件路径
    output_file = "/path/to/endoscopy_language_tasks.jsonl"
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    print("开始生成语言类任务验证集...")
    print("使用train数据集进行生成...")
    print("="*60)
    
    # 生成语言任务数据 (使用所有可用图像)
    entries = generate_language_tasks_jsonl(output_file)
    
    print("\n" + "="*60)
    print("语言任务验证集生成完成!")
    print(f"输出文件: {output_file}")
    print(f"总题目数: {len(entries)}")
    print("注意：答案字段需要后续通过qwen_vl_plus模型生成")
    
    # 额外生成VQA和MRG的单独文件
    vqa_file = "/path/to/endoscopy_vqa_tasks.jsonl"
    mrg_file = "/path/to/endoscopy_mrg_tasks.jsonl"
    
    # 分离VQA和MRG题目
    vqa_entries = [e for e in entries if e["task_type"] == "vqa"]
    mrg_entries = [e for e in entries if e["task_type"] == "mrg"]
    
    # 写入VQA题目文件
    with open(vqa_file, 'w', encoding='utf-8') as f:
        for entry in vqa_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    # 写入MRG题目文件
    with open(mrg_file, 'w', encoding='utf-8') as f:
        for entry in mrg_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"\n额外生成的单独文件:")
    print(f"VQA任务: {vqa_file} ({len(vqa_entries)} 题)")
    print(f"MRG任务: {mrg_file} ({len(mrg_entries)} 题)")

if __name__ == "__main__":
    # 先显示示例格式
    print_sample_data()
    
    main()
