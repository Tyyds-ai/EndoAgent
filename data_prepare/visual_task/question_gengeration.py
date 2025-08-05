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

# 多项选择题问题模板
question_templates = {
    # 内窥镜图像分类任务 - 多项选择题
    "lesion_classification": [
        "Which digestive organ is depicted in this image?",
        "Which anatomical structure is depicted in this endoscopic image?", 
        "Which digestive organ is depicted in this endoscopic image?",
        "Which anatomical structure is depicted in this endoscopic visualization?",
        "What type of tissue or lesion is shown in this endoscopic image?"
    ],
    
    # 内窥镜病变检测任务 - 边界框选择
    "visual_grounding": [
        "Which of the following bounding box coordinates correctly identifies the lesion location in this image?",
        "Select the correct bounding box coordinates [top, left, bottom, right] for the lesion in this endoscopic image:",
        "Which bounding box best localizes the abnormal area in this image?",
        "Choose the accurate coordinates that mark the lesion boundaries:",
        "Which of these coordinate sets correctly identifies the target region?"
    ]
}

# 类别选项定义
classification_options = {
    "A": "normal",
    "B": "polyp", 
    "C": "adenoma",
    "D": "cancer"
}

def generate_wrong_bbox(correct_bbox, image_shape=(400, 400)):
    """
    生成错误的边界框坐标作为干扰项
    """
    wrong_bboxes = []
    height, width = image_shape
    
    for _ in range(3):  # 生成3个错误的边界框
        # 随机生成边界框
        top = random.randint(0, height // 2)
        left = random.randint(0, width // 2)
        bottom = random.randint(top + 20, min(top + 100, height))
        right = random.randint(left + 20, min(left + 100, width))
        
        wrong_bbox = [top, left, bottom, right]
        
        # 确保与正确答案不同
        if wrong_bbox != correct_bbox:
            wrong_bboxes.append(wrong_bbox)
    
    return wrong_bboxes

def generate_lesion_classification(question_id, image_path, category):
    """
    生成分类多项选择题
    """
    question = random.choice(question_templates["lesion_classification"])
    
    # 根据类别确定正确答案
    correct_answer = None
    for key, value in classification_options.items():
        if value == category:
            correct_answer = key
            break
    
    # 构建选项字符串
    options_text = "Options:\n"
    for key, value in classification_options.items():
        options_text += f"{key}. {value}\n"
    options_text += "Please select the correct answer from the options above."
    
    entry = {
        "question_id": question_id,
        "task_type": "lesion_classification",
        "image": os.path.basename(image_path),
        "image_path": image_path,
        "question": question,
        "options": options_text,
        "category": category,
        "correct_answer": correct_answer,
        "answer_choices": classification_options,
        "data_source": "Private"
    }
    
    return entry

def generate_visual_grounding(question_id, image_path, category):
    """
    生成检测边界框多项选择题
    """
    # 只为非正常类别生成检测题
    if category == "normal":
        return None
        
    # 获取正确的边界框
    mask_path = get_mask_path(image_path)
    correct_bbox = get_bounding_box_from_mask(mask_path)
    
    if correct_bbox is None:
        return None
    
    question = random.choice(question_templates["visual_grounding"])
    
    # 生成错误的边界框作为干扰项
    wrong_bboxes = generate_wrong_bbox(correct_bbox)
    
    # 组合所有选项
    all_options = [correct_bbox] + wrong_bboxes
    random.shuffle(all_options)  # 打乱顺序
    
    # 找到正确答案的位置
    correct_answer_idx = all_options.index(correct_bbox)
    correct_answer = chr(65 + correct_answer_idx)  # A, B, C, D
    
    # 构建选项字符串
    options_text = "Options:\n"
    answer_choices = {}
    for i, bbox in enumerate(all_options):
        option_key = chr(65 + i)  # A, B, C, D
        bbox_str = f"[top={bbox[0]}, left={bbox[1]}, bottom={bbox[2]}, right={bbox[3]}]"
        options_text += f"{option_key}. {bbox_str}\n"
        answer_choices[option_key] = bbox
    options_text += "Please select the correct answer from the options above."
    
    entry = {
        "question_id": question_id,
        "task_type": "visual_grounding",
        "image": os.path.basename(image_path),
        "image_path": image_path,
        "question": question,
        "options": options_text,
        "category": category,
        "correct_answer": correct_answer,
        "correct_bbox": correct_bbox,
        "answer_choices": answer_choices,
        "data_source": "Private"
    }
    
    return entry

def get_mask_path(image_path):
    """
    根据图像路径生成对应的掩膜路径
    """
    # 替换路径中的 "image" 为 "mask"
    return image_path.replace("/image/", "/mask/")

def get_bounding_box_from_mask(mask_path):
    """
    从掩膜图像中获取最小外接矩形的边界框坐标
    返回格式为 [top, left, bottom, right]
    如果掩膜不存在或为空，返回None
    """
    if not os.path.exists(mask_path):
        return None
    
    try:
        # 读取掩膜图像
        mask = np.array(Image.open(mask_path).convert('L'))
        
        # 检查掩膜是否为空（全黑）
        if np.max(mask) == 0:
            return None
        
        # 二值化处理
        _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        # 查找轮廓
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
            
        # 获取最大轮廓
        max_contour = max(contours, key=cv2.contourArea)
        
        # 获取边界框
        x, y, w, h = cv2.boundingRect(max_contour)
        
        # 返回格式 [top, left, bottom, right]
        return [y, x, y+h, x+w]
    except Exception as e:
        print(f"处理掩膜文件时出错: {mask_path}, 错误: {str(e)}")
        return None

def collect_all_images_from_paths():
    """
    从指定路径收集所有图像文件
    """
    # 数据路径 (包含test和val)
    base_paths = [
        "/path/to/test",
        "/path/to/val"
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

def split_images_for_tasks(images):
    """
    将图像分配给不同任务：
    - normal类别全部用于分类任务
    - 其他类别一半用于分类，一半用于检测
    """
    task_images = {
        "lesion_classification": {},
        "visual_grounding": {}
    }
    
    for category, img_list in images.items():
        # 打乱图像顺序
        random.shuffle(img_list)
        
        if category == "normal":
            # normal类别全部用于分类任务
            task_images["lesion_classification"][category] = img_list
            print(f"{category}: 分类任务 {len(img_list)} 张，检测任务 0 张")
        else:
            # 其他类别一半一半分配
            mid_point = len(img_list) // 2
            
            # 分类任务使用前一半
            task_images["lesion_classification"][category] = img_list[:mid_point]
            
            # 检测任务使用后一半
            task_images["visual_grounding"][category] = img_list[mid_point:]
            
            print(f"{category}: 分类任务 {len(task_images['lesion_classification'][category])} 张，"
                  f"检测任务 {len(task_images['visual_grounding'][category])} 张")
    
    return task_images

def generate_all_questions_jsonl(output_file):
    """
    生成所有图像的多项选择题JSONL文件
    normal类别全部用于分类，其他类别一半用于分类，一半用于检测
    """
    # 收集所有图像
    all_images = collect_all_images_from_paths()
    
    # 为任务分配图像
    task_images = split_images_for_tasks(all_images)
    
    # 生成数据条目
    entries = []
    question_id = 0
    
    # 生成病变分类多项选择题
    print("\n生成Lesion Classification多项选择题...")
    for category, images in task_images["lesion_classification"].items():
        print(f"处理 {category} 类别的 {len(images)} 张图像...")
        for image_path in tqdm(images, desc=f"分类-{category}"):
            entry = generate_lesion_classification(question_id, image_path, category)
            if entry:
                entries.append(entry)
                question_id += 1
    
    # 生成视觉定位多项选择题
    print("\n生成Visual Grounding多项选择题...")
    for category, images in task_images["visual_grounding"].items():
        if category == "normal":  # 跳过正常类别（实际上已经在split函数中处理了）
            continue
        print(f"处理 {category} 类别的 {len(images)} 张图像...")
        for image_path in tqdm(images, desc=f"定位-{category}"):
            entry = generate_visual_grounding(question_id, image_path, category)
            if entry:
                entries.append(entry)
                question_id += 1
    
    # 写入JSONL文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"\n已成功生成 {len(entries)} 条多项选择题记录到 {output_file}")
    
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
    print(f"\n=== 任务分布统计 ===")
    for task, counts in stats.items():
        print(f"{task}: 总计 {counts['total']} 条记录")
        for category in ["normal", "polyp", "adenoma", "cancer"]:
            if counts[category] > 0:
                print(f"  - {category}: {counts[category]} 条")
    
    return entries

def print_sample_data():
    """打印示例数据格式"""
    print("\n=== 生成的数据格式示例 ===")
    
    # 分类多项选择题示例
    classification_example = {
        "question_id": 0,
        "task_type": "lesion_classification",
        "image": "polyp_001.jpg",
        "image_path": "/path/to/polyp_001.jpg",
        "question": "Which anatomical structure is depicted in this endoscopic image?",
        "options": "Options:\nA. normal\nB. polyp\nC. adenoma\nD. cancer\nPlease select the correct answer from the options above.",
        "category": "polyp",
        "correct_answer": "B",
        "answer_choices": {
            "A": "normal",
            "B": "polyp", 
            "C": "adenoma",
            "D": "cancer"
        },
        "data_source": "Private"
    }
    
    # 检测边界框选择题示例
    detection_example = {
        "question_id": 1,
        "task_type": "visual_grounding",
        "image": "adenoma_002.jpg", 
        "image_path": "/path/to/adenoma_002.jpg",
        "question": "Which of the following bounding box coordinates correctly identifies the lesion location in this image?",
        "options": "Options:\nA. [top=45, left=123, bottom=167, right=245]\nB. [top=12, left=89, bottom=134, right=211]\nC. [top=78, left=156, bottom=200, right=278]\nD. [top=34, left=67, bottom=156, right=189]\nPlease select the correct answer from the options above.",
        "category": "adenoma",
        "correct_answer": "C",
        "correct_bbox": [78, 156, 200, 278],
        "answer_choices": {
            "A": [45, 123, 167, 245],
            "B": [12, 89, 134, 211],
            "C": [78, 156, 200, 278],
            "D": [34, 67, 156, 189]
        },
        "data_source": "Private"
    }
    
    print("分类多项选择题示例:")
    print(json.dumps(classification_example, indent=2, ensure_ascii=False))
    print("\n检测边界框选择题示例:")
    print(json.dumps(detection_example, indent=2, ensure_ascii=False))
    print("\n" + "="*50)

def main():
    """主函数"""
    # 设置随机种子确保可重复性
    random.seed(42)
    
    # 输出文件路径
    output_file = "/path/to/endoscopy_mcq_all_questions.jsonl"
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    print("开始处理所有图像数据...")
    print("="*60)
    
    # 生成所有多项选择题数据
    entries = generate_all_questions_jsonl(output_file)
    
    print("\n" + "="*60)
    print("数据转换完成!")
    print(f"输出文件: {output_file}")
    print(f"总题目数: {len(entries)}")

if __name__ == "__main__":
    # 先显示示例格式
    print_sample_data()
    
    main()