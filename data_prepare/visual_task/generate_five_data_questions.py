import os
import json
import random
from PIL import Image

def get_last_question_id(jsonl_path):
    """从JSONL文件中获取最后一个question_id。"""
    last_id = -1
    if os.path.exists(jsonl_path):
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            if lines:
                try:
                    last_entry = json.loads(lines[-1])
                    if 'question_id' in last_entry:
                        last_id = last_entry['question_id']
                except json.JSONDecodeError:
                    print(f"警告: 无法解析文件 '{jsonl_path}' 的最后一行。将从ID 0开始。")
    return last_id

def generate_wrong_bbox(correct_bbox, image_shape):
    """生成错误的边界框坐标作为干扰项。"""
    wrong_bboxes = []
    height, width = image_shape
    
    # 生成3个不与正确答案重叠的错误边界框
    for _ in range(3):
        # 随机生成边界框
        w_rand = random.randint(20, width // 4)
        h_rand = random.randint(20, height // 4)
        left = random.randint(0, width - w_rand)
        top = random.randint(0, height - h_rand)
        
        wrong_bbox = [top, left, top + h_rand, left + w_rand]
        
        # 简单的重叠检查（IoU > 0）
        # (x1, y1, x2, y2)
        boxA = [correct_bbox[1], correct_bbox[0], correct_bbox[3], correct_bbox[2]]
        boxB = [wrong_bbox[1], wrong_bbox[0], wrong_bbox[3], wrong_bbox[2]]
        
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        
        if interArea == 0: # 确保不重叠
            wrong_bboxes.append(wrong_bbox)

    # 如果生成的数量不够，用完全随机的补充
    while len(wrong_bboxes) < 3:
        w_rand = random.randint(20, width // 4)
        h_rand = random.randint(20, height // 4)
        left = random.randint(0, width - w_rand)
        top = random.randint(0, height - h_rand)
        wrong_bboxes.append([top, left, top + h_rand, left + w_rand])

    return wrong_bboxes[:3]

def generate_new_questions_from_public_data():
    """
    从处理好的公共息肉数据集中生成visual_grounding和lesion_quantification问题，
    并追加到现有的JSONL文件中。
    """
    annotations_path = '/path/to/visual_grounding_annotations.json'
    output_jsonl_path = '/path/to/endoscopy_mcq_all_questions.jsonl'

    # 1. 加载标注文件
    try:
        with open(annotations_path, 'r') as f:
            annotations = json.load(f)
    except FileNotFoundError:
        print(f"错误: 标注文件未找到于 '{annotations_path}'")
        return

    # 2. 获取起始 question_id
    start_id = get_last_question_id(output_jsonl_path) + 1
    print(f"将从 question_id {start_id} 开始生成新条目。")

    new_entries = []
    current_id = start_id
    
    # 用于最终报告的统计字典
    stats = {}

    # 3. 遍历标注，生成问题
    print("开始根据标注文件生成问题...")
    for image_path, bboxes in annotations.items():
        if not os.path.exists(image_path):
            continue

        num_boxes = len(bboxes)
        if num_boxes == 0:
            continue

        # 从路径中提取数据集名称作为数据源
        # e.g., /data/.../CVC-300/images/1.png -> CVC-300
        data_source = os.path.basename(os.path.dirname(os.path.dirname(image_path)))
        
        # 初始化统计
        if data_source not in stats:
            stats[data_source] = {"visual_grounding": 0, "lesion_quantification": 0}

        # ==================================================
        # Case 1: 单个目标 -> Visual Grounding 任务
        # ==================================================
        if num_boxes == 1:
            task_type = "visual_grounding"
            stats[data_source][task_type] += 1
            
            correct_bbox = bboxes[0]
            
            try:
                with Image.open(image_path) as img:
                    img_width, img_height = img.size
            except Exception as e:
                print(f"警告: 无法打开图像 {image_path} 来获取尺寸. 跳过. 错误: {e}")
                continue

            question = random.choice([
                "Which of the following bounding box coordinates correctly identifies the lesion location in this image?",
                "Select the correct bounding box coordinates [top, left, bottom, right] for the lesion in this endoscopic image:",
                "Which bounding box best localizes the abnormal area in this image?"
            ])
            
            wrong_bboxes = generate_wrong_bbox(correct_bbox, (img_height, img_width))
            
            all_options = [correct_bbox] + wrong_bboxes
            random.shuffle(all_options)
            
            correct_answer_idx = all_options.index(correct_bbox)
            correct_answer_char = chr(65 + correct_answer_idx)
            
            options_text = "Options:\n"
            answer_choices = {}
            for i, bbox in enumerate(all_options):
                option_key = chr(65 + i)
                bbox_str = f"[top={bbox[0]}, left={bbox[1]}, bottom={bbox[2]}, right={bbox[3]}]"
                options_text += f"{option_key}. {bbox_str}\n"
                answer_choices[option_key] = bbox
            options_text += "Please select the correct answer from the options above."

            entry = {
                "question_id": current_id,
                "task_type": task_type,
                "image": os.path.basename(image_path),
                "image_path": image_path,
                "question": question,
                "options": options_text,
                "category": "polyp",
                "correct_answer": correct_answer_char,
                "correct_bbox": correct_bbox,
                "answer_choices": answer_choices,
                "data_source": data_source
            }
            new_entries.append(entry)
            current_id += 1

        # ==================================================
        # Case 2: 多个目标 -> Lesion Quantification 任务
        # ==================================================
        else: # num_boxes > 1
            task_type = "lesion_quantification"
            stats[data_source][task_type] += 1

            correct_count = num_boxes
            # 我们只为数量在2,3,4之间的问题创建选项
            if correct_count not in [2, 3, 4]:
                continue

            question = "How many polyps are in the image?"
            answer_map = {2: 'B', 3: 'C', 4: 'D'}
            answer_choices = {'A': 1, 'B': 2, 'C': 3, 'D': 4}
            options_text = "Options:\nA. 1\nB. 2\nC. 3\nD. 4\nPlease select the correct answer from the options above."

            entry = {
                "question_id": current_id,
                "task_type": task_type,
                "image": os.path.basename(image_path),
                "image_path": image_path,
                "question": question,
                "options": options_text,
                "category": "polyp",
                "correct_answer": answer_map.get(correct_count),
                "answer_choices": answer_choices,
                "data_source": data_source
            }
            new_entries.append(entry)
            current_id += 1

    # 4. 将新条目追加到JSONL文件
    with open(output_jsonl_path, 'a', encoding='utf-8') as f:
        for entry in new_entries:
            f.write(json.dumps(entry) + '\n')

    print(f"\n成功生成并追加了 {len(new_entries)} 条新记录到 '{output_jsonl_path}'。")

    # 5. 打印统计报告
    print("\n--- 数据集分配统计报告 ---")
    total_vg = 0
    total_lq = 0
    for source, task_counts in stats.items():
        vg_count = task_counts.get("visual_grounding", 0)
        lq_count = task_counts.get("lesion_quantification", 0)
        total_vg += vg_count
        total_lq += lq_count
        print(f"数据集: {source}")
        print(f"  - Visual Grounding (单目标): {vg_count} 条")
        print(f"  - Lesion Quantification (多目标): {lq_count} 条")
    print("---------------------------------")
    print(f"总计 Visual Grounding: {total_vg} 条")
    print(f"总计 Lesion Quantification: {total_lq} 条")
    print(f"总计新增条目: {total_vg + total_lq} 条")


if __name__ == "__main__":
    generate_new_questions_from_public_data()