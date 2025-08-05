import pandas as pd
import json
import os
import glob
import random
from collections import defaultdict

def generate_sunseg_quantification_questions():
    """
    从SUN-SEG-Annotation数据集中生成息肉量化问题，并追加到现有的JSONL文件中。
    """
    # 定义文件路径
    base_paths = [
        'path/to/Public_Data/SUN-SEG-Annotation/TestEasyDataset',
        'path/to/Public_Data/SUN-SEG-Annotation/TestHardDataset'
    ]
    output_jsonl_path = 'path/to/endoscopy_mcq_all_questions.jsonl'
    
    # 数据集子目录
    datasets = ['Seen', 'Unseen']
    
    # 1. 检查并获取最后一个 question_id
    last_id = -1
    if os.path.exists(output_jsonl_path):
        with open(output_jsonl_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            if lines:
                # 读取最后一行并解析JSON
                try:
                    last_entry = json.loads(lines[-1])
                    if 'question_id' in last_entry:
                        last_id = last_entry['question_id']
                except json.JSONDecodeError:
                    print(f"警告: 无法解析文件 '{output_jsonl_path}' 的最后一行。将从ID 0开始。")
    
    start_id = last_id + 1
    print(f"将从 question_id {start_id} 开始生成新条目。")

    # 2. 处理每个数据集
    all_entries = []
    current_id = start_id
    
    for base_path in base_paths:
        base_name = os.path.basename(base_path)  # TestEasyDataset 或 TestHardDataset
        print(f"\n处理 {base_name}...")
        
        for dataset in datasets:
            print(f"\n处理 {base_name}/{dataset} 数据集...")
            
            # 构建路径
            frame_path = os.path.join(base_path, dataset, 'Frame')
            detection_json_path = os.path.join(base_path, dataset, 'Detection', 'bbox_annotation.json')
            
            # 检查路径是否存在
            if not os.path.exists(frame_path):
                print(f"警告: Frame路径不存在: {frame_path}")
                continue
            
            if not os.path.exists(detection_json_path):
                print(f"警告: Detection JSON不存在: {detection_json_path}")
                continue
            
            # 3. 读取COCO格式的bbox标注文件
            try:
                with open(detection_json_path, 'r', encoding='utf-8') as f:
                    coco_data = json.load(f)
            except Exception as e:
                print(f"错误: 无法读取bbox文件 {detection_json_path}: {e}")
                continue
            
            # 4. 解析COCO格式数据
            images_info = {}  # image_id -> image_info
            annotations_by_image = defaultdict(list)  # image_id -> list of annotations
            
            # 解析images信息
            if 'images' in coco_data:
                for img_data in coco_data['images']:
                    images_info[img_data['id']] = img_data
            
            # 解析annotations信息（注意：键名是 'annotation' 不是 'annotations'）
            if 'annotation' in coco_data:
                for ann_data in coco_data['annotation']:
                    image_id = ann_data['id']  # 这里直接用 'id' 字段
                    annotations_by_image[image_id].append(ann_data)
            
            print(f"找到 {len(images_info)} 个图像，{len(annotations_by_image)} 个图像有标注")
            
            # 5. 按病变数量进行分层采样
            print(f"开始按病变数量分层采样...")
            
            # 收集所有图像的病变数量信息
            image_bbox_counts = {}
            for image_id, annotations in annotations_by_image.items():
                image_bbox_counts[image_id] = len(annotations)
            
            # 按病变数量分组
            images_by_bbox_count = defaultdict(list)
            for image_id, bbox_count in image_bbox_counts.items():
                if bbox_count > 0:  # 只考虑有病变的图像
                    images_by_bbox_count[bbox_count].append(image_id)
            
            # 打印各病变数量的图像数
            for bbox_count, image_list in sorted(images_by_bbox_count.items()):
                print(f"  病变数量 {bbox_count}: {len(image_list)} 个图像")
            
            # 新的采样和处理策略：确保凑满目标数量
            target_counts = {}  # 每种病变数量的目标数量
            for bbox_count, image_list in images_by_bbox_count.items():
                if bbox_count <= 4:  # 只处理1-4个病变
                    if bbox_count == 2:
                        # 2个病变的图像：全部保存
                        target_counts[bbox_count] = len(image_list)
                        print(f"  病变数量 {bbox_count}: 目标 {len(image_list)} 个图像（全部）")
                    elif bbox_count == 1:
                        # 1个病变的图像：最多300张
                        target_counts[bbox_count] = min(300, len(image_list))
                        print(f"  病变数量 {bbox_count}: 目标 {target_counts[bbox_count]} 个图像（从 {len(image_list)} 个中选择）")
                    else:
                        # 3个和4个病变的图像：全部保存（如果有的话）
                        target_counts[bbox_count] = len(image_list)
                        print(f"  病变数量 {bbox_count}: 目标 {len(image_list)} 个图像（全部）")
            
            # 按病变数量处理，确保凑满目标数量
            processed_count = 0
            found_count = 0
            not_found_count = 0
            
            for bbox_count, target_count in target_counts.items():
                if target_count == 0:
                    continue
                    
                print(f"\n处理病变数量 {bbox_count} 的图像，目标: {target_count} 个")
                image_list = images_by_bbox_count[bbox_count]
                random.shuffle(image_list)  # 随机打乱顺序
                
                collected_count = 0
                attempted_count = 0
                
                for image_id in image_list:
                    if collected_count >= target_count:
                        break  # 已经收集够了
                    
                    attempted_count += 1
                    info = images_info[image_id]
                    file_name = info['file_name']
                    annotation_count = len(annotations_by_image.get(image_id, []))
                    
                    # 在Frame目录下查找对应的图像文件
                    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
                    image_path = None
                    
                    # 首先尝试直接在Frame目录下查找
                    for ext in image_extensions:
                        possible_names = [
                            f"{file_name}{ext}",
                            f"{image_id}{ext}",
                            file_name  # 如果file_name已经包含扩展名
                        ]
                        
                        for name in possible_names:
                            direct_path = os.path.join(frame_path, name)
                            if os.path.exists(direct_path):
                                image_path = direct_path
                                break
                        
                        if image_path:
                            break
                    
                    # 如果直接查找失败，遍历Frame下的所有子文件夹
                    if not image_path:
                        for root, dirs, files in os.walk(frame_path):
                            for ext in image_extensions:
                                # 尝试不同的文件名匹配方式
                                possible_names = [
                                    f"{file_name}{ext}",
                                    f"{image_id}{ext}",
                                    file_name  # 如果file_name已经包含扩展名
                                ]
                                
                                for name in possible_names:
                                    if name in files:
                                        image_path = os.path.join(root, name)
                                        break
                                
                                if image_path:
                                    break
                            
                            if image_path:
                                break
                    
                    if not image_path:
                        not_found_count += 1
                        if not_found_count <= 5:  # 只打印前5个找不到的文件
                            print(f"警告: 找不到图像文件 {file_name} (ID: {image_id})")
                        continue  # 跳过这个文件，继续尝试下一个
                    
                    # 找到文件了，生成问答条目
                    found_count += 1
                    collected_count += 1
                    processed_count += 1
                    
                    if processed_count % 50 == 0:
                        print(f"已处理 {processed_count} 个图像（找到 {found_count}，未找到 {not_found_count}）...")
                    
                    # 6. 生成问答条目
                    # 只处理有标注的图像（即有病变的图像）
                    if annotation_count > 0 and annotation_count <= 4:  # 限制在1-4个病变
                        # 定义选项和答案映射
                        answer_map = {1: 'A', 2: 'B', 3: 'C', 4: 'D'}
                        answer_choices = {'A': 1, 'B': 2, 'C': 3, 'D': 4}
                        options_text = "Options:\nA. 1\nB. 2\nC. 3\nD. 4\nPlease select the correct answer from the options above."
                        
                        # 提取bbox信息
                        bboxes = []
                        for ann in annotations_by_image.get(image_id, []):
                            if 'bbox' in ann:
                                bboxes.append(ann['bbox'])
                        
                        entry = {
                            "question_id": current_id,
                            "task_type": "lesion_quantification",
                            "image": os.path.basename(image_path),
                            "image_path": image_path,
                            "question": "How many polyps are in the image?",
                            "options": options_text,
                            "category": "polyp",
                            "correct_answer": answer_map[annotation_count],
                            "answer_choices": answer_choices,
                            "data_source": f"SUN-SEG-{base_name}-{dataset}",
                            "metadata": {
                                "original_id": image_id,
                                "file_name": file_name,
                                "width": info.get('width', 0),
                                "height": info.get('height', 0),
                                "annotation_count": annotation_count,
                                "bboxes": bboxes
                            }
                        }
                        
                        all_entries.append(entry)
                        current_id += 1
                        
                        if len(all_entries) % 100 == 0:
                            print(f"已生成 {len(all_entries)} 个条目...")
                    
                    elif annotation_count > 4:
                        print(f"跳过图像 {image_id}: 病变数量过多 ({annotation_count})")
                
                print(f"  病变数量 {bbox_count}: 目标 {target_count}，尝试 {attempted_count}，成功 {collected_count}")
                
                # 如果没有收集够，给出警告
                if collected_count < target_count:
                    shortage = target_count - collected_count
                    print(f"  ⚠️  病变数量 {bbox_count} 未达到目标，缺少 {shortage} 个（可能是文件缺失）")
            
            print(f"本数据集处理完成：总尝试 {found_count + not_found_count}，找到图像 {found_count}，未找到图像 {not_found_count}，生成条目 {len([e for e in all_entries if e['data_source'] == f'SUN-SEG-{base_name}-{dataset}'])}")

    # 7. 输出统计信息
    print(f"\n=== 数据统计 ===")
    print(f"总共生成 {len(all_entries)} 个问答条目")
    
    # 按病变数量统计
    count_stats = defaultdict(int)
    for entry in all_entries:
        count = entry['metadata']['annotation_count']
        count_stats[count] += 1
    
    for count, num in sorted(count_stats.items()):
        print(f"病变数量 {count}: {num} 个条目")
    
    # 按数据源统计
    source_stats = defaultdict(int)
    for entry in all_entries:
        source = entry['data_source']
        source_stats[source] += 1
    
    for source, num in sorted(source_stats.items()):
        print(f"数据源 {source}: {num} 个条目")

    # 8. 将新条目追加到JSONL文件
    if all_entries:
        with open(output_jsonl_path, 'a', encoding='utf-8') as f:
            for entry in all_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        print(f"\n成功生成并追加了 {len(all_entries)} 条新记录到 '{output_jsonl_path}'。")
    else:
        print("\n没有生成任何条目。")

def save_image_paths_json():
    """
    保存所有图像路径到JSON文件，用于调试和检查
    """
    base_paths = [
        '/path/to/SUN-SEG-Annotation/TestEasyDataset',
        '/path/to/SUN-SEG-Annotation/TestHardDataset'
    ]
    output_json_path = '/path/to/sunseg_image_paths.json'
    
    datasets = ['Seen', 'Unseen']
    all_images = {}
    
    for base_path in base_paths:
        base_name = os.path.basename(base_path)
        all_images[base_name] = {}
        
        for dataset in datasets:
            frame_path = os.path.join(base_path, dataset, 'Frame')
            detection_json_path = os.path.join(base_path, dataset, 'Detection', 'bbox_annotation.json')
            
            if not os.path.exists(frame_path) or not os.path.exists(detection_json_path):
                continue
            
            # 读取COCO格式的bbox数据
            try:
                with open(detection_json_path, 'r', encoding='utf-8') as f:
                    coco_data = json.load(f)
            except:
                continue
            
            # 组织数据
            images_info = {}
            annotations_by_image = defaultdict(list)
            
            if 'images' in coco_data:
                for img_data in coco_data['images']:
                    images_info[img_data['id']] = img_data
            
            # 解析annotations信息（注意：键名是 'annotation' 不是 'annotations'）
            if 'annotation' in coco_data:
                for ann_data in coco_data['annotation']:
                    image_id = ann_data['id']  # 这里直接用 'id' 字段
                    annotations_by_image[image_id].append(ann_data)
            
            # 查找图像文件，但使用更高效的方法
            dataset_images = {}
            processed_ids = 0
            
            # 随机采样图像ID以减少处理时间
            all_image_ids = list(images_info.keys())
            if len(all_image_ids) > 50:  # 每个数据集最多处理50个图像
                sampled_image_ids = random.sample(all_image_ids, 50)
                print(f"从 {len(all_image_ids)} 个图像中随机采样了 {len(sampled_image_ids)} 个")
            else:
                sampled_image_ids = all_image_ids
            
            for image_id in sampled_image_ids:
                if image_id not in images_info:
                    continue
                    
                info = images_info[image_id]
                file_name = info['file_name']
                annotation_count = len(annotations_by_image.get(image_id, []))
                
                # 查找图像文件，使用更高效的方法
                image_path = None
                
                # 首先尝试直接匹配
                for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                    possible_paths = [
                        os.path.join(frame_path, f"{file_name}{ext}"),
                        os.path.join(frame_path, f"{image_id}{ext}"),
                        os.path.join(frame_path, file_name)
                    ]
                    
                    for path in possible_paths:
                        if os.path.exists(path):
                            image_path = path
                            break
                    
                    if image_path:
                        break
                
                # 如果直接匹配失败，尝试在子目录中查找（限制深度）
                if not image_path:
                    for root, dirs, files in os.walk(frame_path):
                        # 限制搜索深度，避免过深的递归
                        level = root.replace(frame_path, '').count(os.sep)
                        if level >= 3:  # 最多搜索3层深度
                            dirs[:] = []  # 不再深入搜索
                            continue
                            
                        # 随机采样文件以减少搜索时间
                        if len(files) > 20:
                            files = random.sample(files, 20)
                        
                        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                            possible_names = [f"{file_name}{ext}", f"{image_id}{ext}", file_name]
                            for name in possible_names:
                                if name in files:
                                    image_path = os.path.join(root, name)
                                    break
                            if image_path:
                                break
                        if image_path:
                            break
                
                if image_path:
                    # 提取bbox信息
                    bboxes = []
                    for ann in annotations_by_image.get(image_id, []):
                        if 'bbox' in ann:
                            bboxes.append(ann['bbox'])
                    
                    dataset_images[image_id] = {
                        'file_name': file_name,
                        'image_path': image_path,
                        'annotation_count': annotation_count,
                        'bboxes': bboxes,
                        'width': info.get('width', 0),
                        'height': info.get('height', 0)
                    }
                    
                    processed_ids += 1
                    if processed_ids % 10 == 0:
                        print(f"已处理 {processed_ids} 个图像...")
                        
                    # 限制每个数据集的图像数量
                    if processed_ids >= 50:
                        break
            
            all_images[base_name][dataset] = dataset_images
    
    # 保存到JSON文件
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(all_images, f, ensure_ascii=False, indent=2)
    
    print(f"图像路径信息保存到: {output_json_path}")
    
    # 打印统计信息
    for base_name, base_data in all_images.items():
        for dataset, images in base_data.items():
            print(f"{base_name}/{dataset}: {len(images)} 个图像")

if __name__ == "__main__":
    print("=== 生成SUN-SEG息肉量化问题 ===")
    
    # 首先保存图像路径信息用于调试
    print("1. 保存图像路径信息...")
    save_image_paths_json()
    
    print("\n2. 生成问答条目...")
    generate_sunseg_quantification_questions()
