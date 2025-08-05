import argparse
import json
import os
import time
from tqdm import tqdm
import shortuuid
import base64
import requests
import jsonlines
from pathlib import Path
from dotenv import load_dotenv
from retry import retry
import logging
import warnings

# 屏蔽不必要的警告
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

# 加载环境变量
_ = load_dotenv()

# 初始化OpenAI客户端（兼容DashScope）
try:
    from openai import OpenAI
    
    client = OpenAI(
        api_key=os.environ.get("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
except ImportError:
    print("请安装OpenAI库: pip install openai")
    exit(1)

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# 获取图像格式
def get_image_format(file_path):
    extension = os.path.splitext(file_path)[1].lower()
    if extension == '.png':
        return 'png'
    elif extension in ['.jpg', '.jpeg']:
        return 'jpeg'
    elif extension == '.webp':
        return 'webp'
    else:
        # 默认使用jpeg格式
        return 'jpeg'

# 使用retry装饰器处理API错误和重试
@retry(exceptions=(requests.exceptions.RequestException, Exception), 
       tries=4, delay=5, backoff=2, jitter=(1, 3))
def get_answer(api_key, question_id, question, image_path, max_tokens, category=None, task_type=None):
    """调用Qwen-VL-Max API获取回答"""
    try:
        if image_path is not None and os.path.exists(image_path):
            # 编码图像
            base64_image = encode_image(image_path)
            image_format = get_image_format(image_path)
            
            # 根据类别构建更具体的提示
            category_descriptions = {
                "normal": "normal tissue without any abnormalities",
                "polyp": "polyp (abnormal growth)",
                "adenoma": "adenoma (benign tumor)",
                "cancer": "cancer (malignant tumor)"
            }
            
            # 系统提示词 - 专为内窥镜分析定制
            system_message = """You are an expert endoscopy analysis assistant that helps gastroenterologists interpret endoscopic images.
You have specialized knowledge in detecting, classifying, and analyzing gastrointestinal lesions including normal tissue, polyp, adenoma and cancer.
Provide detailed, medically accurate responses for endoscopic image analysis questions.
Base your answers only on visible evidence in the image and be precise in your descriptions."""

            # 构建增强的问题提示
            enhanced_question = question
            if category:
                category_desc = category_descriptions.get(category, f"{category} findings")
                if task_type == "vqa":
                    enhanced_question = f"""This endoscopic image shows {category_desc}. 
Based on the image content and the fact that this shows {category_desc}, please answer the following question: {question}

Please provide a detailed and medically accurate answer based on what you can observe in the image."""
                elif task_type == "mrg":
                    enhanced_question = f"""This endoscopic image shows {category_desc}. 
Based on the image content and the fact that this shows {category_desc}, please generate a detailed medical report addressing the following: {question}

Please provide a comprehensive medical report that includes:
1. Description of visible findings
2. Clinical significance
3. Relevant diagnostic considerations"""

            # 使用OpenAI兼容模式调用千问大模型
            try:
                response = client.chat.completions.create(
                    model="qwen-vl-plus",
                    messages=[
                        {"role": "system", 
                         "content": [{"type": "text", "text": system_message}]},
                        {"role": "user", 
                         "content": [
                            {"type": "image_url", 
                             "image_url": {"url": f"data:image/{image_format};base64,{base64_image}"}},
                            {"type": "text", "text": enhanced_question}
                         ]}
                    ],
                    temperature=0.7,
                    max_tokens=max_tokens,
                    timeout=60  # 设置60秒超时
                )
                return response.choices[0].message.content
                
            except Exception as e:
                # 处理API错误
                print(f"DashScope API 错误: {e}")
                # 让retry装饰器处理重试
                raise e
                
        else:
            return "图像路径不存在或未提供图像路径"
            
    except Exception as e:
        print(f"处理图像或API调用时出错: {str(e)}")
        raise e


def main():
    parser = argparse.ArgumentParser(description='使用 Qwen-VL-Max 生成内窥镜图像问答')
    parser.add_argument('--api-key', required=False, default=os.environ.get("DASHSCOPE_API_KEY"),
                        help='您的 DashScope API 密钥 (默认使用环境变量)')
    parser.add_argument('--input', default="data/endoscopy_language_tasks.jsonl", 
                        help='包含问题的输入 JSONL 文件')
    parser.add_argument('--output', default="data/endoscopy_language_tasks_with_answers.jsonl", 
                        help='输出结果的 JSONL 文件')
    parser.add_argument('--max-tokens', type=int, default=1024, help='输出的最大 token 数')
    args = parser.parse_args()

    # 确保API密钥存在
    if not args.api_key:
        print("错误: 未提供DashScope API密钥，请通过--api-key参数或DASHSCOPE_API_KEY环境变量设置")
        return

    # 输出配置信息
    print(f"DashScope API URL: https://dashscope.aliyuncs.com/compatible-mode/v1")
    print(f"DashScope API Key: {'已设置' if args.api_key else '未设置'}")
    print(f"模型: qwen-vl-plus")

    # 确保输出目录存在
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    # 加载问题数据
    questions = []
    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            for line in f:
                questions.append(json.loads(line))
        print(f"成功加载 {len(questions)} 个问题")
    except Exception as e:
        print(f"加载问题文件时出错: {e}")
        return

    # 检查是否有已处理的结果，跳过已处理的问题
    processed_ids = set()
    results = []
    
    if os.path.exists(args.output):
        try:
            with open(args.output, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        result = json.loads(line)
                        if 'question_id' in result and 'qwen_answer' in result:
                            processed_ids.add(result['question_id'])
                            results.append(result)
                    except json.JSONDecodeError:
                        continue
            print(f"已加载 {len(processed_ids)} 条已处理结果")
        except Exception as e:
            print(f"加载已处理结果时出错: {e}")

    # 筛选未处理的问题
    questions_to_process = [q for q in questions if q.get('question_id') not in processed_ids]
    print(f"需要处理 {len(questions_to_process)} 个新问题")

    # 处理每个问题
    for item in tqdm(questions_to_process, desc="处理问题"):
        question_id = item.get('question_id', shortuuid.uuid())
        question = item.get('question', '')  # 使用question字段而不是pre_text
        
        # 获取图像路径 - 使用完整路径
        image_path = item.get('image_path')
        
        # 获取类别和任务类型信息
        category = item.get('category')
        task_type = item.get('task_type')
        
        if not image_path or not os.path.exists(image_path):
            print(f"警告: 图像路径不存在: {image_path}")
            continue  # 跳过没有有效图像的问题
        
        # 调用 API 获取回答
        try:
            start_time = time.time()
            answer = get_answer(args.api_key, question_id, question, image_path, args.max_tokens, category, task_type)
            processing_time = time.time() - start_time
            
            # 更新条目
            item_copy = item.copy()
            item_copy['qwen_answer'] = answer
            item_copy['qwen_processed_at'] = time.strftime("%Y-%m-%d %H:%M:%S")
            item_copy['qwen_processing_time'] = processing_time
            
            results.append(item_copy)
            
            # 实时保存结果
            with jsonlines.open(args.output, mode='w') as writer:
                writer.write_all(results)
                
            print(f"已完成问题 {question_id} (类别: {category}, 任务: {task_type}), 处理时间: {processing_time:.2f}秒")
            
            # 添加短暂间隔，避免API限制
            time.sleep(0.5)
            
        except Exception as e:
            print(f"处理问题 {question_id} 失败: {e}")
            
            # 记录失败信息
            item_copy = item.copy()
            item_copy['qwen_answer'] = f"处理失败: {str(e)}"
            item_copy['qwen_error'] = str(e)
            item_copy['qwen_processed_at'] = time.strftime("%Y-%m-%d %H:%M:%S")
            
            results.append(item_copy)
            
            # 实时保存结果
            with jsonlines.open(args.output, mode='w') as writer:
                writer.write_all(results)
                
            # 遇到错误时额外等待，避免持续错误
            time.sleep(5)

    print(f"处理完成，结果已保存到 {args.output}")

if __name__ == '__main__':
    main()