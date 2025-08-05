import os
import sys
import json
import argparse
import time
import random
import base64
from tqdm import tqdm
from pprint import pprint
from collections import defaultdict
from dotenv import load_dotenv
import asyncio
import logging
import warnings
import traceback
import requests
from retry import retry

# 屏蔽不必要的警告
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

# 加载环境变量
_ = load_dotenv()

# 初始化OpenAI客户端（兼容DashScope）
try:
    from openai import OpenAI
    from openai import AsyncOpenAI
    
    client = OpenAI(
        api_key=os.environ.get("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    
    async_client = AsyncOpenAI(
        api_key=os.environ.get("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
except ImportError:
    print("请安装OpenAI库: pip install openai")
    sys.exit(1)

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
        return 'jpeg'

class QwenEvalPromptGenerator:
    """生成用于Qwen-VL-Plus评估回答的提示"""
    
    # VQA任务评估提示词
    vqa_evaluation_prompt = """You are an expert endoscopy analysis assistant acting as a judge to evaluate the quality of two AI responses to an endoscopic image analysis question.

## Task: Visual Question Answering (VQA)
Your role is to evaluate which response better answers the specific question about this endoscopic image.

## Image Information:
- **True Category**: {category}
- **Question**: {question}

## Evaluation Criteria:
1. **Answer Accuracy**: How well does the response directly answer the specific question asked?
2. **Category Recognition**: Does the response correctly identify the lesion type ({category})?
3. **Visual Evidence**: Does the response mention specific visual features that support the answer?
4. **Medical Accuracy**: Are the medical terms and descriptions accurate?
5. **Relevance**: How relevant is the response to the specific question?
6. **Completeness**: Does the response provide sufficient detail to be helpful?

## Scoring Instructions:
- Give scores from 1-10 for each response (higher = better quality)
- Consider that the true lesion category is **{category}**
- Do not give higher scores simply because a response is longer or more verbose. Focus on quality, accuracy, and relevance, not length.
- Responses that correctly identify this category should score higher
- Focus on how well each response answers the specific question
- Consider both accuracy and usefulness of the information provided

Please first output two numbers separated by a space (scores for Response 1 and Response 2), then provide detailed evaluation explaining your scoring decision.

## Response 1:
{answer1}

## Response 2:
{answer2}
"""

    # MRG任务评估提示词
    mrg_evaluation_prompt = """You are an expert gastroenterologist acting as a judge to evaluate the quality of two AI-generated medical reports for an endoscopic image.

## Task: Medical Report Generation (MRG)
Your role is to evaluate which report provides better clinical documentation and analysis.

## Image Information:
- **True Category**: {category}
- **Report Request**: {question}

## Evaluation Criteria:
1. **Diagnostic Accuracy**: Does the report correctly identify the lesion type ({category})?
2. **Clinical Structure**: Is the report properly structured with appropriate medical sections?
3. **Medical Terminology**: Are medical terms used correctly and appropriately?
4. **Detailed Description**: Does the report provide adequate description of visible findings?
5. **Clinical Significance**: Does the report explain the clinical importance of findings?
6. **Recommendations**: Are appropriate follow-up or treatment recommendations provided?
7. **Professional Quality**: Would this report meet clinical documentation standards?

## Scoring Instructions:
- Give scores from 1-10 for each report (higher = better quality)
- Consider that the true lesion category is **{category}**
- Do not give higher scores simply because a response is longer or more verbose. Focus on quality, accuracy, and relevance, not length.
- Reports that correctly identify and address this category should score higher
- Evaluate both the accuracy and clinical utility of each report
- Consider completeness, clarity, and professional presentation

Please first output two numbers separated by a space (scores for Report 1 and Report 2), then provide detailed evaluation explaining your scoring decision.

## Report 1:
{answer1}

## Report 2:
{answer2}
"""

    @staticmethod
    def generate_evaluation_prompt(task_type, category, question, answer1, answer2):
        """生成评估提示词"""
        if task_type == "vqa":
            return QwenEvalPromptGenerator.vqa_evaluation_prompt.format(
                category=category,
                question=question,
                answer1=answer1,
                answer2=answer2
            )
        elif task_type == "mrg":
            return QwenEvalPromptGenerator.mrg_evaluation_prompt.format(
                category=category,
                question=question,
                answer1=answer1,
                answer2=answer2
            )
        else:
            raise ValueError(f"Unsupported task type: {task_type}")

    @staticmethod
    def generate_messages(sample):
        """生成用于比较的消息"""
        # 英文版类别映射
        category_map = {
            "normal": "Normal Colonic Mucosa",
            "polyp": "Polyp",
            "adenoma": "Adenoma", 
            "cancer": "Cancerous Lesion"
        }
        
        category = category_map.get(sample['category'], sample['category'])
        task_type = sample['task_type']
        
        # 随机决定答案顺序，防止顺序偏见
        if random.random() > 0.5:
            answer1, answer2 = sample['qwen_answer'], sample['answer_endoagent_reflexion']
            sample['ans1_type'] = 'Qwen-VL-Plus'
            sample['ans2_type'] = 'EndoAgent'
        else:
            answer1, answer2 = sample['answer_endoagent_reflexion'], sample['qwen_answer']
            sample['ans1_type'] = 'EndoAgent'
            sample['ans2_type'] = 'Qwen-VL-Plus'
        
        sample['ans1'], sample['ans2'] = answer1, answer2
        
        # 生成评估提示词
        evaluation_prompt = QwenEvalPromptGenerator.generate_evaluation_prompt(
            task_type, category, sample['question'], answer1, answer2
        )
        
        # 编码图像
        image_path = sample.get('image_path')
        if image_path and os.path.exists(image_path):
            base64_image = encode_image(image_path)
            image_format = get_image_format(image_path)
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/{image_format};base64,{base64_image}"}
                        },
                        {
                            "type": "text",
                            "text": evaluation_prompt
                        }
                    ]
                }
            ]
        else:
            # 如果没有图像，只使用文本
            messages = [
                {
                    "role": "user", 
                    "content": evaluation_prompt
                }
            ]
        
        return messages


class EndoEvaluation:
    """评估器类，负责评估回答质量"""
    
    @staticmethod
    def get_category(x):
        """获取样本类别"""
        return x.get('category', 'unknown')
    
    @staticmethod
    def get_type(x):
        """获取样本任务类型"""
        return x.get('task_type', 'unknown')
    
    @staticmethod
    def get_avg(x):
        """计算平均值"""
        return sum([float(y) for y in x]) / len(x)
    
    @staticmethod
    def find_score_in_response(response):
        """从响应中提取分数"""
        # 首先尝试从第一行获取分数
        first_line = response.strip().split('\n')[0]
        scores = []
        
        # 尝试找出包含两个数字的部分
        for part in first_line.split():
            try:
                score = float(part)
                if 0 <= score <= 10:  # 分数应在1-10之间
                    scores.append(score)
            except ValueError:
                continue
        
        # 如果找到两个分数，返回它们
        if len(scores) == 2:
            return scores
        
        # 如果第一行没找到，尝试在整个响应中查找
        import re
        score_pattern = r'(\d+(\.\d+)?)\s+(\d+(\.\d+)?)'
        matches = re.search(score_pattern, response)
        
        if matches:
            try:
                score1 = float(matches.group(1))
                score2 = float(matches.group(3))
                return [score1, score2]
            except (ValueError, IndexError):
                pass
        
        # 如果仍然没找到，返回空值
        return None
    
    @staticmethod
    def eval(samples):
        """评估样本结果"""
        predictions = []
        for sample in samples:
            result_text = sample.get('result', '')
            scores = EndoEvaluation.find_score_in_response(result_text)
            
            if scores:
                # 根据回答类型调整分数顺序
                if sample.get('ans1_type') == 'Qwen-VL-Plus':
                    qwen_score, endoagent_score = scores
                else:
                    endoagent_score, qwen_score = scores
                
                predictions.append((
                    sample['question_id'],
                    sample['task_type'],
                    EndoEvaluation.get_category(sample),
                    [qwen_score, endoagent_score]
                ))
            else:
                print(f"警告: 问题 {sample['question_id']} 无法解析评分: {result_text[:50]}...")
        
        # 按任务类型和类别计算统计信息
        score_type_dict = defaultdict(lambda: defaultdict(list))
        score_category_dict = defaultdict(lambda: defaultdict(list))
        
        for q_id, q_type, category, (qwen_score, endoagent_score) in predictions:
            # 按任务类型统计
            score_type_dict[q_type]['qwen'].append(qwen_score)
            score_type_dict[q_type]['endoagent'].append(endoagent_score)
            score_type_dict['all']['qwen'].append(qwen_score)
            score_type_dict['all']['endoagent'].append(endoagent_score)
            
            # 按类别统计
            score_category_dict[category]['qwen'].append(qwen_score)
            score_category_dict[category]['endoagent'].append(endoagent_score)
        
        # 计算结果
        result_by_type = defaultdict(dict)
        result_by_category = defaultdict(dict)
        
        print("\n=== 按任务类型的评估结果 ===")
        for q_type, score_dict in score_type_dict.items():
            if not score_dict['qwen'] or not score_dict['endoagent']:
                continue
                
            qwen_avg = EndoEvaluation.get_avg(score_dict['qwen'])
            endoagent_avg = EndoEvaluation.get_avg(score_dict['endoagent'])
            relative_score = (endoagent_avg / qwen_avg) * 100 if qwen_avg > 0 else 0
            
            result_by_type[q_type] = {
                'qwen_score': qwen_avg,
                'endoagent_score': endoagent_avg,
                'relative_score': relative_score,
                'data_size': len(score_dict['qwen'])
            }
            
            print(f"类型: {q_type}")
            print(f"  Qwen-VL-Plus 平均分: {qwen_avg:.2f}")
            print(f"  EndoAgent 平均分: {endoagent_avg:.2f}")
            print(f"  相对分数: {relative_score:.2f}%")
            print(f"  样本数: {len(score_dict['qwen'])}")
            
            # 显示获胜次数
            win_count = sum(1 for ea, qwen in zip(score_dict['endoagent'], score_dict['qwen']) if ea > qwen)
            tie_count = sum(1 for ea, qwen in zip(score_dict['endoagent'], score_dict['qwen']) if ea == qwen)
            loss_count = sum(1 for ea, qwen in zip(score_dict['endoagent'], score_dict['qwen']) if ea < qwen)
            
            print(f"  EndoAgent 获胜: {win_count} ({win_count/len(score_dict['qwen'])*100:.2f}%)")
            print(f"  平局: {tie_count} ({tie_count/len(score_dict['qwen'])*100:.2f}%)")
            print(f"  Qwen-VL-Plus 获胜: {loss_count} ({loss_count/len(score_dict['qwen'])*100:.2f}%)")
            print("")
        
        print("\n=== 按类别的评估结果 ===")
        for category, score_dict in score_category_dict.items():
            if not score_dict['qwen'] or not score_dict['endoagent']:
                continue
                
            qwen_avg = EndoEvaluation.get_avg(score_dict['qwen'])
            endoagent_avg = EndoEvaluation.get_avg(score_dict['endoagent'])
            relative_score = (endoagent_avg / qwen_avg) * 100 if qwen_avg > 0 else 0
            
            result_by_category[category] = {
                'qwen_score': qwen_avg,
                'endoagent_score': endoagent_avg,
                'relative_score': relative_score,
                'data_size': len(score_dict['qwen'])
            }
            
            print(f"类别: {category}")
            print(f"  Qwen-VL-Plus 平均分: {qwen_avg:.2f}")
            print(f"  EndoAgent 平均分: {endoagent_avg:.2f}")
            print(f"  相对分数: {relative_score:.2f}%")
            print(f"  样本数: {len(score_dict['qwen'])}")
            
            # 显示获胜次数
            win_count = sum(1 for ea, qwen in zip(score_dict['endoagent'], score_dict['qwen']) if ea > qwen)
            tie_count = sum(1 for ea, qwen in zip(score_dict['endoagent'], score_dict['qwen']) if ea == qwen)
            loss_count = sum(1 for ea, qwen in zip(score_dict['endoagent'], score_dict['qwen']) if ea < qwen)
            
            print(f"  EndoAgent 获胜: {win_count} ({win_count/len(score_dict['qwen'])*100:.2f}%)")
            print(f"  平局: {tie_count} ({tie_count/len(score_dict['qwen'])*100:.2f}%)")
            print(f"  Qwen-VL-Plus 获胜: {loss_count} ({loss_count/len(score_dict['qwen'])*100:.2f}%)")
            print("")
        
        return {
            'by_type': result_by_type,
            'by_category': result_by_category
        }


# 使用retry装饰器处理API错误和重试
@retry(exceptions=(requests.exceptions.RequestException, Exception), 
       tries=4, delay=5, backoff=2, jitter=(1, 3))
async def call_qwen_async(messages, model="qwen-vl-plus"):
    """异步调用Qwen-VL-Plus API"""
    try:
        response = await async_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,
            max_tokens=1024,
            timeout=60
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"API调用异常: {e}")
        # 添加重试延迟
        await asyncio.sleep(5)
        raise e


async def process_batch_async(batch):
    """异步处理一批评估请求"""
    tasks = []
    for sample in batch:
        messages = QwenEvalPromptGenerator.generate_messages(sample)
        tasks.append(call_qwen_async(messages))
    
    # 等待所有任务完成
    responses = await asyncio.gather(*tasks, return_exceptions=True)
    
    # 更新样本中的结果
    results = []
    for sample, response in zip(batch, responses):
        if isinstance(response, Exception):
            print(f"警告: 问题 {sample['question_id']} 处理失败: {response}")
        elif response:
            sample_copy = sample.copy()
            sample_copy['result'] = response
            results.append(sample_copy)
        else:
            print(f"警告: 问题 {sample['question_id']} 返回空响应")
    
    return results


async def async_main():
    """异步主函数"""
    parser = argparse.ArgumentParser(description='使用Qwen-VL-Plus评估EndoAgent与Qwen-VL-Plus的回答质量')
    parser.add_argument('--input_path', type=str, default='data/endoscopy_language_tasks_with_endoagent_reflexion.jsonl',
                        help='包含两种回答的输入文件')
    parser.add_argument('--output_path', type=str, default='data/eval_qwen_endoagent_reflexion_results.jsonl',
                        help='包含评估结果的输出文件')
    parser.add_argument('--batch_size', type=int, default=3, help='批处理大小')
    parser.add_argument('--task_types', nargs='+', default=['vqa', 'mrg'], 
                        help='要评估的任务类型')
    args = parser.parse_args()
    
    # 确保API密钥存在
    if not os.environ.get("DASHSCOPE_API_KEY"):
        print("错误: 未设置DASHSCOPE_API_KEY环境变量")
        return
    
    # 加载输入数据
    print(f"加载输入文件: {args.input_path}")
    input_data = []
    try:
        with open(args.input_path, 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line)
                # 检查是否同时包含两种回答，且任务类型符合要求
                if ('qwen_answer' in entry and 'answer_endoagent_reflexion' in entry and 
                    entry.get('task_type') in args.task_types):
                    input_data.append(entry)
                else:
                    missing_fields = []
                    if 'qwen_answer' not in entry:
                        missing_fields.append('qwen_answer')
                    if 'answer_endoagent_reflexion' not in entry:
                        missing_fields.append('answer_endoagent_reflexion')
                    if entry.get('task_type') not in args.task_types:
                        missing_fields.append(f"task_type not in {args.task_types}")
                    print(f"警告: 跳过条目(ID: {entry.get('question_id')}), 缺少: {', '.join(missing_fields)}")
    except FileNotFoundError:
        print(f"错误: 输入文件未找到: {args.input_path}")
        return
    except json.JSONDecodeError as e:
        print(f"错误: 输入文件包含无效JSON: {e}")
        return
    
    print(f"成功加载 {len(input_data)} 条有效条目")
    
    # 按任务类型统计
    task_counts = defaultdict(int)
    for entry in input_data:
        task_counts[entry.get('task_type', 'unknown')] += 1
    
    print("任务类型分布:")
    for task_type, count in task_counts.items():
        print(f"  {task_type}: {count} 条")
    
    # 检查之前的输出文件以避免重复处理
    processed_ids = set()
    results = []
    if os.path.exists(args.output_path):
        try:
            with open(args.output_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        result = json.loads(line)
                        q_id = result.get('question_id')
                        if q_id is not None:
                            processed_ids.add(q_id)
                            results.append(result)
                    except json.JSONDecodeError:
                        continue
            print(f"已加载 {len(processed_ids)} 条已处理结果")
        except Exception as e:
            print(f"加载已处理结果时出错: {e}")
    
    # 筛选未处理的条目
    to_process = [entry for entry in input_data if entry.get('question_id') not in processed_ids]
    print(f"需要处理 {len(to_process)} 条新条目")
    
    if not to_process:
        print("没有未处理的条目，直接进行评估")
    else:
        # 分批处理
        batches = [to_process[i:i+args.batch_size] for i in range(0, len(to_process), args.batch_size)]
        print(f"将处理 {len(batches)} 批次，每批 {args.batch_size} 条")
        
        new_results = []
        for i, batch in enumerate(batches):
            try:
                print(f"处理批次 {i+1}/{len(batches)}...")
                batch_results = await process_batch_async(batch)
                new_results.extend(batch_results)
                
                # 实时保存所有结果
                all_results = results + new_results
                with open(args.output_path, 'w', encoding='utf-8') as f:
                    for result in all_results:
                        f.write(json.dumps(result, ensure_ascii=False) + '\n')
                
                # 添加一些延迟以避免API限制
                if i < len(batches) - 1:
                    await asyncio.sleep(2)
                    
            except Exception as e:
                print(f"处理批次 {i+1} 时出错: {e}")
                print(traceback.format_exc())
                # 保存已处理的结果
                all_results = results + new_results
                with open(args.output_path, 'w', encoding='utf-8') as f:
                    for result in all_results:
                        f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        # 更新结果列表
        results.extend(new_results)
        print(f"成功处理 {len(new_results)} 条新条目")
    
    # 评估结果
    print("\n开始评估结果...")
    eval_results = EndoEvaluation.eval(results)
    
    # 保存评估汇总结果
    summary_path = os.path.splitext(args.output_path)[0] + '_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(eval_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n评估汇总结果已保存至: {summary_path}")
    print("评估完成！")


def main():
    """同步主函数入口点"""
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"程序执行出错: {e}")
        print(traceback.format_exc())


if __name__ == "__main__":
    main()