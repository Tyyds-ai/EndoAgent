import os
import base64
import json
from typing import Dict, Any, List
from dotenv import load_dotenv
from openai import OpenAI

class ResultEnhancer:
    """结果增强器：利用 GPT-4o 对工具输出结果进行综合增强"""
    
    def __init__(self, api_key=None, base_url=None, model="gpt-4o"):
        """初始化结果增强器"""
        _ = load_dotenv()  # 加载环境变量
        
        self.client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        )
        self.model = model
        # 添加结果缓存，避免重复增强
        self.enhancement_cache = {}

    @staticmethod
    def encode_image(image_path: str) -> str:
        """将图像编码为 base64 字符串"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图像文件不存在: {image_path}")
            
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def _convert_to_text(self, data):
        """将工具结果转换为文本格式"""
        if isinstance(data, str):
            return data
        elif isinstance(data, dict):
            try:
                return json.dumps(data, ensure_ascii=False, indent=2)
            except:
                return str(data)
        else:
            return str(data)
            
    def _extract_tool_info(self, tool_outputs):
        """从工具输出中提取关键信息"""
        tool_summary = []
        
        for tool_output in tool_outputs:
            tool_name = tool_output.get("tool_name", "未知工具")
            result = tool_output.get("result", {})
            
            # 提取不同工具的关键结果
            if "classification" in tool_name.lower():
                if isinstance(result, dict):
                    class_info = f"分类结果: {result.get('class_name', '未知')}"
                    confidence = result.get('confidence', 0)
                    description = result.get('description', '')
                    tool_summary.append(f"【病变分类工具】\n{class_info} (置信度: {confidence:.2f})\n{description}")
            
            elif "detection" in tool_name.lower():
                if isinstance(result, dict):
                    objects = result.get('objects_detected', 0)
                    description = result.get('description', '')
                    tool_summary.append(f"【病变检测工具】\n检测到 {objects} 个目标\n{description}")
            
            elif "segmentation" in tool_name.lower():
                if isinstance(result, dict):
                    lesion_percent = result.get('lesion_area_percentage', 0)
                    num_lesions = result.get('num_lesions', 0)
                    description = result.get('description', '')
                    tool_summary.append(f"【病变分割工具】\n病变区域百分比: {lesion_percent:.2f}%\n检测病变数: {num_lesions}\n{description}")
            
            elif "vqa" in tool_name.lower():
                if isinstance(result, dict):
                    response = result.get('result', result.get('description', ''))
                    vqa_question = result.get('question', '')
                    tool_summary.append(f"【内窥镜问答工具】\n问题: {vqa_question}\n回答: {response}")
            
            elif "report" in tool_name.lower():
                if isinstance(result, dict):
                    report = result.get('report', '')
                    tool_summary.append(f"【报告生成工具】\n{report}")
            
            else:
                # 通用工具结果处理
                result_str = self._convert_to_text(result)
                tool_summary.append(f"【{tool_name}】\n{result_str}")
                
        return tool_summary

    def enhance_all_results(self, 
                           question: str, 
                           image_path: str, 
                           agent_responses: List[str], 
                           tool_outputs: List[Dict], 
                           max_tokens: int = 2048) -> str:
        """综合增强所有工具输出和Agent回应
        
        Args:
            question: 用户原始问题
            image_path: 图像路径
            agent_responses: Agent的所有响应列表
            tool_outputs: 工具输出列表，每项包含tool_name和result
            max_tokens: 最大输出token数
            
        Returns:
            str: 增强后的综合结果
        """
        if not os.path.exists(image_path):
            return "⚠️ 无法访问图像文件，无法生成增强分析结果。"
        
        # 生成缓存键 - 基于问题、图像和工具输出
        cache_key = f"{hash(question)}_{hash(image_path)}_{hash(str(tool_outputs))}_comprehensive"
        
        # 检查缓存
        if cache_key in self.enhancement_cache:
            return self.enhancement_cache[cache_key]
        
        try:
            # 提取工具输出概要
            tool_summary = self._extract_tool_info(tool_outputs)
            
            # 整合所有工具结果
            all_results = "\n\n====================\n\n".join(tool_summary)
            
            # 整合Agent响应
            agent_summary = "\n\n".join([r for r in agent_responses if r]) if agent_responses else ""
            
            # 构建系统提示
            system_prompt = """You are an expert endoscopy analysis assistant that helps gastroenterologists interpret endoscopic images.
You have specialized knowledge in detecting, classifying, and analyzing gastrointestinal lesions including normal tissue, polyps, adenomas and cancerous lesions.
Provide detailed, medically accurate responses for endoscopic image analysis questions.
Answer based on the visible evidence in the picture, the Agent response, and the results of the tool analysis, with accurate descriptions."""

            # 构建用户提示
            user_prompt = f"""
Original Question:
{question}

Agent Responses:
{agent_summary}

Tool Analysis Results:
{all_results}

"""

            # 编码图像
            base64_image = self.encode_image(image_path)
            
            # 调用GPT-4o进行综合增强
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]}
                ],
                temperature=0.5,
                max_tokens=max_tokens
            )
            
            enhanced_result = response.choices[0].message.content
            
            # 添加标题，让结果更明显
            final_result = f"# 🔬 增强分析结果\n\n{enhanced_result}"
            
            # 缓存结果
            self.enhancement_cache[cache_key] = final_result
            
            return final_result
            
        except Exception as e:
            error_msg = f"综合增强结果时出错: {str(e)}"
            print(error_msg)
            return f"⚠️ {error_msg}\n请参考上述工具的原始分析结果。"