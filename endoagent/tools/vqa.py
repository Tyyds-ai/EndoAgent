import sys
# 确保能够导入ColonGPT模块
sys.path.append('/path/to/EndoAgent/endoagent/tools/Config/IntelliScope')

from typing import Dict, List, Optional, Tuple, Type, Any, ClassVar, Union
from pydantic import BaseModel, Field
import os
import torch
import time
import traceback
import random
from pathlib import Path
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool


# 定义输入模型
class EndoscopyVQAInput(BaseModel):
    """内窥镜图像问答分析的输入参数"""

    image_path: str = Field(
        ..., description="内窥镜图像文件的路径，支持JPG和PNG格式"
    )
    question: str = Field(
        "", description="对图像的具体问题，如果为空则生成一般性描述"
    )
    classification_result: Optional[Any] = Field(
        None, description="分类工具的结果(可选)"
    )
    detection_result: Optional[Any] = Field(
        None, description="检测工具的结果(可选)"
    )

class EndoscopyVQATool(BaseTool):
    """内窥镜图像问答工具

    这个工具使用ColonGPT模型分析内窥镜图像并回答相关问题或提供详细描述。
    它可以理解图像内容、识别病变特征，并提供专业的医学解释。
    """

    name: str = "endoscopy_vqa_tool"
    description: str = (
        "分析内窥镜图像并回答相关问题或提供详细描述。此工具能够：\n"
        "1. 识别和描述图像中的病变特征\n"
        "2. 解释图像中可见的解剖结构\n"
        "3. 描述组织表面特征(如颜色、纹理等)\n"
        "4. 提供关于内窥镜检查所见的专业医学解读\n"
        "输入应为内窥镜图像的路径和可选的特定问题。"
    )
    args_schema: Type[BaseModel] = EndoscopyVQAInput
    classifier_tool: Optional[Any] = Field(default=None, description="分类工具的实例")
    detection_tool: Optional[Any] = Field(default=None, description="检测工具的实例")
    
    # 模型参数
    model_path: str = 'endoagent/tools/Config/IntelliScope/cache/checkpoint/ColonGPT-phi1.5-siglip-stg1'
    model_base: str = 'endoagent/tools/Config/IntelliScope/cache/downloaded-weights/phi-1.5'
    model_type: str = 'phi-1.5'
    device: str = "cuda"
    max_new_tokens: int = 512
    temperature: float = 0.2
    
    # 类变量用于缓存模型，避免重复加载
    _tokenizer = None
    _model = None
    _image_processor = None
    _context_len = None
    
    # 预设问题模板
    question_templates: ClassVar[List[str]] = [
        "Describe what you see in this endoscopy image.",
        "Interpret what this endoscopic image shows.",
        "Detail the visual elements in this endoscopy image.",
        "Explain the endoscopic image's visuals thoroughly.",
        "Offer a thorough explanation of this endoscopic image."
    ]
    
    def __init__(
        self,
        model_path: str = None,
        model_base: str = None,
        device: str = "cuda",
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        classifier_tool = None,
        detection_tool = None,
    ):
        """初始化内窥镜VQA工具
        
        Args:
            model_path: ColonGPT模型路径
            model_base: 基础模型路径
            device: 运行设备 ('cuda' 或 'cpu')
            max_new_tokens: 生成的最大token数
            temperature: 生成文本的温度参数
            classifier_tool: 分类工具实例
            detection_tool: 检测工具实例
        """
        # 给父类构造函数传递所有参数，包括工具实例
        super().__init__(
            classifier_tool=classifier_tool,
            detection_tool=detection_tool
        )
        
        if model_path:
            self.model_path = model_path
        if model_base:
            self.model_base = model_base
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        
        # 验证模型路径
        if not os.path.exists(self.model_path):
            print(f"警告: 模型路径不存在: {self.model_path}")
        
        if not os.path.exists(self.model_base):
            print(f"警告: 基础模型路径不存在: {self.model_base}")
                
        print(f"内窥镜VQA工具初始化完成，模型路径: {self.model_path}")
        print(f"依赖工具状态: 分类工具{'已连接' if classifier_tool else '未连接'}, "
              f"检测工具{'已连接' if detection_tool else '未连接'}")
        
        # 修正环境变量，指向正确的视觉模型
        os.environ["VISION_TOWER_PATH"] = "google/siglip-so400m-patch14-384"
    
    def _select_question(self, user_question: str = "") -> str:
        """根据用户输入选择或使用预设问题模板
        
        Args:
            user_question: 用户输入的问题
            
        Returns:
            str: 最终使用的问题
        """
        if user_question and len(user_question.strip()) > 0:
            return user_question
        else:
            # 随机选择一个预设模板
            return random.choice(self.question_templates)
    
    def _is_normal_image(self, classification_result: Any, detection_result: Any) -> bool:
        """判断图像是否为正常图像（基于分类和检测结果）
        
        Args:
            classification_result: 分类工具的结果
            detection_result: 检测工具的结果
            
        Returns:
            bool: 如果图像正常返回True，否则返回False
        """
        # 默认为异常，除非能确认正常
        is_normal_classification = False
        is_normal_detection = False
        
        # 检查分类结果
        if classification_result is not None:
            if isinstance(classification_result, tuple) and len(classification_result) > 0:
                if isinstance(classification_result[0], dict):
                    normal_prob = classification_result[0].get('Normal', 0)
                    # 如果Normal概率大于0.5或者是最高概率，判断为正常
                    if normal_prob > 0.5 or normal_prob == max(classification_result[0].values()):
                        is_normal_classification = True
                        print(f"VQA分析：分类结果显示为正常，Normal概率为{normal_prob}")
        
        # 检查检测结果
        if detection_result is not None:
            if isinstance(detection_result, tuple) and len(detection_result) > 0:
                if isinstance(detection_result[0], dict):
                    objects_detected = detection_result[0].get('objects_detected', 0)
                    if objects_detected == 0:
                        is_normal_detection = True
                        print(f"VQA分析：检测结果未发现异常，检测到的对象数: {objects_detected}")
            elif isinstance(detection_result, dict):
                objects_detected = detection_result.get('objects_detected', 0)
                if objects_detected == 0:
                    is_normal_detection = True
                    print(f"VQA分析：检测结果未发现异常，检测到的对象数: {objects_detected}")
        
        # 必须两者都认为是正常，才最终判定为正常
        return is_normal_classification and is_normal_detection
    
    def _load_model_if_needed(self):
        """懒加载模型，如果模型还没有加载"""
        if EndoscopyVQATool._tokenizer is None or EndoscopyVQATool._model is None:
            try:
                # 设置正确的路径
                sys.path = [p for p in sys.path if 'Code/ColonGPT' not in p]
                sys.path.insert(0, '/path/to/EndoAgent')
                sys.path.insert(0, '/path/to/EndoAgent/endoagent/tools/Config/IntelliScope')
                
                # 动态导入所需模块
                from colongpt.model.builder import load_pretrained_model
                from colongpt.util.utils import disable_torch_init
                
                # 禁用PyTorch初始化
                disable_torch_init()
                
                # 加载模型
                print("加载ColonGPT模型...")
                from colongpt.util.mm_utils import get_model_name_from_path
                model_name = get_model_name_from_path(self.model_path)
                
                EndoscopyVQATool._tokenizer, EndoscopyVQATool._model, EndoscopyVQATool._image_processor, EndoscopyVQATool._context_len = \
                    load_pretrained_model(
                        self.model_path, self.model_base, model_name,
                        self.model_type, False, False, device=self.device
                    )
                print("ColonGPT模型加载完成")
                
            except Exception as e:
                print(f"加载模型时出错: {str(e)}")
                print(traceback.format_exc())
                raise
    
    def _process_image_and_question(self, image_path: str, question: str) -> str:
        """使用ColonGPT模型处理图像和问题"""
        try:
            sys.path.append('/path/to/EndoAgent/endoagent/tools/Config/IntelliScope')
            
            import torch
            from PIL import Image
            
            # 导入所需模块
            from colongpt.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
            from colongpt.conversation import conv_templates, SeparatorStyle
            from colongpt.model.builder import load_pretrained_model
            from colongpt.util.utils import disable_torch_init
            from colongpt.util.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
            
            # 禁用PyTorch初始化
            disable_torch_init()
            
            # 在这个方法内直接加载模型，避免使用静态成员
            print("加载ColonGPT模型...")
            model_name = get_model_name_from_path(self.model_path)
            tokenizer, model, image_processor, context_len = load_pretrained_model(
                self.model_path, self.model_base, model_name,
                self.model_type, False, False, device=self.device
            )
            print("ColonGPT模型加载完成")
            
            # 设置对话模式
            conv_mode = "colongpt"
            conv = conv_templates[conv_mode].copy()
            
            # 加载图像
            print(f"处理图像: {image_path}")
            image = Image.open(image_path).convert('RGB')
            image_tensor = process_images([image], image_processor, model.config)
            if type(image_tensor) is list:
                image_tensor = [img.to(model.device, dtype=model.dtype) for img in image_tensor]
            else:
                image_tensor = image_tensor.to(model.device, dtype=model.dtype)
            
            # 准备输入
            inp = DEFAULT_IMAGE_TOKEN + '\n' + question
            conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            
            # 准备生成参数
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
            
            # 生成回答
            print("生成回答...")
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    do_sample=True if self.temperature > 0 else False,
                    temperature=self.temperature,
                    max_new_tokens=self.max_new_tokens,
                    use_cache=True,
                    repetition_penalty=1.0,
                    stopping_criteria=[stopping_criteria]
                )
            
            # 解码输出
            outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
            print(f"生成回答完成，长度: {len(outputs)}")
            return outputs
            
        except Exception as e:
            print(f"处理图像和问题时出错: {str(e)}")
            print(traceback.format_exc())
            return f"分析失败: {str(e)}"
    
    def _run(
        self,
        image_path: str,
        question: str = "",
        classification_result: Any = None,
        detection_result: Any = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """执行内窥镜VQA
    
        Args:
            image_path: 内窥镜图像路径
            question: 关于图像的问题(可选)
            classification_result: 分类工具的结果(可选)
            detection_result: 检测工具的结果(可选)
            run_manager: 回调管理器
    
        Returns:
            Dict[str, Any]: 包含回答的结果
        """
        try:
            start_time = time.time()
            print(f"VQA工具处理图像: {image_path}")
            
            # 检查图像路径
            if not os.path.exists(image_path):
                return {
                    "error": f"找不到图像文件: {image_path}",
                    "description": f"错误: 找不到图像文件: {image_path}",
                    "result": f"错误: 找不到图像文件: {image_path}",
                    "status": "error"
                }
            
            # 如果没有传递分类结果，且有分类工具，则运行分类
            if classification_result is None and self.classifier_tool:
                print("VQA工具: 未提供分类结果，调用分类工具...")
                try:
                    classification_result = self.classifier_tool._run(image_path=image_path)
                except Exception as e:
                    print(f"VQA工具调用分类工具失败: {e}")
            
            # 如果没有传递检测结果，且有检测工具，则运行检测
            if detection_result is None and self.detection_tool:
                print("VQA工具: 未提供检测结果，调用检测工具...")
                try:
                    detection_result = self.detection_tool._run(image_path=image_path)
                except Exception as e:
                    print(f"VQA工具调用检测工具失败: {e}")
            
            # 选择问题
            final_question = self._select_question(question)
            print(f"VQA问题: {final_question}")
            
            # 根据分类和检测结果判断图像是否正常
            is_normal = self._is_normal_image(classification_result, detection_result)
            if is_normal:
                print("图像被判定为正常，跳过大模型分析")
                # 对于正常图像，跳过大模型处理
                analysis_path = "simplified"
                # 但仍然需要返回一个结果
                response = "内窥镜检查未见明显异常，图像显示为正常结构。"
            else:
                print("图像可能存在异常，执行大模型分析")
                # 对于可能异常的图像，调用大模型处理
                response = self._process_image_and_question(image_path, final_question)
                analysis_path = "complete"
            
            process_time = time.time() - start_time
            print(f"VQA处理完成，用时: {process_time:.2f}秒")
            
            # 构建输出
            result = {
                "description": response,
                "result": response,
                "image_path": image_path,
                "question": final_question,
                "processing_time": f"{process_time:.2f}秒",
                "tool_name": "endoscopy_vqa_tool",
                "status": "success",
                "analysis_path": analysis_path,
                "vqa_results": [{
                    "question": final_question,
                    "answer": response
                }]
            }
            
            return result
            
        except Exception as e:
            error_msg = str(e)
            traceback_info = traceback.format_exc()
            print(f"VQA处理出错: {error_msg}")
            print(traceback_info)
            
            # 构建错误输出
            error_result = {
                "error": error_msg,
                "description": f"分析过程中出错: {error_msg}",
                "result": f"无法生成内窥镜图像的描述: {error_msg}",
                "image_path": image_path,
                "status": "error",
                "vqa_results": [{
                    "question": question if question else "描述图像",
                    "answer": f"分析失败: {error_msg}"
                }]
            }
            
            return error_result
    
    async def _arun(
        self,
        image_path: str,
        question: str = "",
        classification_result: Any = None,
        detection_result: Any = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """异步执行内窥镜VQA"""
        return self._run(image_path, question, classification_result, detection_result, run_manager)