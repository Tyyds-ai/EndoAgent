from typing import Dict, List, Optional, Tuple, Type, Any
from pathlib import Path
import uuid
import os
import sys
import torch
import numpy as np
from PIL import Image
import cv2
import faiss
import traceback
import warnings

from pydantic import BaseModel, Field
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool

# 导入息肉生成相关模块
from diffusers import StableDiffusionInpaintPipeline
from endoagent.tools.Config.Polyp_Gen.Matcher import Dinov2Matcher
from endoagent.tools.Config.Polyp_Gen.LocalMatching import MaskProposer

# 忽略无害警告
warnings.filterwarnings("ignore")

from pydantic import create_model

# 使用 create_model 动态创建输入模型
EndoscopyGenerationInput = create_model(
    'EndoscopyGenerationInput',
    image_path=(str, Field(..., description="要生成息肉的内窥镜图像文件路径，支持JPG和PNG格式")),
    prompt=(str, Field("Polyp", description="息肉类型提示词，如'Polyp'")),
    steps=(int, Field(50, description="生成步数，值越大图像质量越高但速度也越慢"))
)

class EndoscopyPolypGenerationTool(BaseTool):
    """内窥镜息肉生成工具

    此工具专门用于在内窥镜图像中生成逼真的息肉区域。
    利用图像检索和掩码匹配自动确定息肉生成位置，然后使用扩散模型生成息肉。
    """

    name: str = "endoscopy_generation_tool"
    description: str = (
        "在内窥镜图像中自动生成逼真的息肉(仅限息肉生成)。此工具可以：\n"
        "1. 自动确定适合生成息肉的区域\n"
        "2. 基于提供的提示词生成息肉\n"
        "3. 创建高质量的合成息肉图像\n"
        "4. 输出原始图像、息肉掩码和带息肉的结果图像\n"
        "输入应为内窥镜图像的路径，可选择息肉特征描述和生成质量参数。"
    )
    args_schema: Type[BaseModel] = EndoscopyGenerationInput
    
    # 模型参数
    model: Optional[StableDiffusionInpaintPipeline] = None
    matcher: Optional[Dinov2Matcher] = None
    model_path: str = 'endoagent/tools/Config/Polyp_Gen/checkpoint'
    database_path: str = 'endoagent/tools/Config/Polyp_Gen/data/database/test_images.index'
    paths_file: str = None  # 默认为None，将自动使用与数据库同名的_paths.txt文件
    mask_dir: str = 'endoagent/tools/Config/Polyp_Gen/data/LDPolypVideo/Test/Masks'
    temp_dir: Path = Path("temp")
    cuda: bool = True

    def __init__(
        self,
        model_path: str = None,
        database_path: str = None,
        mask_dir: str = None, 
        device: str = 'cuda',
        temp_dir: Optional[str] = "temp"
    ):
        """初始化内窥镜息肉生成工具

        Args:
            model_path: StableDiffusion模型路径，默认为类变量中的路径
            database_path: 数据库索引文件路径，默认为类变量中的路径
            mask_dir: 掩码目录，默认为类变量中的路径
            device: 计算设备，'cuda'或'cpu'
            temp_dir: 保存输出图像的临时目录
        """
        super().__init__()
        try:
            # 设置临时目录
            self.temp_dir = Path(temp_dir)
            self.temp_dir.mkdir(exist_ok=True)
            
            # 更新路径（如果提供了新的路径）
            if model_path:
                self.model_path = model_path
            if database_path:
                self.database_path = database_path
            if mask_dir:
                self.mask_dir = mask_dir
                
            # 设置paths_file
            if self.paths_file is None:
                self.paths_file = os.path.splitext(self.database_path)[0] + "_paths.txt"
                
            # 设置CUDA使用
            self.cuda = device == 'cuda' and torch.cuda.is_available()
            
            # 初始化特征匹配器
            print("初始化特征匹配器...")
            self.matcher = Dinov2Matcher(half_precision=False)
            
            # 初始化StableDiffusion模型
            print(f"加载StableDiffusion模型，路径: {self.model_path}")
            self.model = StableDiffusionInpaintPipeline.from_pretrained(
                self.model_path,
                revision="fp16",
                torch_dtype=torch.float16 if self.cuda else torch.float32,
                safety_checker=None
            )
            
            if self.cuda:
                self.model = self.model.to("cuda")
            else:
                self.model = self.model.to("cpu")
            
            print(f"成功初始化内窥镜息肉生成工具，使用设备：{device}")
            
        except Exception as e:
            print(f"初始化内窥镜息肉生成工具时出错: {str(e)}")
            print(traceback.format_exc())
            raise
    
    def get_mask_path_from_image_path(self, image_path):
        """根据图像路径获取对应的掩码路径"""
        filename = os.path.basename(image_path)
        
        if filename.startswith("img_"):
            mask_filename = "mask_" + filename.split("img_")[1]
        else:
            basename, ext = os.path.splitext(filename)
            mask_filename = f"mask_{basename}{ext}"
        
        mask_path = os.path.join(self.mask_dir, mask_filename)
        return mask_path
    
    def retrieve_similar_images(self, query_image_path, top_k=1):
        """在数据库中检索与查询图像最相似的图像"""
        # 提取查询图像特征
        with torch.no_grad():
            image = cv2.cvtColor(cv2.imread(query_image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            image_tensor, _, _ = self.matcher.prepare_image(image)
            global_features = self.matcher.extract_global_features(image_tensor)
        
        # 转换为float32并归一化
        vector = np.float32(global_features)
        faiss.normalize_L2(vector)
        
        # 从数据库读取索引并搜索
        index = faiss.read_index(self.database_path)
        distances, indices = index.search(vector, top_k)
        
        # 读取图像路径
        image_paths = []
        if os.path.exists(self.paths_file):
            with open(self.paths_file, "r") as f:
                image_paths = [line.strip() for line in f.readlines()]
        else:
            print(f"警告：找不到路径文件 {self.paths_file}，无法获取图像路径")
        
        return distances, indices, image_paths
    
    def generate_mask(self, query_image_path, output_mask_path):
        """为查询图像生成掩码"""
        print(f"在数据库中检索与 {query_image_path} 相似的图像...")
        distances, indices, image_paths = self.retrieve_similar_images(query_image_path)
        
        if not image_paths:
            raise ValueError("无法获取图像路径列表，请检查数据库和路径文件")
        
        # 获取最相似图像的索引和路径
        best_index = indices[0][0]
        
        if best_index >= len(image_paths):
            raise ValueError(f"检索结果索引 {best_index} 超出图像路径列表范围 {len(image_paths)}")
        
        best_match_path = image_paths[best_index]
        print(f"最佳匹配: {best_match_path} (距离: {distances[0][0]:.4f})")
        
        # 获取对应的掩码路径
        mask_path = self.get_mask_path_from_image_path(best_match_path)
        
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"找不到对应的掩码文件 {mask_path}")
        
        print(f"找到对应的掩码: {mask_path}")
        
        # 使用MaskProposer生成掩码提案
        print(f"正在生成掩码提案...")
        mask = MaskProposer(
            origin_image=best_match_path,
            origin_mask=mask_path, 
            target_image=query_image_path, 
            target_mask=output_mask_path,
            matching_figure_save_path=None
        )
        
        print(f"掩码提案已保存至: {output_mask_path}")
        return output_mask_path
    
    def generate_polyp(self, image_path, mask_path, output_path, prompt="Polyp", steps=50):
        """使用StableDiffusion模型生成息肉图像"""
        # 加载图像和掩码
        image = Image.open(image_path)
        mask_image = Image.open(mask_path)
        
        print(f"正在生成息肉图像，提示词: '{prompt}'...")
        gen_image = self.model(
            prompt=prompt, 
            image=image, 
            mask_image=mask_image,
            width=image.size[0], 
            height=image.size[1], 
            num_inference_steps=steps,
        ).images[0]
        
        gen_image.save(output_path)
        print(f"息肉生成完成，已保存至: {output_path}")
        return gen_image
    
    def process_image(self, image_path, prompt="Polyp", steps=50):
        """处理图像并生成息肉
        
        Args:
            image_path: 图像文件路径
            prompt: 生成提示词
            steps: 生成步数
            
        Returns:
            dict: 包含生成结果的词典
        """
        # 检查文件是否存在
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"找不到图像文件: {image_path}")
        
        # 生成临时文件名
        query_filename = os.path.basename(image_path)
        query_basename, query_ext = os.path.splitext(query_filename)
        
        # 为掩码和生成结果设置路径
        mask_path = str(self.temp_dir / f"{query_basename}_mask{query_ext}")
        output_path = str(self.temp_dir / f"{query_basename}_polyp{query_ext}")
        
        # 步骤1: 生成掩码
        mask_path = self.generate_mask(image_path, mask_path)
        
        # 步骤2: 生成息肉图像
        gen_image = self.generate_polyp(image_path, mask_path, output_path, prompt, steps)
        
        # 读取原图以获取尺寸
        original_image = Image.open(image_path)
        
        # 构建返回结果
        return {
            "original_image": original_image,
            "mask_image": Image.open(mask_path),
            "generated_image": gen_image,
            "mask_path": mask_path,
            "generated_path": output_path,
            "original_size": original_image.size,
            "prompt": prompt,
            "steps": steps
        }
    
    def _run(
        self,
        image_path: str,
        prompt: str = "Polyp",
        steps: int = 50,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Tuple[Dict[str, Any], Dict]:
        """执行内窥镜息肉生成"""
        try:
            print(f"EndoscopyPolypGenerationTool: 开始处理图像 {image_path}")
            print(f"EndoscopyPolypGenerationTool: 生成提示词 {prompt}")
            
            # 检查文件是否存在
            if not os.path.exists(image_path):
                print(f"EndoscopyPolypGenerationTool: 错误 - 图像不存在: {image_path}")
                raise FileNotFoundError(f"找不到图像文件: {image_path}")
            
            # 处理图像
            print(f"EndoscopyPolypGenerationTool: 调用 process_image")
            results = self.process_image(image_path, prompt, steps)
            
            # 生成描述文本
            generation_text = (
                f"已在指定区域生成{prompt}。\n\n"
                f"生成步数: {steps}\n"
                f"原始图像大小: {results['original_size'][0]}×{results['original_size'][1]}像素\n"
                f"系统自动确定了最适合生成息肉的区域，并基于相似病例生成了逼真的息肉。"
            )
            
            # 构建输出结果
            output = {
                "generation_image_path": results["generated_path"],
                "mask_image_path": results["mask_path"],
                "prompt": prompt,
                "steps": steps,
                "description": generation_text,
                "segmentation_image_path": results["generated_path"],  # 兼容界面显示
                "display_note": "生成结果图像可通过EndoscopyImageVisualizerTool工具查看，请务必调用该工具显示结果。"
            }
            
            # 构建元数据
            metadata = {
                "image_path": image_path,
                "mask_path": results["mask_path"],
                "generated_path": results["generated_path"],
                "prompt": prompt,
                "steps": steps,
                "image_size": results["original_size"],
                "device": "cuda" if self.cuda else "cpu",
                "analysis_status": "completed",
                "visualization_available": True,
                "visualization_path": results["generated_path"],
            }
            
            print(f"EndoscopyPolypGenerationTool: 返回生成结果")
            return output, metadata
            
        except Exception as e:
            error_msg = str(e)
            traceback_info = traceback.format_exc()
            print(f"生成出错: {error_msg}")
            print(traceback_info)
            error_output = {"error": error_msg}
            error_metadata = {
                "traceback": traceback_info,
                "analysis_status": "failed",
                "image_path": image_path
            }
            return error_output, error_metadata

    async def _arun(
        self,
        image_path: str,
        prompt: str = "Polyp",
        steps: int = 50,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Tuple[Dict[str, Any], Dict]:
        """异步执行内窥镜息肉生成"""
        return self._run(image_path, prompt, steps)