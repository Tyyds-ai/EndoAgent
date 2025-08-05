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

# 导入分割模块
from endoagent.tools.segmentation import EndoscopySegmentationTool

# 忽略无害警告
warnings.filterwarnings("ignore")

from pydantic import create_model

# 使用 create_model 动态创建输入模型
EndoscopyNormalGenerationInput = create_model(
    'EndoscopyNormalGenerationInput',
    image_path=(str, Field(..., description="包含息肉的内窥镜图像文件路径，支持JPG和PNG格式")),
    steps=(int, Field(50, description="生成步数，值越大图像质量越高但速度也越慢")),
    use_segmentation=(bool, Field(True, description="是否使用分割工具自动检测息肉区域，如为False则使用数据库匹配方式"))
)

class EndoscopyNormalGenerationTool(BaseTool):
    """内窥镜息肉去除工具

    此工具专门用于将含有息肉的内窥镜图像转换为正常图像，去除息肉区域。
    可以使用分割模型自动识别息肉区域，或通过图像检索匹配确定息肉位置，然后使用扩散模型将息肉区域替换为正常黏膜。
    """

    name: str = "endoscopy_normal_generation_tool"
    description: str = (
        "将内窥镜息肉图像转换为正常图像(息肉去除)。此工具可以：\n"
        "1. 使用分割模型自动定位息肉区域\n"
        "2. 将息肉区域替换为正常的黏膜组织\n"
        "3. 创建高质量的无病变内窥镜图像\n"
        "4. 输出原始图像、息肉掩码和去除息肉后的结果图像\n"
        "输入应为含有息肉的内窥镜图像的路径，默认使用分割工具自动检测息肉区域。"
    )
    args_schema: Type[BaseModel] = EndoscopyNormalGenerationInput
    
    # 模型参数
    model: Optional[StableDiffusionInpaintPipeline] = None
    matcher: Optional[Dinov2Matcher] = None
    segmentation_tool: Optional[EndoscopySegmentationTool] = None
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
        segmentation_tool: EndoscopySegmentationTool = None,
        device: str = 'cuda',
        temp_dir: Optional[str] = "temp"
    ):
        """初始化内窥镜息肉去除工具

        Args:
            model_path: StableDiffusion模型路径，默认为类变量中的路径
            database_path: 数据库索引文件路径，默认为类变量中的路径
            mask_dir: 掩码目录，默认为类变量中的路径
            segmentation_tool: 用于息肉分割的工具实例
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
            
            # 保存分割工具引用
            self.segmentation_tool = segmentation_tool
                
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
            
            print(f"成功初始化内窥镜息肉去除工具，使用设备：{device}")
            print(f"分割工具状态: {'已连接' if self.segmentation_tool else '未连接'}")
            
        except Exception as e:
            print(f"初始化内窥镜息肉去除工具时出错: {str(e)}")
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
    
    def generate_mask_with_segmentation(self, image_path, output_mask_path):
        """使用分割模型生成息肉掩码"""
        print(f"使用分割模型检测息肉区域: {image_path}")
        
        if not self.segmentation_tool:
            raise ValueError("分割工具未初始化，无法使用分割方法生成掩码")
        
        # 调用分割工具分割息肉
        seg_result = self.segmentation_tool.process_image(image_path)
        
        if not seg_result.get("lesion_detected", False):
            print("警告: 分割模型未检测到息肉区域")
            # 创建空白掩码
            img = np.array(Image.open(image_path))
            blank_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
            Image.fromarray(blank_mask).save(output_mask_path)
            return output_mask_path
        
        # 获取分割掩码并转换为PIL图像
        mask_image = seg_result.get("mask_image")
        
        # 将掩码转换为OpenCV格式进行处理
        mask_np = np.array(mask_image)
        
        # 使用OpenCV找到最小矩形边界框
        contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 创建空白掩码
        refined_mask = np.zeros_like(mask_np)
        
        # 对每个轮廓计算最小外接矩形并填充
        for contour in contours:
            if cv2.contourArea(contour) < 100:  # 过滤小区域
                continue
            x, y, w, h = cv2.boundingRect(contour)
            # 扩大矩形边界(可选，增加5像素边界)
            x = max(0, x-5)
            y = max(0, y-5)
            w = min(mask_np.shape[1]-x, w+10)
            h = min(mask_np.shape[0]-y, h+10)
            # 填充矩形区域
            refined_mask[y:y+h, x:x+w] = 255
        
        # 保存处理后的掩码
        Image.fromarray(refined_mask).save(output_mask_path)
        print(f"息肉分割掩码(矩形化)已保存至: {output_mask_path}")
        
        return output_mask_path
    
    def generate_mask_with_matching(self, query_image_path, output_mask_path):
        """使用图像匹配生成掩码"""
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
    
    def generate_normal_image(self, image_path, mask_path, output_path, steps=50):
        """使用StableDiffusion模型去除息肉生成正常图像"""
        # 加载图像和掩码
        image = Image.open(image_path)
        mask_image = Image.open(mask_path)
        
        # 使用特定提示词生成正常组织
        prompt = "Normal colon mucosa, healthy endoscopic view, no polyps, no lesions"
        
        print(f"正在生成正常组织图像...")
        gen_image = self.model(
            prompt=prompt, 
            image=image, 
            mask_image=mask_image,
            negative_prompt="polyp, lesion, abnormal, disease",
            width=image.size[0], 
            height=image.size[1], 
            num_inference_steps=steps,
        ).images[0]
        
        gen_image.save(output_path)
        print(f"正常组织图像生成完成，已保存至: {output_path}")
        return gen_image
    
    def process_image(self, image_path, steps=50, use_segmentation=True):
        """处理图像并去除息肉
        
        Args:
            image_path: 图像文件路径
            steps: 生成步数
            use_segmentation: 是否使用分割模型识别息肉
            
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
        output_path = str(self.temp_dir / f"{query_basename}_normal{query_ext}")
        
        # 步骤1: 生成掩码
        if use_segmentation and self.segmentation_tool:
            mask_path = self.generate_mask_with_segmentation(image_path, mask_path)
        else:
            mask_path = self.generate_mask_with_matching(image_path, mask_path)
        
        # 步骤2: 生成正常图像
        gen_image = self.generate_normal_image(image_path, mask_path, output_path, steps)
        
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
            "steps": steps
        }
    
    def _run(
        self,
        image_path: str,
        steps: int = 50,
        use_segmentation: bool = True,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Tuple[Dict[str, Any], Dict]:
        """执行内窥镜息肉去除"""
        try:
            print(f"EndoscopyNormalGenerationTool: 开始处理图像 {image_path}")
            print(f"EndoscopyNormalGenerationTool: 使用分割模型: {use_segmentation}")
            
            # 检查文件是否存在
            if not os.path.exists(image_path):
                print(f"EndoscopyNormalGenerationTool: 错误 - 图像不存在: {image_path}")
                raise FileNotFoundError(f"找不到图像文件: {image_path}")
            
            # 处理图像
            print(f"EndoscopyNormalGenerationTool: 调用 process_image")
            results = self.process_image(image_path, steps, use_segmentation)
            
            # 生成描述文本
            generation_text = (
                f"已将息肉区域转换为正常结肠黏膜。\n\n"
                f"生成步数: {steps}\n"
                f"掩码生成方法: {'分割模型自动检测' if use_segmentation else '图像匹配'}\n"
                f"原始图像大小: {results['original_size'][0]}×{results['original_size'][1]}像素\n"
                f"系统{'使用分割模型自动检测息肉区域' if use_segmentation else '通过图像匹配寻找相似息肉区域'}，并替换为正常的黏膜组织。"
            )
            
            # 构建输出结果
            output = {
                "generation_image_path": results["generated_path"],
                "mask_image_path": results["mask_path"],
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
                "steps": steps,
                "use_segmentation": use_segmentation,
                "image_size": results["original_size"],
                "device": "cuda" if self.cuda else "cpu",
                "analysis_status": "completed",
                "visualization_available": True,
                "visualization_path": results["generated_path"],
            }
            
            print(f"EndoscopyNormalGenerationTool: 返回生成结果")
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
        steps: int = 50,
        use_segmentation: bool = True,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Tuple[Dict[str, Any], Dict]:
        """异步执行内窥镜息肉去除"""
        return self._run(image_path, steps, use_segmentation)