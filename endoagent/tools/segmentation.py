from typing import Dict, List, Optional, Tuple, Type, Any
from pathlib import Path
import uuid
import os
import sys
import torch
import numpy as np
from PIL import Image
import traceback
import warnings

from pydantic import BaseModel, Field
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from torchvision import transforms

# 导入分割工具相关模块
from detectron2.data import MetadataCatalog
from detectron2.utils.colormap import random_color
from endoagent.tools.Config.UniEndo.modeling.BaseModel import BaseModel
from endoagent.tools.Config.UniEndo.modeling import build_model
from endoagent.tools.Config.UniEndo.utils.arguments import load_opt_command
from endoagent.tools.Config.UniEndo.utils.visualizer import Visualizer
from endoagent.tools.Config.UniEndo.utils.distributed import init_distributed

# 忽略无害警告
warnings.filterwarnings("ignore")


from pydantic import create_model

# 使用 create_model 动态创建输入模型，这与测试中使用的方法相同
EndoscopySegmentationInput = create_model(
    'EndoscopySegmentationInput',
    image_path=(str, Field(..., description="内窥镜图像文件的路径，支持JPG和PNG格式")),
    output_type=(str, Field("overlay", description="输出类型: 'mask'为二值分割掩码, 'overlay'为带原始图像的可视化叠加"))
)

class EndoscopySegmentationTool(BaseTool):
    """内窥镜病变区域分割分析工具

    此工具使用预训练的语义分割模型来检测内窥镜图像中的病变区域，包括内镜和其他异常组织。
    可以输出病变的二值掩码或叠加在原始图像上的可视化效果。
    """

    name: str = "endoscopy_segmentation_tool"
    description: str = (
        "分析内窥镜图像并检测其中的病变区域，如内镜等。此工具可以：\n"
        "1. 生成病变区域的精确分割掩码\n"
        "2. 计算病变区域的面积百分比\n"
        "3. 标记病变区域的边界框位置\n"
        "4. 提供分割结果的可视化\n"
        "输入应为内窥镜图像的路径，输出可以是二值掩码或原图上的可视化叠加。"
    )
    args_schema: Type[BaseModel] = EndoscopySegmentationInput
    # ...其余代码保持不变...
    device: torch.device = None
    model: Optional[object] = None
    image_transform: Optional[transforms.Compose] = None
    stuff_classes: List[str] = ["Colonoscope Polyp"]
    temp_dir: Path = Path("temp")
    opt: Any = None  
    cmdline_args: Any = None

    def __init__(
        self,
        model_path: str = '/path/to/EndoAgent/endoagent/models/uniendo_seg/model_state_dict.pt',
        config_path: str = '/path/to/EndoAgent/endoagent/models/uniendo_seg/xdecoder_focall_my1_colon_lap.yaml',
        device: str = 'cuda',
        temp_dir: Optional[str] = "temp"
    ):
        """初始化内镜分割工具

        Args:
            model_path: 预训练模型权重路径
            config_path: 模型配置文件路径
            device: 计算设备，'cuda'或'cpu'
            temp_dir: 保存输出图像的临时目录
        """
        super().__init__()
        try:
            # 设置临时目录
            self.temp_dir = Path(temp_dir)
            self.temp_dir.mkdir(exist_ok=True)
            
            # 检查CUDA可用性
            if device == 'cuda' and torch.cuda.is_available():
                torch.cuda.empty_cache()
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
            
            # 加载模型配置
            args = ['evaluate', '--conf_files', config_path]
            self.opt, self.cmdline_args = load_opt_command(args)
            self.opt = init_distributed(self.opt)
            
            # 初始化模型
            self.model = BaseModel(self.opt, build_model(self.opt))
            self.model = self.model.from_pretrained(model_path)
            self.model = self.model.eval().to(self.device)
            
            # 设置图像预处理
            self.image_transform = transforms.Compose([
                transforms.Resize(512, interpolation=Image.BICUBIC)
            ])
            
            # 设置元数据
            self.setup_metadata()
            
        except Exception as e:
            print(f"初始化内镜分割工具时出错: {str(e)}")
            print(traceback.format_exc())
            raise
    
    def setup_metadata(self):
        """设置模型的分类元数据"""
        # 设置类别颜色和ID映射
        stuff_colors = [random_color(rgb=True, maximum=255).astype(int).tolist() 
                         for _ in range(len(self.stuff_classes))]
        stuff_dataset_id_to_contiguous_id = {x: x for x in range(len(self.stuff_classes))}
        
        # 清理并注册新的元数据
        if "demo" in MetadataCatalog.list():
            MetadataCatalog.remove("demo")
        
        MetadataCatalog.get("demo").set(
            stuff_colors=stuff_colors,
            stuff_classes=self.stuff_classes,
            stuff_dataset_id_to_contiguous_id=stuff_dataset_id_to_contiguous_id,
        )
        
        # 生成文本嵌入
        self.model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(
            self.stuff_classes + ["background"], 
            is_eval=True
        )
        
        # 设置模型元数据
        metadata = MetadataCatalog.get('demo')
        self.model.model.metadata = metadata
        self.model.model.sem_seg_head.num_classes = len(self.stuff_classes)
    
    def process_image(self, image_path):
        """处理图像并进行病变分割
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            dict: 包含分割结果的词典
        """
        # 检查文件是否存在
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"找不到图像文件: {image_path}")
        
        # 读取图像
        image_ori = Image.open(image_path).convert("RGB")
        width, height = image_ori.size
        
        # 预处理图像
        image = self.image_transform(image_ori)
        image = np.asarray(image)
        image_ori_array = np.asarray(image_ori)
        
        # 转换为模型输入格式
        images = torch.from_numpy(image.copy()).permute(2, 0, 1).to(self.device)
        batch_inputs = [{'image': images, 'height': height, 'width': width}]
        
        # 模型推理
        with torch.no_grad():
            outputs = self.model.forward(batch_inputs)
            
            # 处理分割结果
            sem_seg = (~(outputs[-1]['sem_seg'] > 0.5)).squeeze().type(torch.uint8)
            sem_seg = sem_seg.cpu().numpy()
            
            # 创建二值掩码
            mask = (1-sem_seg) * 255
            mask_image = Image.fromarray(mask.astype(np.uint8))
            
            # 创建叠加可视化
            metadata = MetadataCatalog.get('demo')
            visual = Visualizer(image_ori_array, metadata=metadata)
            overlay = visual.draw_sem_seg(sem_seg, alpha=0.5).get_image()
            overlay_image = Image.fromarray(overlay)
            
            # 计算分割统计信息
            total_pixels = sem_seg.size
            lesion_pixels = np.sum(sem_seg == 0)
            lesion_area_percentage = float(lesion_pixels / total_pixels)
            
            # 计算边界框
            from skimage import measure
            regions = measure.regionprops(measure.label((1-sem_seg).astype(np.uint8)))
            bboxes = []
            for region in regions:
                if region.area < 100:  # 过滤小区域
                    continue
                bbox = [region.bbox[1], region.bbox[0], region.bbox[3], region.bbox[2]]
                bboxes.append(bbox)
            
            # 构建返回结果
            return {
                "mask_image": mask_image,
                "overlay_image": overlay_image,
                "lesion_detected": lesion_pixels > 100,
                "lesion_area_percentage": lesion_area_percentage,
                "bounding_boxes": bboxes,
                "num_lesions": len(bboxes),
                "original_size": (width, height),
            }
    
    def _run(
        self,
        image_path: str,
        output_type: str = "overlay",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Tuple[Dict[str, Any], Dict]:
        """执行内窥镜病变分割分析"""
        try:
            print(f"EndoscopySegmentationTool: 开始处理图像 {image_path}")
            print(f"EndoscopySegmentationTool: 输出类型为 {output_type}")
            
            # 检查文件是否存在
            if not os.path.exists(image_path):
                print(f"EndoscopySegmentationTool: 错误 - 图像不存在: {image_path}")
                raise FileNotFoundError(f"找不到图像文件: {image_path}")
            
            # 处理图像
            print(f"EndoscopySegmentationTool: 调用 process_image")
            results = self.process_image(image_path)
            
            # 保存输出图像
            output_filename = f"lesion_segmentation_{uuid.uuid4().hex[:8]}.png"
            output_path = self.temp_dir / output_filename
            print(f"EndoscopySegmentationTool: 将输出保存到 {output_path}")
            
            if output_type.lower() == "mask":
                results["mask_image"].save(output_path)
                viz_path = str(output_path)
                result_image = results["mask_image"]
                print(f"EndoscopySegmentationTool: 已保存掩码图像")
            else:  # overlay
                results["overlay_image"].save(output_path)
                viz_path = str(output_path)
                result_image = results["overlay_image"]
                print(f"EndoscopySegmentationTool: 已保存叠加图像")
            
            output = {
                "segmentation_image_path": viz_path,
                "lesion_detected": results["lesion_detected"],
                "lesion_area_percentage": results["lesion_area_percentage"] * 100,
                "num_lesions": results["num_lesions"],
                "bounding_boxes": results["bounding_boxes"],
                # 添加额外的描述信息，帮助模型更好地格式化输出
                "description": (
                    f"分割分析结果显示图像中{'检测到' if results['lesion_detected'] else '未检测到'}病变区域。"
                    f"病变区域占图像总面积的{results['lesion_area_percentage'] * 100:.2f}%。"
                    f"共检测到{results['num_lesions']}个病变区域。"
                ),
                "segmentation_type": self.stuff_classes[0],  # 提供检测类型信息
                "display_note": "图像已通过界面直接显示，请勿在响应中使用Markdown图像语法。"  # 添加这个提示
            }
            
            # 构建元数据
            metadata = {
                "image_path": image_path,
                "output_type": output_type,
                "original_size": results["original_size"],
                "device": self.device.type,
                "analysis_status": "completed",
                "visualization_available": True,
                "visualization_path": viz_path,
            }
            
            print(f"EndoscopySegmentationTool: 返回结果 - 检测到病变: {output['lesion_detected']}, 图像路径: {viz_path}")
            return output, metadata  # 返回元组而不是单个字典
            
        except Exception as e:
            # ...错误处理代码...
            error_msg = str(e)
            traceback_info = traceback.format_exc()
            error_output = {"error": error_msg}
            error_metadata = {
                "traceback": traceback_info,
                "analysis_status": "failed",
                "image_path": image_path
            }
            return error_output, error_metadata  # 错误情况也返回元组

    async def _arun(
        self,
        image_path: str,
        output_type: str = "overlay",
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Tuple[Dict[str, Any], Dict]:
        """异步执行内镜分割分析"""
        result = self._run(image_path, output_type)
        # 如果结果已经是元组，直接返回
        if isinstance(result, tuple):
            return result
        # 如果结果是字典，将其转换为兼容格式
        metadata = result.pop("analysis", {})
        return result, metadata