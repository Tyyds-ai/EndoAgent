from typing import Dict, Optional, Tuple, Type
from pydantic import BaseModel, Field
import os
import torch
import numpy as np
import traceback
import random
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from typing import Dict, Optional, Tuple, Type, ClassVar
from torchvision.transforms import Compose
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool

# 假设model.py中有resnet18定义
from endoagent.tools.Config.AFACNet.model import resnet18


class EndoscopyImageInput(BaseModel):
    """Input for endoscopy image analysis tools. Only supports JPG or PNG images."""

    image_path: str = Field(
        ..., description="Path to the endoscopy image file, only supports JPG or PNG images"
    )


class EndoscopyClassifierTool(BaseTool):
    """Tool that classifies endoscopy images for multiple pathologies.

    This tool uses a pre-trained ResNet18 model to analyze endoscopy images and
    predict the likelihood of various pathologies including:
    
    Adenoma, Cancer, Normal, Polyp

    The output values represent the probability (from 0 to 1) of each condition being present in the image.
    """

    name: str = "endoscopy_classifier"
    description: str = (
        "A tool that analyzes endoscopy images and classifies them for 4 different categories. "
        "Input should be the path to an endoscopy image file. "
        "Output is a dictionary of categories and their predicted probabilities (0 to 1). "
        "Categories include: Adenoma, Cancer, Normal, and Polyp. "
        "Higher values indicate a higher likelihood of the condition being present."
    )
    args_schema: Type[BaseModel] = EndoscopyImageInput
    device: Optional[str] = "cuda"
    model: Optional[object] = None
    image_transform: Optional[Compose] = None
    pathologies: list[str] = ["Adenoma", "Cancer", "Normal", "Polyp"]

    def __init__(self, weight_path: str = "endoagent/models/afacnet_cls.pth", device: Optional[str] = "cuda"):
        super().__init__()
        try:
            # 检查 CUDA 是否可用
            if device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
                self.device = torch.device("cuda")
                print(f"使用 CUDA GPU: {torch.cuda.get_device_name(0)}")
                print(f"可用显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
                
                # 为了避免某些CUDA错误，设置这些环境变量
                # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
            else:
                print("CUDA 不可用，使用 CPU")
                self.device = torch.device("cpu")
            
            # 直接在目标设备上初始化模型
            print(f"正在 {self.device} 上初始化模型...")
            self.model = resnet18(num_classes=4, inputchannel=3).to(self.device)
            
            # 直接在目标设备上加载权重
            # print(f"正在加载权重到 {self.device}: {weight_path}")
            weight_map_location = self.device
            self.model.load_state_dict(torch.load(weight_path, map_location=weight_map_location))
            
            # 确保模型处于评估模式
            self.model.eval()
            
            # 创建与数据集相同的转换
            self.image_transform = transforms.Compose([
                transforms.Resize([400, 400]),
            ])
            
            # print(f"成功初始化内窥镜分类器，使用设备：{self.device}")
        except Exception as e:
            print(f"初始化内窥镜分类器时出错: {str(e)}")
            raise

    def _swap(self, img, crop=(40, 40), p=0.75, pmask=0.15):
        """
        完全复制dataset.py中的swap函数
        """
        def crop_image(image, cropnum):
            width, high = image.shape[1], image.shape[2]
            crop_x = [int((width / cropnum[0]) * i) for i in range(cropnum[0] + 1)]
            crop_y = [int((high / cropnum[1]) * i) for i in range(cropnum[1] + 1)]
            im_list = []
            for j in range(len(crop_y) - 1):
                for i in range(len(crop_x) - 1):
                    img = image[:, crop_y[j]:min(crop_y[j + 1], high), crop_x[i]:min(crop_x[i + 1], width)]
                    img = np.fft.fftshift(np.fft.fftn(img))
                    img_low = img[:,img.shape[1]-2:img.shape[1]+2, img.shape[1]-2:img.shape[1]+2].copy()
                    index = img.shape[1] * img.shape[2]

                    masknum = int(index * pmask)
                    patchlist = random.sample(range(0, index), masknum)
                    for k in patchlist:
                        img[:, k//img.shape[1], k%img.shape[1]] = 0
                    img[:,img.shape[1]-2:img.shape[1]+2, img.shape[1]-2:img.shape[1]+2] = img_low

                    im_list.append(img)
            return im_list

        images = crop_image(img, crop)
        img1 = []
        for i in range(crop[0]):
            img1.append(np.concatenate(images[i*crop[0]:(i+1)*crop[0]],1))
        toImage = np.concatenate(img1,2)
        return toImage

    def _process_image(self, image_path: str) -> torch.Tensor:
        """处理输入的内窥镜图像进行模型推理。
        
        完全按照BowelDataset.__getitem__中的图像处理步骤实现。
        """
        try:
            from PIL import Image
            
            # 检查文件是否存在
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"找不到图像文件: {image_path}")
                
            # 加载图像 - 与BowelDataset相同
            img = Image.open(image_path)
            
            # 应用转换 - 与BowelDataset相同
            if self.image_transform is not None:
                img = self.image_transform(img)
            
            # 以下步骤完全复制自BowelDataset.__getitem__
            img = np.array(img)
            img = img.astype(np.float32)
            img = img / 255.0
            img = img.transpose((2, 0, 1))  # CHW格式
            
            # 使用与dataset.py完全相同的swap函数
            img_F = self._swap(img, (40, 40), p=0.75, pmask=0.15)
            F_real = np.real(img_F)
            F_imag = np.imag(img_F)
            F_complex = np.concatenate((F_real, F_imag), axis=0)
            
            # 转换为PyTorch张量 - 与BowelDataset相同
            F_complex = torch.tensor(F_complex).float().unsqueeze(0)  # 添加批次维度
            
            # print(f"处理后的张量形状: {F_complex.shape}")
            
            return F_complex
            
        except Exception as e:
            print(f"图像处理错误: {str(e)}")
            print(traceback.format_exc())
            raise
    
    def _run(
        self,
        image_path: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Tuple[Dict[str, float], Dict]:
        """对内窥镜图像进行分类，检测多种病理情况。"""
        try:
            # 处理图像
            img = self._process_image(image_path)
            
            # 将图像移至正确的设备
            img = img.to(self.device)
            
            # 使用不需要梯度的上下文进行推理
            with torch.inference_mode():
                print(f"在 {self.device} 上运行推理, 张量形状: {img.shape}")
                outputs = self.model(img)
                probs = torch.nn.functional.softmax(outputs, dim=1)[0]
                print(f"推理成功! 输出形状: {outputs.shape}")
            
            # 创建结果字典
            # 将numpy数组转换为Python原生类型，避免界面显示问题
            output = {name: float(value) for name, value in zip(self.pathologies, probs.detach().cpu().numpy())}
            
            metadata = {
                "image_path": image_path,
                "analysis_status": "completed",
                "device": self.device.type,
                "note": "Probabilities range from 0 to 1, with higher values indicating higher likelihood of the condition.",
            }
            
            return output, metadata
        except Exception as e:
            print(f"分类错误: {str(e)}")
            print(traceback.format_exc())
            return {"error": str(e)}, {
                "image_path": image_path,
                "analysis_status": "failed",
            }

    async def _arun(
        self,
        image_path: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Tuple[Dict[str, float], Dict]:
        """异步对内窥镜图像进行分类。"""
        return self._run(image_path)