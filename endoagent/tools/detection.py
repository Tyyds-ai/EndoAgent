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


# 导入YOLO检测模型
from endoagent.tools.Config.yolov8.yolo import YOLO

# 忽略无害警告
warnings.filterwarnings("ignore")

from pydantic import create_model

# 使用 create_model 动态创建输入模型
EndoscopyDetectionInput = create_model(
    'EndoscopyDetectionInput',
    image_path=(str, Field(..., description="内窥镜图像文件的路径，支持JPG和PNG格式")),
    confidence=(float, Field(0.35, description="检测置信度阈值，值越低检出的目标越多，但可能增加误检")),
    save_result=(bool, Field(True, description="是否保存检测结果图像"))
)

class EndoscopyDetectionTool(BaseTool):
    """内窥镜病变检测工具

    此工具使用YOLOv8目标检测模型检测内窥镜图像中的病变区域，提供边界框定位。
    注意：该工具只提供位置检测，不提供具体的病变类别分类。
    """

    name: str = "endoscopy_detection_tool"
    description: str = (
        "分析内窥镜图像并检测其中的病变区域，提供精确的边界框定位。此工具可以：\n"
        "1. 检测图像中的多个病变位置并绘制边界框\n"
        "2. 为每个检测到的病变区域提供置信度\n"
        "3. 统计检测到的病变区域数量\n"
        "4. 生成可视化的检测结果图像\n"
        "输入应为内窥镜图像的路径，输出为检测结果和统计信息。"
    )
    args_schema: Type[BaseModel] = EndoscopyDetectionInput
    
    # 模型参数
    model: Optional[YOLO] = None
    model_path: str = '/path/to/EndoAgent/yolov8-pytorch-master/logs/sgd-0.01/ep003-loss3.391-val_loss3.426.pth'
    classes_path: str = '/path/to/EndoAgent/yolov8-pytorch-master/model_data/sun_classes.txt'
    input_shape: List[int] = [640, 640]
    phi: str = 'x'
    cuda: bool = True
    letterbox_image: bool = True
    nms_iou: float = 0.3
    temp_dir: Path = Path("temp")

    def __init__(
        self,
        model_path: str = None,
        classes_path: str = None,
        device: str = 'cuda',
        temp_dir: Optional[str] = "temp",
        confidence: float = 0.35
    ):
        """初始化内窥镜检测工具

        Args:
            model_path: 预训练模型权重路径，默认使用类变量中的路径
            classes_path: 类别名称文件路径，默认使用类变量中的路径
            device: 计算设备，'cuda'或'cpu'
            temp_dir: 保存输出图像的临时目录
            confidence: 检测置信度阈值
        """
        super().__init__()
        try:
            # 设置临时目录
            self.temp_dir = Path(temp_dir)
            self.temp_dir.mkdir(exist_ok=True)
            
            # 更新模型路径（如果提供了新的路径）
            if model_path:
                self.model_path = model_path
            if classes_path:
                self.classes_path = classes_path
                
            # 设置CUDA使用
            self.cuda = device == 'cuda' and torch.cuda.is_available()
            
            # 初始化YOLO模型
            self.model = YOLO(
                model_path=self.model_path,
                classes_path=self.classes_path,
                input_shape=self.input_shape,
                phi=self.phi,
                confidence=confidence,
                nms_iou=self.nms_iou,
                letterbox_image=self.letterbox_image,
                cuda=self.cuda
            )
            
            print(f"成功初始化内窥镜病变检测工具，使用设备：{device}")
            
        except Exception as e:
            print(f"初始化内窥镜病变检测工具时出错: {str(e)}")
            print(traceback.format_exc())
            raise
    
    def process_image(self, image_path, confidence=None):
        """处理图像并进行病变检测
        
        Args:
            image_path: 图像文件路径
            confidence: 可选的置信度阈值，覆盖模型默认值
            
        Returns:
            dict: 包含检测结果的词典
        """
        # 检查文件是否存在
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"找不到图像文件: {image_path}")
        
        # 更新置信度（如果提供）
        if confidence is not None and confidence != self.model.confidence:
            self.model.confidence = confidence
        
        # 读取图像
        image = Image.open(image_path)
        
        # 运行检测
        results = {}
        
        # 检测图像
        with torch.no_grad():
            # 根据resize_image和预处理准备输入数据
            from endoagent.tools.Config.yolov8.utils_yolo.utils import cvtColor, preprocess_input, resize_image
            image_rgb = cvtColor(image)
            image_shape = np.array(np.shape(image_rgb)[0:2])
            
            # 获取图像对应的预处理数据
            image_data = resize_image(image_rgb, (self.model.input_shape[1], self.model.input_shape[0]), 
                                      self.model.letterbox_image)
            image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)
            
            # 转换为Torch张量
            images = torch.from_numpy(image_data)
            if self.model.cuda:
                images = images.cuda()
            
            # 前向传播
            outputs = self.model.net(images)
            outputs = self.model.bbox_util.decode_box(outputs)
            
            # 非极大值抑制
            detection_results = self.model.bbox_util.non_max_suppression(
                outputs, self.model.num_classes, self.model.input_shape,
                image_shape, self.model.letterbox_image,
                conf_thres=self.model.confidence, nms_thres=self.model.nms_iou
            )
            
            # 处理检测结果
            if detection_results[0] is None:
                # 没有检测到目标
                results = {
                    "objects_detected": 0,
                    "detections": [],
                    "image_size": image.size
                }
            else:
                # 获取检测结果
                top_conf = detection_results[0][:, 4]
                top_boxes = detection_results[0][:, :4]
                
                # 构建检测结果列表
                detections = []
                
                for i in range(len(top_conf)):
                    confidence = float(top_conf[i])
                    bbox = top_boxes[i].tolist()
                    
                    # 创建检测结果项 - 移除类别信息
                    detection_item = {
                        "confidence": confidence,
                        "bbox": bbox  # [top, left, bottom, right]
                    }
                    detections.append(detection_item)
                
                # 整合结果 - 移除类别计数
                results = {
                    "objects_detected": len(detections),
                    "detections": detections,
                    "image_size": image.size
                }
            
            # 生成检测结果可视化图像 - 使用原始的检测函数
            # 但不在结果中使用类别信息
            result_image = self.model.detect_image(image_rgb, count=True)
            results["result_image"] = result_image
            
            return results
    
    def _run(
        self,
        image_path: str,
        confidence: float = 0.35,
        save_result: bool = True,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """执行内窥镜病变检测分析"""
        try:
            print(f"EndoscopyDetectionTool: 开始处理图像 {image_path}")
            print(f"EndoscopyDetectionTool: 置信度阈值 {confidence}")
            
            # 检查文件是否存在
            if not os.path.exists(image_path):
                print(f"EndoscopyDetectionTool: 错误 - 图像不存在: {image_path}")
                raise FileNotFoundError(f"找不到图像文件: {image_path}")
            
            # 处理图像
            print(f"EndoscopyDetectionTool: 调用 process_image")
            results = self.process_image(image_path, confidence)
            
            # 保存输出图像
            output_filename = f"detection_result_{uuid.uuid4().hex[:8]}.png"
            output_path = self.temp_dir / output_filename
            print(f"EndoscopyDetectionTool: 将检测结果保存到 {output_path}")
            
            if save_result and "result_image" in results:
                results["result_image"].save(output_path)
                viz_path = str(output_path)
                print(f"EndoscopyDetectionTool: 已保存检测结果图像")
            else:
                viz_path = None
            
            # 生成检测结果描述 - 移除类别信息
            detection_text = ""
            if results["objects_detected"] > 0:
                detection_text = f"Detected {results['objects_detected']} lesion objects.\n\n"
                
                detection_text += "\nDetailed detection results:\n"
                for i, det in enumerate(results["detections"]):
                    detection_text += f"{i+1}. Confidence: {det['confidence']:.2f}\n"
            else:
                detection_text = "No lesion objects detected in this endoscopic image."
            
            # 构建输出结果 - 移除类别信息
            output = {
                "detection_image_path": viz_path,
                "objects_detected": results["objects_detected"],
                "detections": results["detections"],
                "description": detection_text,
                "confidence": confidence,
                "display_note": "Detection result image is displayed directly in the interface."
            }
            
            print(f"EndoscopyDetectionTool: 返回结果 - 检测到目标: {output['objects_detected']}")
            return output
            
        except Exception as e:
            error_msg = str(e)
            traceback_info = traceback.format_exc()
            print(f"检测出错: {error_msg}")
            print(traceback_info)
            error_output = {"error": error_msg}
            return error_output

    async def _arun(
        self,
        image_path: str,
        confidence: float = 0.35,
        save_result: bool = True,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """异步执行内窥镜病变检测分析"""
        return self._run(image_path, confidence, save_result)