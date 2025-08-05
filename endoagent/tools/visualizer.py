from typing import Optional, Dict
from pydantic import BaseModel, Field

import os
import matplotlib.pyplot as plt
import skimage.io
from langchain_core.callbacks import CallbackManagerForToolRun, AsyncCallbackManagerForToolRun
from langchain_core.tools import BaseTool


class ImagePathInput(BaseModel):
    """Input for image visualization tool."""
    
    image_path: str = Field(..., description="Path to the image file to visualize")


class EndoscopyImageVisualizerTool(BaseTool):
    """Tool that visualizes endoscopy images and saves them as PNG files.
    
    This tool takes an image path, displays the image, and optionally saves it to a specified location.
    """
    
    name: str = "endoscopy_image_visualizer"
    description: str = (
        "A tool that visualizes endoscopy images. "
        "Input should be the path to an image file. "
        "The tool will display and optionally save the image."
    )
    # 添加 save_dir 字段
    save_dir: str = "temp/vis_images"
    
    def __init__(self, save_dir: str = "temp"):
        super().__init__()
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
    def _run(
        self,
        image_path: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict[str, str]:
        """Visualize the image and save it to the specified location.
        
        Args:
            image_path (str): Path to the image file to visualize.
            run_manager (Optional[CallbackManagerForToolRun]): Callback manager for the tool run.
            
        Returns:
            Dict[str, str]: A dictionary containing the file paths of the saved images.
        """
        try:
            # 读取图像
            image = skimage.io.imread(image_path)
            
            # 创建图像文件名
            base_name = os.path.basename(image_path)
            file_name = os.path.splitext(base_name)[0]
            save_path = os.path.join(self.save_dir, f"{file_name}_visualized.png")
            
            # 可视化并保存图像
            plt.figure(figsize=(10, 8))
            plt.imshow(image)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
            plt.close()
            
            return {
                "original_image": image_path,
                "visualized_image": save_path,
                "status": "success"
            }
            
        except Exception as e:
            return {
                "original_image": image_path,
                "status": "failed",
                "error": str(e)
            }
            
    async def _arun(
        self,
        image_path: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Dict[str, str]:
        """异步可视化图像。
        
        目前调用同步版本，因为可视化本身不是异步的。
        
        Args:
            image_path (str): 要可视化的图像文件的路径。
            run_manager (Optional[AsyncCallbackManagerForToolRun]): 工具运行的异步回调管理器。
            
        Returns:
            Dict[str, str]: 包含保存的图像文件路径的字典。
        """
        return self._run(image_path)