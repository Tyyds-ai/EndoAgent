import os
import sys

# 添加必要的系统路径
sys.path.append('/path/to/EndoAgent')
sys.path.append('/path/to/EndoAgent/endoagent/tools/Config/UniEndo')
sys.path.append('/path/to/EndoAgent/endoagent/tools/Config/yolov8')
sys.path.append('/path/to/EndoAgent/endoagent/tools/Config/Polyp_Gen')
sys.path.append('/path/to/EndoAgent/endoagent/tools/Config/IntelliScope')

# # 设置 CUDA 环境变量，帮助调试
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # 使用指定的 GPU
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # 同步执行
import warnings
import logging
warnings.filterwarnings("ignore")
# 设置日志级别为WARNING，这会屏蔽所有INFO和DEBUG级别的日志输出
logging.basicConfig(level=logging.WARNING)

import PIL.Image
from typing import *
from dotenv import load_dotenv
from transformers import logging

from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI

# 导入自定义工具
from endoagent.tools.classification import EndoscopyClassifierTool
from endoagent.tools.segmentation import EndoscopySegmentationTool
from endoagent.tools.detection import EndoscopyDetectionTool
from endoagent.tools.generation_polyp import EndoscopyPolypGenerationTool
from endoagent.tools.generation_normal import EndoscopyNormalGenerationTool
from endoagent.tools.vqa import EndoscopyVQATool
from endoagent.tools.report_generation import EndoscopyReportGeneratorTool


# 假设我们使用与endoagent相同的代理和工具基础结构
from endoagent.agent import *
from endoagent.tools import *
from endoagent.utils import *

# 导入界面
# from interface_en import create_demo
from interface_zh import create_demo

warnings.filterwarnings("ignore")
logging.set_verbosity_error()
_ = load_dotenv()


def initialize_endo_agent(
    prompt_file, tools_to_use=None, model_weights_dir="endoagent/models", temp_dir="temp", device="cuda"
):
    """初始化内窥镜Agent，配置指定工具和配置。

    Args:
        prompt_file (str): 包含系统提示的文件路径
        tools_to_use (List[str], optional): 要初始化的工具名称列表。如果为None，则初始化所有工具。
        model_weights_dir (str, optional): 包含模型权重的目录。默认为"endoagent/models"。
        temp_dir (str, optional): 临时文件目录。默认为"temp"。
        device (str, optional): 运行模型的设备。默认为"cuda"。

    Returns:
        Tuple[Agent, Dict[str, BaseTool]]: 初始化的代理和工具实例字典
    """
    # 加载提示
    prompts = load_prompts_from_file(prompt_file)
    prompt = prompts.get("ENDOSCOPY_ASSISTANT", "You are an AI assistant specialized in analyzing endoscopy images.")

    # 初始化工具字典
    tools_dict = {}
    tools_to_use = tools_to_use or ["EndoscopyClassifierTool", "EndoscopySegmentationTool", "EndoscopyDetectionTool", 
                                    "EndoscopyImageVisualizerTool", "EndoscopyPolypGenerationTool", "EndoscopyNormalGenerationTool",
                                    "EndoscopyVQATool", "EndoscopyReportGeneratorTool"]
    
    # 按照依赖顺序初始化工具
    # 1. 首先初始化分类工具
    if "EndoscopyClassifierTool" in tools_to_use:
        tools_dict["EndoscopyClassifierTool"] = EndoscopyClassifierTool(
            weight_path=f"{model_weights_dir}/afacnet_cls.pth", 
            device=device
        )
    
    # 2. 初始化检测工具
    if "EndoscopyDetectionTool" in tools_to_use:
        tools_dict["EndoscopyDetectionTool"] = EndoscopyDetectionTool(
            model_path="endoagent/tools/Config/yolov8/logs/sgd-0.01/ep003-loss3.391-val_loss3.426.pth",
            classes_path="endoagent/tools/Config/yolov8/model_data/sun_classes.txt",
            device=device,
            temp_dir=temp_dir,
            confidence=0.35
        )
    
    # 3. 初始化分割工具
    if "EndoscopySegmentationTool" in tools_to_use:
        tools_dict["EndoscopySegmentationTool"] = EndoscopySegmentationTool(
            model_path=f"{model_weights_dir}/uniendo_seg/model_state_dict.pt",
            config_path=f"{model_weights_dir}/uniendo_seg/xdecoder_focall_my1_colon_lap.yaml",
            device=device,
            temp_dir=temp_dir
        )
    
    # 4. 初始化其他不依赖工具的基础工具
    if "EndoscopyImageVisualizerTool" in tools_to_use:
        tools_dict["EndoscopyImageVisualizerTool"] = ImageVisualizerTool()
    
    # 5. 初始化息肉生成和息肉去除工具 - 息肉去除工具依赖分割工具
    if "EndoscopyPolypGenerationTool" in tools_to_use:
        tools_dict["EndoscopyPolypGenerationTool"] = EndoscopyPolypGenerationTool(
            model_path="endoagent/tools/Config/Polyp_Gen/checkpoint",
            database_path="endoagent/tools/Config/Polyp_Gen/data/database/test_images.index",
            mask_dir="endoagent/tools/Config/Polyp_Gen/data/LDPolypVideo/Test/Masks",
            device=device,
            temp_dir=temp_dir
        )
    
    if "EndoscopyNormalGenerationTool" in tools_to_use:
        # 息肉去除工具依赖分割工具
        segmentation_tool = tools_dict.get("EndoscopySegmentationTool")
        tools_dict["EndoscopyNormalGenerationTool"] = EndoscopyNormalGenerationTool(
            model_path="endoagent/tools/Config/Polyp_Gen/checkpoint",
            database_path="endoagent/tools/Config/Polyp_Gen/data/database/test_images.index",
            mask_dir="endoagent/tools/Config/Polyp_Gen/data/LDPolypVideo/Test/Masks",
            segmentation_tool=segmentation_tool,
            device=device,
            temp_dir=temp_dir
        )
    
    # 5. 初始化VQA工具 - 依赖分类和检测工具
    if "EndoscopyVQATool" in tools_to_use:
        # 确保分类和检测工具已经初始化
        classifier_tool = tools_dict.get("EndoscopyClassifierTool")
        detection_tool = tools_dict.get("EndoscopyDetectionTool")
        
        # VQA工具初始化时传入分类和检测工具
        tools_dict["EndoscopyVQATool"] = EndoscopyVQATool(
            model_path="endoagent/tools/Config/IntelliScope/cache/checkpoint/ColonGPT-phi1.5-siglip-stg1",
            model_base="endoagent/tools/Config/IntelliScope/cache/downloaded-weights/phi-1.5",
            device=device,
            classifier_tool=classifier_tool,
            detection_tool=detection_tool
        )
    
    # 6. 初始化报告生成工具 - 依赖多个工具
    if "EndoscopyReportGeneratorTool" in tools_to_use:
        tools_dict["EndoscopyReportGeneratorTool"] = EndoscopyReportGeneratorTool(
            classifier_tool=tools_dict.get("EndoscopyClassifierTool"),
            segmentation_tool=tools_dict.get("EndoscopySegmentationTool"),
            detection_tool=tools_dict.get("EndoscopyDetectionTool"),
            vqa_tool=tools_dict.get("EndoscopyVQATool"),
            device=device,
            temp_dir=temp_dir
        )

    # 设置检查点和模型
    checkpointer = MemorySaver()
    model = ChatOpenAI(model="gpt-4o", temperature=0.7, top_p=0.95)
    
    # 初始化代理
    agent = Agent(
        model,
        tools=list(tools_dict.values()),
        log_tools=True,
        log_dir="endo_logs",
        system_prompt=prompt,
        checkpointer=checkpointer,
    )

    print("EndoAgent初始化完成")
    return agent, tools_dict

if __name__ == "__main__":
    """
    这是EndoAgent应用程序的主入口点。
    它使用选定的工具初始化代理并创建演示。
    """
    print("启动EndoAgent服务器...")

    # 选择要使用的工具
    selected_tools = [
        "EndoscopyClassifierTool",
        "EndoscopySegmentationTool",
        "EndoscopyImageVisualizerTool",
        "EndoscopyDetectionTool",
        "EndoscopyPolypGenerationTool",
        "EndoscopyNormalGenerationTool",
        "EndoscopyVQATool",
        "EndoscopyReportGeneratorTool"
        # 在这里添加更多工具
    ]

    # 创建系统提示文件路径
    prompt_file = "endoagent/docs/system_prompts.txt"
    
    # 确保提示文件所在目录存在
    os.makedirs(os.path.dirname(prompt_file), exist_ok=True)
    
    print("初始化EndoAgent和工具")
    agent, tools_dict = initialize_endo_agent(
        prompt_file, 
        tools_to_use=selected_tools
    )
    
    print("创建Gradio演示界面")
    # demo = create_demo(agent, tools_dict, title="EndoAgent - 内窥镜图像分析助手")
    
    # 是否启用增强功能
    # enable_enhancement = True
    # demo = create_demo(agent, tools_dict, enable_enhancement=enable_enhancement)
    demo = create_demo(agent, tools_dict)
    
    print("启动Gradio服务器")
    demo.launch(server_name="0.0.0.0", server_port=8686, share=True)