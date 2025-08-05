import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type, Any, Union
from pydantic import BaseModel, Field
import traceback
import numpy as np # 导入 numpy

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

# 导入其他工具
from endoagent.tools.classification import EndoscopyClassifierTool
from endoagent.tools.segmentation import EndoscopySegmentationTool
from endoagent.tools.detection import EndoscopyDetectionTool
from endoagent.tools.vqa import EndoscopyVQATool
from langchain_openai import ChatOpenAI # 确保导入 ChatOpenAI

# --- 添加辅助函数 ---
def convert_numpy_types(obj):
    """递归地将字典或列表中的 NumPy 类型转换为 Python 原生类型"""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list) or isinstance(obj, tuple):
        # 特别处理 classification_result 返回的元组结构
        if len(obj) == 2 and isinstance(obj[0], dict) and isinstance(obj[1], dict):
             return (convert_numpy_types(obj[0]), convert_numpy_types(obj[1]))
        return [convert_numpy_types(elem) for elem in obj]
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist() # 将 NumPy 数组转换为列表
    else:
        return obj
# --- 辅助函数结束 ---

class EndoscopyReportInput(BaseModel):
    """内窥镜报告生成的输入参数"""
    
    image_path: str = Field(
        ..., description="内窥镜图像文件路径"
    )
    patient_info: str = Field(
        "", description="患者基本信息(可选)，如'男，45岁，腹痛3天'"
    )
    clinical_history: str = Field(
        "", description="相关临床病史(可选)"
    )

class EndoscopyReportGeneratorTool(BaseTool):
    """内窥镜报告生成工具
    
    该工具会综合调用分割、检测和VQA工具，汇总所有结果，
    并使用GPT模型生成专业、全面的内窥镜检查报告。
    """
    
    name: str = "endoscopy_report_generator_tool"
    description: str = (
        "生成全面的内窥镜检查报告。此工具将：\n"
        "1. 自动检测并分析图像中的病变\n"
        "2. 进行精确的病变分类和分割\n"
        "3. 综合多种分析结果生成专业医学报告\n"
        "4. 提供诊断建议和后续治疗推荐\n"
        "输入应为内窥镜图像路径和可选的患者信息。"
    )
    args_schema: Type[BaseModel] = EndoscopyReportInput
    
    # --- 添加以下字段声明 ---
    classifier_tool: EndoscopyClassifierTool = Field(...)
    segmentation_tool: EndoscopySegmentationTool = Field(...)
    detection_tool: EndoscopyDetectionTool = Field(...)
    vqa_tool: EndoscopyVQATool = Field(...)
    temp_dir: str = Field(...)
    device: str = Field(...)
    gpt_model: ChatOpenAI = Field(...) # 注意类型是 ChatOpenAI

    # __init__ 方法保持不变，Pydantic 会自动处理赋值
    def __init__(
        self,
        classifier_tool: Optional[EndoscopyClassifierTool] = None,
        segmentation_tool: Optional[EndoscopySegmentationTool] = None,
        detection_tool: Optional[EndoscopyDetectionTool] = None,
        vqa_tool: Optional[EndoscopyVQATool] = None,
        temp_dir: str = "temp",
        device: str = "cuda",
        gpt_model: str = "gpt-4o",
        **kwargs: Any # 添加 **kwargs 以接收 BaseModel 的额外参数
    ):
        # 使用 super().__init__() 并传递参数给 BaseModel
        # 如果工具未提供，则创建默认实例
        model_weights_dir = "endoagent/models"
        _classifier_tool = classifier_tool or EndoscopyClassifierTool(
            weight_path=f"{model_weights_dir}/afacnet_cls.pth",
            device=device
        )
        _segmentation_tool = segmentation_tool or EndoscopySegmentationTool(
            model_path=f"{model_weights_dir}/uniendo_seg/model_state_dict.pt",
            config_path=f"{model_weights_dir}/uniendo_seg/xdecoder_focall_my1_colon_lap.yaml",
            device=device,
            temp_dir=temp_dir
        )
        _detection_tool = detection_tool or EndoscopyDetectionTool(
            model_path="endoagent/tools/Config/yolov8/logs/sgd-0.01/ep003-loss3.391-val_loss3.426.pth",
            classes_path="endoagent/tools/Config/yolov8/model_data/sun_classes.txt",
            device=device,
            temp_dir=temp_dir,
            confidence=0.35
        )
        _vqa_tool = vqa_tool or EndoscopyVQATool(
            model_path="endoagent/tools/Config/IntelliScope/cache/checkpoint/ColonGPT-phi1.5-siglip-stg1",
            model_base="endoagent/tools/Config/IntelliScope/cache/downloaded-weights/phi-1.5",
            device=device
        )
        _gpt_model_instance = ChatOpenAI(model=gpt_model, temperature=0.3)

        # 调用父类的 __init__ 方法，传递所有字段的值
        super().__init__(
            classifier_tool=_classifier_tool,
            segmentation_tool=_segmentation_tool,
            detection_tool=_detection_tool,
            vqa_tool=_vqa_tool,
            temp_dir=temp_dir,
            device=device,
            gpt_model=_gpt_model_instance,
            **kwargs # 传递其他可能的 BaseModel 参数
        )

        # 确保临时目录存在
        os.makedirs(self.temp_dir, exist_ok=True)
        print("内窥镜报告生成工具初始化完成")
    
    def _run_classification(self, image_path: str) -> Dict[str, Any]:
        """运行分类工具"""
        try:
            print(f"执行分类分析: {image_path}")
            # 使用 self.classifier_tool 访问实例
            result = self.classifier_tool._run(image_path)
            return result
        except Exception as e:
            print(f"分类分析失败: {str(e)}")
            return {"error": f"分类分析失败: {str(e)}"}
    
    # 对 _run_segmentation, _run_detection, _run_vqa 做类似修改
    def _run_segmentation(self, image_path: str) -> Dict[str, Any]:
        """运行分割工具"""
        try:
            print(f"执行分割分析: {image_path}")
            result = self.segmentation_tool._run(image_path) # 使用 self.segmentation_tool
            return result
        except Exception as e:
            print(f"分割分析失败: {str(e)}")
            return {"error": f"分割分析失败: {str(e)}"}

    def _run_detection(self, image_path: str) -> Dict[str, Any]:
        """运行检测工具"""
        try:
            print(f"执行检测分析: {image_path}")
            result = self.detection_tool._run(image_path) # 使用 self.detection_tool
            return result
        except Exception as e:
            print(f"检测分析失败: {str(e)}")
            return {"error": f"检测分析失败: {str(e)}"}

    def _run_vqa(self, 
                  image_path: str, 
                  classification_result: Any = None,
                  detection_result: Any = None) -> Dict[str, Any]:
        """运行VQA工具，传递分类和检测结果，避免重复分析
        
        Args:
            image_path: 图像路径
            classification_result: 分类工具的结果
            detection_result: 检测工具的结果
            
        Returns:
            Dict[str, Any]: VQA结果
        """
        try:
            print(f"执行VQA分析: {image_path}")
            question = "请详细描述这张内窥镜图像显示的情况，包括可见的任何解剖结构、病变或异常。"
            
            # 使用 self.vqa_tool 并传递分类和检测结果
            result = self.vqa_tool._run(
                image_path=image_path,
                question=question,
                classification_result=classification_result,
                detection_result=detection_result
            )
            
            return result
                
        except Exception as e:
            print(f"VQA分析失败: {str(e)}")
            print(traceback.format_exc())
            return {
                "error": f"VQA分析失败: {str(e)}",
                "status": "error",
                "vqa_results": [{
                    "question": question,
                    "answer": f"分析失败: {str(e)}"
                }]
            }
    
    def _generate_comprehensive_report(self, 
                                     image_path: str,
                                     classification_result: Any,
                                     segmentation_result: Any,
                                     detection_result: Any,
                                     vqa_result: Any,
                                     patient_info: str = "",
                                     clinical_history: str = "") -> str:
        """使用GPT模型生成综合报告"""
        try:
            # --- 在序列化之前转换 NumPy 类型 ---
            classification_result_serializable = convert_numpy_types(classification_result)
            segmentation_result_serializable = convert_numpy_types(segmentation_result)
            detection_result_serializable = convert_numpy_types(detection_result)
            vqa_result_serializable = convert_numpy_types(vqa_result)
            # --- 转换结束 ---
            
            # 判断是否是简化流程 - 修改为安全访问字典属性
            is_simplified = False
            
            # 检查segmentation_result是否为字典且有status键
            if isinstance(segmentation_result, dict) and "status" in segmentation_result:
                seg_status = segmentation_result.get("status") == "skipped"
            else:
                seg_status = False
                
            # 检查vqa_result是否为字典且有status键
            if isinstance(vqa_result, dict) and "status" in vqa_result:
                vqa_status = vqa_result.get("status") == "skipped"
            else:
                vqa_status = False
                
            is_simplified = seg_status and vqa_status
            
            # 根据不同流程调整提示信息
            if is_simplified:
                prompt = f"""
                你是一位经验丰富的内窥镜医学专家，需要根据以下分析结果生成一份内窥镜检查报告。
                初步分类和检测分析显示患者结果正常，因此跳过了更复杂的分析过程。
                请使用规范的医学术语，结构清晰，生成一份简洁明了的正常检查报告。
    
                ## 患者信息:
                {patient_info if patient_info else "未提供"}
                
                ## 临床病史:
                {clinical_history if clinical_history else "未提供"}
                
                ## 分类分析结果:
                {json.dumps(classification_result_serializable, ensure_ascii=False, indent=2)}
                
                ## 检测分析结果:
                {json.dumps(detection_result_serializable, ensure_ascii=False, indent=2)}
                
                请根据以上信息生成一份简洁的内窥镜检查报告，包括以下部分：
                1. 检查信息（检查类型、检查日期）
                2. 患者基本信息（如果提供）
                3. 临床病史（如果提供）
                4. 检查所见（描述观察到的正常情况）
                5. 诊断意见（正常所见）
                6. 建议（如有必要的随访建议）
                
                报告应使用专业但清晰的语言，简明扼要地说明检查结果正常。
                """
            else:
                prompt = f"""
                你是一位经验丰富的内窥镜医学专家，需要根据以下分析结果生成一份专业、全面的内窥镜检查报告。
                由于初步分析发现了潜在异常，因此执行了完整的分析流程，包括分割和详细描述分析。
                请使用规范的医学术语，结构清晰，包含所有重要发现和建议。
    
                ## 患者信息:
                {patient_info if patient_info else "未提供"}
                
                ## 临床病史:
                {clinical_history if clinical_history else "未提供"}
                
                ## 分类分析结果:
                {json.dumps(classification_result_serializable, ensure_ascii=False, indent=2)}
                
                ## 检测分析结果:
                {json.dumps(detection_result_serializable, ensure_ascii=False, indent=2)}
                
                ## 分割分析结果:
                {json.dumps(segmentation_result_serializable, ensure_ascii=False, indent=2)}
                
                ## 视觉问答分析结果:
                {json.dumps(vqa_result_serializable, ensure_ascii=False, indent=2)}
                
                请根据以上信息生成一份结构完整的内窥镜检查报告，包括以下部分：
                1. 检查信息（检查类型、检查日期）
                2. 患者基本信息（如果提供）
                3. 临床病史（如果提供）
                4. 检查所见（详细描述观察到的情况）
                5. 病变描述（如果有，包括位置、大小、形态、颜色等）
                6. 诊断意见
                7. 建议（包括后续随访、治疗建议等）
                
                报告应使用专业但清晰的语言，避免过于技术性的术语，以便患者也能理解。报告格式应整洁、规范。
                """
            
            # 使用 self.gpt_model 访问实例
            response = self.gpt_model.invoke(prompt)
            report = response.content.strip()
            return report
        except Exception as e:
            print(f"报告生成失败: {str(e)}")
            print(traceback.format_exc())
            # 返回更具体的错误信息
            if isinstance(e, TypeError) and "not JSON serializable" in str(e):
                 return f"报告生成失败: 内部数据序列化错误 ({str(e)})"
            return f"报告生成失败: {str(e)}"
    
    def _run(
        self,
        image_path: str,
        patient_info: str = "",
        clinical_history: str = "",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """执行内窥镜报告生成
        
        Args:
            image_path: 内窥镜图像路径
            patient_info: 患者基本信息（可选）
            clinical_history: 相关临床病史（可选）
            run_manager: 回调管理器
            
        Returns:
            Dict[str, Any]: 包含生成报告和分析结果的字典
        """
        try:
            start_time = time.time()
            print(f"开始生成内窥镜报告: {image_path}")
            
            # 检查图像路径
            if not os.path.exists(image_path):
                return {
                    "error": f"找不到图像文件: {image_path}",
                    "report": f"错误: 找不到图像文件: {image_path}"
                }
            
            # 1. 首先运行分类分析
            classification_result = self._run_classification(image_path)
            print("分类分析完成")
            
            # 2. 运行检测分析
            detection_result = self._run_detection(image_path)
            print("检测分析完成")
            
            # 3. 判断是否需要进一步分析
            # 检查分类结果是否为正常
            is_normal_classification = False
            if isinstance(classification_result, tuple) and len(classification_result) > 0:
                # 通常classification_result格式为: ({'Normal': 0.85, ...}, {...})
                if isinstance(classification_result[0], dict):
                    normal_prob = classification_result[0].get('Normal', 0)
                    # 如果Normal概率大于其他类别，则视为正常分类
                    is_normal_classification = normal_prob > 0.5 or normal_prob == max(classification_result[0].values())
                    print(f"分类结果：Normal概率 = {normal_prob}, 是否正常 = {is_normal_classification}")
            
            # 检查检测结果是否无异常
            is_normal_detection = False
            if isinstance(detection_result, tuple):
                # 元组第一个元素是结果字典
                if len(detection_result) > 0 and isinstance(detection_result[0], dict):
                    objects_detected = detection_result[0].get('objects_detected', 0)
                    is_normal_detection = objects_detected == 0
                    print(f"检测结果：检测到的对象数 = {objects_detected}, 是否正常 = {is_normal_detection}")
            elif isinstance(detection_result, dict):
                # 直接是字典
                objects_detected = detection_result.get('objects_detected', 0)
                is_normal_detection = objects_detected == 0
                print(f"检测结果：检测到的对象数 = {objects_detected}, 是否正常 = {is_normal_detection}")
            
            # 设置默认值，避免未定义错误
            segmentation_result = {"status": "skipped", "note": "跳过分割分析 - 分类和检测结果均为正常"}
            vqa_result = {"status": "skipped", "note": "跳过VQA分析 - 分类和检测结果均为正常"}
            
            # 如果分类结果为异常或检测结果有异常，继续进行分割和VQA分析
            if not (is_normal_classification and is_normal_detection):
                print("检测到异常，执行完整分析流程...")
                
                # 3. 运行分割分析
                segmentation_result = self._run_segmentation(image_path)
                print("分割分析完成")
                
                # 4. 运行VQA分析
                vqa_result = self._run_vqa(image_path)
                print("VQA分析完成")
            else:
                print("分类和检测结果均为正常，跳过分割和VQA分析...")
            
            # 5. 生成综合报告
            report = self._generate_comprehensive_report(
                image_path,
                classification_result,
                segmentation_result,
                detection_result,
                vqa_result,
                patient_info,
                clinical_history
            )
            
            # 检查报告生成是否实际失败
            if report.startswith("报告生成失败"):
                 print(f"报告生成步骤失败: {report}")
            else:
                print("报告生成完成")
            
            # 计算处理时间
            process_time = time.time() - start_time
            
            # 构建返回结果
            result = {
                "report": report,
                "classification_result": classification_result,
                "detection_result": detection_result,
                "segmentation_result": segmentation_result,
                "vqa_result": vqa_result,
                "image_path": image_path,
                "patient_info": patient_info,
                "clinical_history": clinical_history,
                "processing_time": f"{process_time:.2f}秒",
                "status": "success",
                "analysis_path": "simplified" if (is_normal_classification and is_normal_detection) else "complete"
            }
            
            # 如果报告生成失败，确保最终结果反映这一点
            if report.startswith("报告生成失败"):
                result["report"] = report # 更新报告内容为错误信息
                result["status"] = "error" # 更新状态
                if "error" not in result: # 添加错误键（如果不存在）
                    result["error"] = report.replace("报告生成失败: ", "")
    
            print(f"报告生成过程完成，最终状态: {result.get('status', 'unknown')}, 用时: {process_time:.2f}秒")
            return result
            
        except Exception as e:
            error_msg = str(e)
            traceback_info = traceback.format_exc()
            print(f"报告生成出错: {error_msg}")
            print(traceback_info)
            
            return {
                "error": error_msg,
                "report": f"报告生成失败: {error_msg}",
                "image_path": image_path,
                "status": "error"
            }
    
    async def _arun(
        self,
        image_path: str,
        patient_info: str = "",
        clinical_history: str = "",
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """异步执行内窥镜报告生成"""
        return self._run(image_path, patient_info, clinical_history, run_manager)