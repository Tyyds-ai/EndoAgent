import os
import sys
import json
import time
import warnings
from typing import Optional, List, Dict, Any
from pathlib import Path
import traceback
import tempfile
import base64
from PIL import Image
import io

# Suppress warnings
warnings.filterwarnings("ignore")

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver

from .base import BaseAPI

try:
    from endoagent.agent import Agent
    from endoagent.tools.classification import EndoscopyClassifierTool
    from endoagent.tools.segmentation import EndoscopySegmentationTool
    from endoagent.tools.detection import EndoscopyDetectionTool
    from endoagent.tools.vqa import EndoscopyVQATool
    from endoagent.tools.report_generation import EndoscopyReportGeneratorTool
    from endoagent.utils import load_prompts_from_file
except ImportError as e:
    print(f"Warning: Failed to import EndoAgent components: {e}")
    print("Please ensure EndoAgent is properly installed and paths are correct.")



# 多轮Reflexion智能决策与自我优化Agent
class EndoAgentReflexionWrapper(BaseAPI):
    """
    EndoAgent多轮Reflexion智能决策与自我优化Agent，集成VLMEval框架。
    """
    is_api: bool = True
    allowed_types = ['text', 'image']

    def __init__(
        self,
        prompt_file: str = "EndoAgent/endoagent/docs/system_prompts.txt",
        model_weights_dir: str = "EndoAgent/endoagent/models",
        temp_dir: str = "temp",
        device: str = "cuda",
        tools_to_use: Optional[List[str]] = None,
        model_name: str = "gpt-4o",
        temperature: float = 0.7,
        top_p: float = 0.95,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        max_tokens: int = 2048,
        retry: int = 5,
        verbose: bool = True,
        max_reflexion_rounds: int = 3,
        **kwargs
    ):
        super().__init__(retry=retry, verbose=verbose, **kwargs)
        self.prompt_file = prompt_file
        self.model_weights_dir = model_weights_dir
        self.temp_dir = Path(temp_dir)
        self.device = device
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.dump_image = False
        self.max_reflexion_rounds = max_reflexion_rounds
        self.verbose = verbose

        self.temp_dir.mkdir(exist_ok=True)

        if tools_to_use is None:
            self.tools_to_use = [
                "EndoscopyClassifierTool",
                "EndoscopySegmentationTool",
                "EndoscopyDetectionTool",
                "EndoscopyVQATool",
                "EndoscopyReportGeneratorTool"
            ]
        else:
            self.tools_to_use = tools_to_use

        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        if api_base:
            os.environ["OPENAI_BASE_URL"] = api_base

        self._initialize_agent()

    def _initialize_agent(self):
        if self.verbose:
            print("Initializing EndoAgent...")
        try:
            try:
                prompts = load_prompts_from_file(self.prompt_file)
                self.system_prompt = prompts.get(
                    "ENDOSCOPY_ASSISTANT",
                    "You are an AI assistant specialized in analyzing endoscopy images."
                )
                if self.verbose:
                    print(f"Loaded prompt from: {self.prompt_file}")
            except Exception as e:
                if self.verbose:
                    print(f"Failed to load prompts: {e}, using default")
                self.system_prompt = "You are an AI assistant specialized in analyzing endoscopy images."

            self.tools_dict = self._initialize_tools()
            self.llm = ChatOpenAI(
                model=self.model_name,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens
            )
            self.checkpointer = MemorySaver()
            self.agent = Agent(
                self.llm,
                tools=list(self.tools_dict.values()),
                log_tools=True,
                log_dir="endo_logs",
                system_prompt=self.system_prompt,
                checkpointer=self.checkpointer,
            )
            if self.verbose:
                print("EndoAgent initialized successfully!")
        except Exception as e:
            print(f"Failed to initialize EndoAgent: {e}")
            print(traceback.format_exc())
            raise

    def _initialize_tools(self) -> Dict[str, Any]:
        tools_dict = {}
        if self.verbose:
            print(f"Initializing tools: {', '.join(self.tools_to_use)}")
        try:
            if "EndoscopyClassifierTool" in self.tools_to_use:
                if self.verbose:
                    print("  - Initializing EndoscopyClassifierTool...")
                tools_dict["EndoscopyClassifierTool"] = EndoscopyClassifierTool(
                    weight_path=os.path.join("/path/to/EndoAgent", self.model_weights_dir.replace("EndoAgent/", ""), "afacnet_cls.pth"),
                    device=self.device
                )
            if "EndoscopyDetectionTool" in self.tools_to_use:
                if self.verbose:
                    print("  - Initializing EndoscopyDetectionTool...")
                model_path_det = "/path/to/EndoAgent/endoagent/tools/Config/yolov8/logs/sgd-0.01/ep003-loss3.391-val_loss3.426.pth"
                classes_path_det = "/path/to/EndoAgent/endoagent/tools/Config/yolov8/model_data/sun_classes.txt"
                tools_dict["EndoscopyDetectionTool"] = EndoscopyDetectionTool(
                    model_path=model_path_det,
                    classes_path=classes_path_det,
                    device=self.device,
                    temp_dir=str(self.temp_dir),
                    confidence=0.35
                )
            if "EndoscopySegmentationTool" in self.tools_to_use:
                if self.verbose:
                    print("  - Initializing EndoscopySegmentationTool...")
                model_path_seg = "/path/to/EndoAgent/endoagent/models/uniendo_seg/model_state_dict.pt"
                config_path_seg = "/path/to/EndoAgent/endoagent/models/uniendo_seg/xdecoder_focall_my1_colon_lap.yaml"
                tools_dict["EndoscopySegmentationTool"] = EndoscopySegmentationTool(
                    model_path=model_path_seg,
                    config_path=config_path_seg,
                    device=self.device,
                    temp_dir=str(self.temp_dir)
                )
            if "EndoscopyVQATool" in self.tools_to_use:
                if self.verbose:
                    print("  - Initializing EndoscopyVQATool...")
                classifier_tool = tools_dict.get("EndoscopyClassifierTool")
                detection_tool = tools_dict.get("EndoscopyDetectionTool")
                model_path_vqa = "/path/to/EndoAgent/endoagent/tools/Config/IntelliScope/cache/checkpoint/ColonGPT-phi1.5-siglip-stg1"
                model_base_vqa = "/path/to/EndoAgent/endoagent/tools/Config/IntelliScope/cache/downloaded-weights/phi-1.5"
                tools_dict["EndoscopyVQATool"] = EndoscopyVQATool(
                    model_path=model_path_vqa,
                    model_base=model_base_vqa,
                    device=self.device,
                    classifier_tool=classifier_tool,
                    detection_tool=detection_tool
                )
            if "EndoscopyReportGeneratorTool" in self.tools_to_use:
                if self.verbose:
                    print("  - Initializing EndoscopyReportGeneratorTool...")
                classifier_tool = tools_dict.get("EndoscopyClassifierTool")
                segmentation_tool = tools_dict.get("EndoscopySegmentationTool")
                detection_tool = tools_dict.get("EndoscopyDetectionTool")
                vqa_tool = tools_dict.get("EndoscopyVQATool")
                tools_dict["EndoscopyReportGeneratorTool"] = EndoscopyReportGeneratorTool(
                    classifier_tool=classifier_tool,
                    segmentation_tool=segmentation_tool,
                    detection_tool=detection_tool,
                    vqa_tool=vqa_tool,
                    device=self.device,
                    temp_dir=str(self.temp_dir)
                )
            if self.verbose:
                print(f"Successfully initialized {len(tools_dict)} tools")
        except Exception as e:
            print(f"Failed to initialize tools: {e}")
            print(traceback.format_exc())
            raise
        return tools_dict

    def generate_inner(self, inputs, **kwargs):
        """
        多轮Reflexion智能决策主入口，支持VLMEval调用。
        Args:
            inputs: List of input messages in VLMEval格式
            **kwargs: 其他参数
        Returns:
            tuple: (ret_code, response, log)
        """
        try:
            image_path = None
            question_text = ""
            for item in inputs:
                if item['type'] == 'image':
                    image_path = item['value']
                elif item['type'] == 'text':
                    question_text += item['value'] + " "
            question_text = question_text.strip()

            # 初始化记忆体
            short_term_memory = []  # [(action, observation)]
            long_term_memory = []   # [reflexion]
            log_list = []
            response = None

            for round_idx in range(self.max_reflexion_rounds):
                # 1. 执行智能决策
                action, observation = self._execute_step(question_text, image_path, short_term_memory, long_term_memory)
                short_term_memory.append((action, observation))
                log_list.append(f"Round {round_idx+1} Action: {action}\nObservation: {observation}")
                if self.verbose:
                    print(f"[Reflexion] Round {round_idx+1} Action: {action}")
                    print(f"[Reflexion] Round {round_idx+1} Observation: {observation}")

                # 2. 生成反思提示，调用模型自我优化
                reflexion_prompt = self._build_reflexion_prompt(question_text, short_term_memory, long_term_memory)
                reflexion = self._reflexion_step(reflexion_prompt)
                long_term_memory.append(reflexion)
                log_list.append(f"Round {round_idx+1} Reflexion: {reflexion}")
                if self.verbose:
                    print(f"[Reflexion] Round {round_idx+1} Reflexion: {reflexion}")

                # 3. 判断是否终止
                if self._is_task_complete(observation, reflexion):
                    response = observation
                    break

            # 返回最终结果
            # return 0, response or "No valid response", "\n".join(log_list)
            return 0, response if response is not None else short_term_memory[-1][1], "\n".join(log_list)
        except Exception as e:
            error_msg = f"EndoAgentReflexion failed: {str(e)}"
            if self.verbose:
                print(error_msg)
                print(traceback.format_exc())
            return 1, error_msg, str(e)

    def _execute_step(self, question, image_path, short_term_memory, long_term_memory):
        """
        执行一次智能决策，调用EndoAgent工具链。
        Returns: (action, observation)
        """
        messages = []
        temp_image_path = None
        # 检查 image_path 是否为 base64 或 data URI
        if image_path:
            if isinstance(image_path, str) and image_path.startswith("data:image"):
                # 解析 base64 数据
                try:
                    header, b64data = image_path.split(",", 1)
                    img_bytes = base64.b64decode(b64data)
                    # 保存为临时文件
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg", dir=str(self.temp_dir)) as tmp_img:
                        tmp_img.write(img_bytes)
                        temp_image_path = tmp_img.name
                    messages.append(HumanMessage(content=f"path: {temp_image_path}"))
                except Exception as e:
                    if self.verbose:
                        print(f"Failed to decode and save base64 image: {e}")
                    messages.append(HumanMessage(content=f"[Warning: Failed to decode image: {e}]"))
            elif os.path.exists(image_path):
                messages.append(HumanMessage(content=f"path: {image_path}"))
            else:
                messages.append(HumanMessage(content=f"[Warning: Image path not found: {image_path}]"))
        if question:
            messages.append(HumanMessage(content=question))
        if long_term_memory:
            messages.append(SystemMessage(content="Previous Reflexions: " + " | ".join(long_term_memory)))
        if short_term_memory:
            actions_str = " | ".join([f"Action: {a}, Obs: {o}" for a, o in short_term_memory])
            messages.append(SystemMessage(content="Action Chain: " + actions_str))

        # 自动将工具输出图片转 base64 并传递给 LLM
        # 检查 short_term_memory/observation/action 是否包含图片路径
        def encode_image_to_base64(img_path):
            try:
                with open(img_path, "rb") as f:
                    img_bytes = f.read()
                img_b64 = base64.b64encode(img_bytes).decode("utf-8")
                return img_b64
            except Exception as e:
                return None

        # 收集所有图片路径（如工具输出的 result_image/detection_image_path/segmentation_image_path/generated_path 等）
        image_keys = ["result_image", "detection_image_path", "generated_path", "segmentation_image_path"]
        for mem in short_term_memory:
            # mem = (action, observation)
            for obj in mem:
                if isinstance(obj, dict):
                    for k in image_keys:
                        if k in obj and isinstance(obj[k], str) and os.path.exists(obj[k]):
                            img_b64 = encode_image_to_base64(obj[k])
                            if img_b64:
                                messages.append(HumanMessage(content=f"tool_image_base64:{k}:{img_b64}"))

        thread_id = f"reflexion_{int(time.time() * 1000)}"
        for attempt in range(self.retry):
            try:
                response = self.agent.workflow.invoke(
                    {"messages": messages},
                    {"configurable": {"thread_id": thread_id}}
                )
                final_answer = self._extract_final_answer(response)
                # 清理临时图片文件
                if temp_image_path and os.path.exists(temp_image_path):
                    try:
                        os.remove(temp_image_path)
                    except Exception:
                        pass
                return final_answer, final_answer
            except Exception as e:
                if self.verbose:
                    print(f"Attempt {attempt + 1} failed: {e}")
                if attempt == self.retry - 1:
                    # 清理临时图片文件
                    if temp_image_path and os.path.exists(temp_image_path):
                        try:
                            os.remove(temp_image_path)
                        except Exception:
                            pass
                    raise
                time.sleep(1)

    def _build_reflexion_prompt(self, question, short_term_memory, long_term_memory):
        """
        构造Reflexion反思提示，参考ReflexionAgent。
        """
        prompt = f"Task: {question}\n"
        if short_term_memory:
            prompt += "Action Chain:\n" + "\n".join([f"Action: {a}, Obs: {o}" for a, o in short_term_memory]) + "\n"
        if long_term_memory:
            prompt += "Previous Reflexions:\n" + "\n".join(long_term_memory) + "\n"
        prompt += "请自我反思并优化下一步决策。"
        return prompt

    def _reflexion_step(self, prompt):
        """
        调用大模型生成反思内容。
        """
        for attempt in range(self.retry):
            try:
                response = self.llm.invoke([HumanMessage(content=prompt)])
                if isinstance(response, AIMessage):
                    return response.content
                elif isinstance(response, str):
                    return response
                else:
                    return str(response)
            except Exception as e:
                if self.verbose:
                    print(f"Reflexion attempt {attempt + 1} failed: {e}")
                if attempt == self.retry - 1:
                    raise
                time.sleep(1)

    def _is_task_complete(self, observation, reflexion):
        """
        判断任务是否完成，可根据observation/reflexion内容自定义。
        """
        # 简单判断：如果observation包含done/完成/结束等关键词
        obs_str = str(observation).lower()
        if any(x in obs_str for x in ["done", "完成", "结束"]):
            return True
        return False

    def _extract_final_answer(self, response: Dict[str, Any]) -> str:
        try:
            if isinstance(response, dict) and "messages" in response:
                messages_list = response["messages"]
                if isinstance(messages_list, list):
                    for msg in reversed(messages_list):
                        if isinstance(msg, AIMessage) and msg.content:
                            content = msg.content
                            if isinstance(content, str):
                                return content
                            elif isinstance(content, list):
                                return " ".join(str(item) for item in content)
                            else:
                                return str(content)
                    tool_results = []
                    for msg in messages_list:
                        if hasattr(msg, "type") and msg.type == "tool":
                            content = msg.content
                            if isinstance(content, str):
                                tool_results.append(content)
                            elif isinstance(content, list):
                                tool_results.append(" ".join(str(item) for item in content))
                            else:
                                tool_results.append(str(content))
                    if tool_results:
                        return "Tool results: " + "; ".join(tool_results)
            response_str = str(response)
            return response_str[:1000] if len(response_str) > 1000 else response_str
        except Exception as e:
            return f"Error extracting answer: {str(e)}"

    def use_custom_prompt(self, dataset_name: str) -> bool:
        return False

    def set_dump_image(self, dump_image: bool):
        self.dump_image = dump_image
        if self.verbose:
            print(f"Set dump_image to: {dump_image}")

    def build_prompt(self, line, dataset):
        return dataset.build_prompt(line)

    def chat_inner(self, message, dataset=None):
        return self.generate_inner(message)

    @property
    def model_name_property(self) -> str:
        return f"EndoAgentReflexion-{self.model_name}"

    def __repr__(self) -> str:
        return f"EndoAgentReflexionWrapper(model={self.model_name}, tools={len(self.tools_to_use)})"

# For backward compatibility
EndoAgentReflexion = EndoAgentReflexionWrapper
