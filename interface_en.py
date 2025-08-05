import os
import re
import numpy as np
import gradio as gr
from pathlib import Path
import time
import shutil
from dotenv import load_dotenv
from typing import AsyncGenerator, List, Optional, Tuple
from gradio import ChatMessage

# 添加预设图像的路径常量（使用您指定的路径）
EXAMPLE_IMAGES_DIR = Path("/path/to/EndoAgent/demo/endoscopy")
os.makedirs(EXAMPLE_IMAGES_DIR, exist_ok=True)

load_dotenv()

class ChatInterface:
    """
    A chat interface for interacting with a medical AI agent through Gradio.

    Handles file uploads, message processing, and chat history management.
    Supports both regular image files and DICOM medical imaging files.
    """

    def __init__(self, agent, tools_dict):
        """
        Initialize the chat interface.

        Args:
            agent: The medical AI agent to handle requests
            tools_dict (dict): Dictionary of available tools for image processing
        """
        self.agent = agent
        self.tools_dict = tools_dict
        self.upload_dir = Path("temp")
        self.upload_dir.mkdir(exist_ok=True)
        self.current_thread_id = None
        # Separate storage for original and display paths
        self.original_file_path = None  # For LLM (.dcm or other)
        self.display_file_path = None  # For UI (always viewable format)

    def handle_upload(self, file_path: str) -> str:
        """
        Handle new file upload and set appropriate paths.

        Args:
            file_path (str): Path to the uploaded file

        Returns:
            str: Display path for UI, or None if no file uploaded
        """
        if not file_path:
            return None

        source = Path(file_path)
        timestamp = int(time.time())

        # Save original file with proper suffix
        suffix = source.suffix.lower()
        saved_path = self.upload_dir / f"upload_{timestamp}{suffix}"
        shutil.copy2(file_path, saved_path)  # Use file_path directly instead of source
        self.original_file_path = str(saved_path)

        # Handle DICOM conversion for display only
        if suffix == ".dcm":
            output, _ = self.tools_dict["DicomProcessorTool"]._run(str(saved_path))
            self.display_file_path = output["image_path"]
        else:
            self.display_file_path = str(saved_path)

        return self.display_file_path

    def add_message(
        self, message: str, display_image: str, history: List[dict]
    ) -> Tuple[List[dict], gr.Textbox]:
        """
        Add a new message to the chat history.

        Args:
            message (str): Text message to add
            display_image (str): Path to image being displayed
            history (List[dict]): Current chat history

        Returns:
            Tuple[List[dict], gr.Textbox]: Updated history and textbox component
        """
        # 添加调试输出
        print(f"收到消息: '{message}', 图像路径: '{display_image}'")
        print(f"历史记录项数: {len(history) if history else 0}")

        image_path = self.original_file_path or display_image

        if image_path is not None:
            print(f"添加图像路径: {image_path}")
            history.append({"role": "user", "content": {"path": image_path}})
        if message is not None:
            print(f"添加文本消息: {message}")
            history.append({"role": "user", "content": message})

        return history, gr.Textbox(value=message, interactive=False)

    async def process_message(
        self, message: str, display_image: Optional[str], chat_history: List[ChatMessage]
    ) -> AsyncGenerator[Tuple[List[ChatMessage], Optional[str], str], None]:
        """
        Process a message and generate responses.

        Args:
            message (str): User message to process
            display_image (Optional[str]): Path to currently displayed image
            chat_history (List[ChatMessage]): Current chat history

        Yields:
            Tuple[List[ChatMessage], Optional[str], str]: Updated chat history, display path, and empty string
        """
        # 添加调试输出
        print(f"开始处理消息: '{message}'")
        print(f"当前聊天历史记录数: {len(chat_history) if chat_history else 0}")

        chat_history = chat_history or []

        # 初始化线程
        if not self.current_thread_id:
            self.current_thread_id = str(time.time())
            print(f"创建新线程ID: {self.current_thread_id}")

        # 构建消息数组
        messages = []
        image_path = self.original_file_path or display_image
        if image_path is not None:
            print(f"图像路径: {image_path}")
            messages.append({"role": "user", "content": f"path: {image_path}"})
        if message is not None:
            print(f"文本消息: {message}")
            messages.append({"role": "user", "content": message})

        # 添加处理中提示
        chat_history.append(ChatMessage(role="assistant", content="正在处理您的请求..."))
        yield chat_history, self.display_file_path, ""

        # 移除"处理中"消息
        chat_history.pop()

        print(f"准备调用工作流，发送 {len(messages)} 条消息")

        try:
            for event in self.agent.workflow.stream(
                {"messages": messages}, {"configurable": {"thread_id": self.current_thread_id}}
            ):
                if isinstance(event, dict):
                    if "process" in event:
                        content = event["process"]["messages"][-1].content
                        if content:
                            content = re.sub(r"temp/[^\s]*", "", content)
                            chat_history.append(ChatMessage(role="assistant", content=content))
                            yield chat_history, self.display_file_path, ""

                    elif "execute" in event:
                        for message in event["execute"]["messages"]:
                            tool_name = message.name
                            content = eval(message.content)
                            # 判断是否是元组返回值（像分类工具）或单一字典（像VQA工具）
                            if isinstance(content, tuple) and len(content) > 0:
                                tool_result = content[0]
                            else:
                                tool_result = content
                            
                            print(f"工具执行结果: {tool_name} - {tool_result}")
                            
                            # 检查工具执行是否出错
                            if isinstance(tool_result, dict) and "error" in tool_result:
                                error_msg = tool_result.get("error", "未知错误")
                                traceback_info = tool_result.get("traceback", "无详细信息")
                                print(f"工具执行错误: {error_msg}")
                                print(f"错误详情: {traceback_info}")
                                
                                chat_history.append(
                                    ChatMessage(
                                        role="assistant",
                                        content=f"❌ {tool_name} 执行失败: {error_msg}",
                                        metadata={"title": "Error", "details": traceback_info}
                                    )
                                )
                            elif tool_result:
                                # 增强分割工具结果处理
                                # 检查是否为分割工具结果且有图像路径
                                if tool_name == "endoscopy_segmentation_tool" and "segmentation_image_path" in tool_result:
                                    # 先添加分割结果图像
                                    image_path = tool_result["segmentation_image_path"]
                                    chat_history.append(
                                        ChatMessage(
                                            role="assistant",
                                            content={"path": image_path},
                                        )
                                    )
                                    
                                    # 然后添加包含格式提示的文本描述
                                    if "description" in tool_result:
                                        description = tool_result["description"]
                                        segment_type = tool_result.get("segmentation_type", "病变")
                                        
                                        formatted_text = (
                                            f"内窥镜分割分析结果:\n\n"
                                            f"{description}\n\n"
                                            f"检测类型: {segment_type}\n"
                                            f"病变区域百分比: {tool_result['lesion_area_percentage']:.2f}%\n"
                                            f"检测到的病变数量: {tool_result['num_lesions']}\n\n"
                                            f"[图像已自动显示，无需插入Markdown图像链接]"  # 添加这个提示
                                        )
                                        
                                        chat_history.append(
                                            ChatMessage(
                                                role="assistant",
                                                content=formatted_text,
                                                metadata={"title": f"🖼️ 分割分析结果"},
                                            )
                                        )
                                # 添加检测工具结果处理
                                elif tool_name == "endoscopy_detection_tool" and "detection_image_path" in tool_result:
                                    # 先添加检测结果图像
                                    image_path = tool_result["detection_image_path"]
                                    if image_path:
                                        chat_history.append(
                                            ChatMessage(
                                                role="assistant",
                                                content={"path": image_path},
                                            )
                                        )

                                    # 然后添加检测结果描述
                                    if "description" in tool_result:
                                        description = tool_result["description"]
                                        objects_detected = tool_result.get("objects_detected", 0)

                                        formatted_text = (
                                            f"内窥镜病变检测结果:\n\n"
                                            f"{description}\n\n"
                                            f"检测使用的置信度阈值: {tool_result.get('confidence', 0.35):.2f}\n"
                                            f"[图像已自动显示，无需插入Markdown图像链接]"
                                        )

                                        chat_history.append(
                                            ChatMessage(
                                                role="assistant",
                                                content=formatted_text,
                                                metadata={"title": f"🖼️ 检测分析结果 ({objects_detected}个目标)"},
                                            )
                                        )
                                # 添加生成工具结果处理
                                elif tool_name == "endoscopy_generation_tool" and "generation_image_path" in tool_result:
                                    # 先添加生成结果图像
                                    image_path = tool_result["generation_image_path"]
                                    chat_history.append(
                                        ChatMessage(
                                            role="assistant",
                                            content={"path": image_path},
                                        )
                                    )
                                    
                                    # 然后添加掩码图像(可选)
                                    if "mask_image_path" in tool_result:
                                        mask_path = tool_result["mask_image_path"]
                                        chat_history.append(
                                            ChatMessage(
                                                role="assistant",
                                                content={"path": mask_path},
                                                metadata={"title": "息肉生成掩码"}
                                            )
                                        )
                                    
                                    # 最后添加包含格式提示的文本描述
                                    if "description" in tool_result:
                                        description = tool_result["description"]
                                        formatted_text = (
                                            f"内窥镜息肉生成结果:\n\n"
                                            f"{description}\n\n"
                                            f"生成提示词: {tool_result['prompt']}\n"
                                            f"生成步数: {tool_result['steps']}\n\n"
                                            f"[图像已自动显示，无需插入Markdown图像链接]"
                                        )
                                        
                                        chat_history.append(
                                            ChatMessage(
                                                role="assistant",
                                                content=formatted_text,
                                                metadata={"title": f"🖼️ 息肉生成结果"},
                                            )
                                        )
                                # 添加VQA工具结果处理
                                elif tool_name == "endoscopy_vqa_tool" and isinstance(tool_result, dict):
                                    if "error" in tool_result:
                                        error_msg = tool_result.get("error", "未知错误")
                                        chat_history.append(
                                            ChatMessage(
                                                role="assistant",
                                                content=f"❌ VQA分析失败: {error_msg}",
                                                metadata={"title": "错误"}
                                            )
                                        )
                                    else:
                                        # 获取VQA结果
                                        response = tool_result.get("result", 
                                                     tool_result.get("description", "无法获取分析结果"))
                                        question = tool_result.get("question", "")
                                        
                                        # 格式化VQA回答
                                        formatted_text = (
                                            f"内窥镜图像分析结果:\n\n"
                                            f"{response}\n\n"
                                        )
                                        
                                        if question:
                                            formatted_text += f"分析问题: {question}\n"
                                        
                                        # 添加到聊天历史
                                        chat_history.append(
                                            ChatMessage(
                                                role="assistant",
                                                content=formatted_text,
                                                metadata={"title": "🔍 图像问答分析"}
                                            )
                                        )
                                elif tool_name == "endoscopy_report_generator_tool" and "report" in tool_result:
                                    report_text = tool_result["report"]
                                    
                                    # 添加报告到聊天历史
                                    chat_history.append(
                                        ChatMessage(
                                            role="assistant",
                                            content=report_text,
                                            metadata={"title": "📋 内窥镜检查报告"}
                                        )
                                    )
                                # 常规工具结果处理    
                                else:
                                    metadata = {"title": f"🖼️ Image from tool: {tool_name}"}
                                    formatted_result = " ".join(
                                        line.strip() for line in str(tool_result).splitlines()
                                    ).strip()
                                    metadata["description"] = formatted_result
                                    chat_history.append(
                                        ChatMessage(
                                            role="assistant",
                                            content=formatted_result,
                                            metadata=metadata,
                                        )
                                    )
                    
                                # For image_visualizer, use display path
                                if tool_name == "image_visualizer":
                                    self.display_file_path = tool_result["image_path"]
                                    chat_history.append(
                                        ChatMessage(
                                            role="assistant",
                                            # content=gr.Image(value=self.display_file_path),
                                            content={"path": self.display_file_path},
                                        )
                                    )
                    
                                yield chat_history, self.display_file_path, ""
        except Exception as e:
            chat_history.append(
                ChatMessage(
                    role="assistant", content=f"❌ Error: {str(e)}", metadata={"title": "Error"}
                )
            )
            yield chat_history, self.display_file_path, ""


def create_demo(agent, tools_dict):
    """
    Create a Gradio demo interface for the medical AI agent.

    Args:
        agent: The medical AI agent to handle requests
        tools_dict (dict): Dictionary of available tools for image processing

    Returns:
        gr.Blocks: Gradio Blocks interface
    """
    print("Creating Gradio demo interface")
    interface = ChatInterface(agent, tools_dict)

    # Define medical endoscopy theme colors
    theme = gr.themes.Soft(
        primary_hue="teal",     # Endoscopy medical green tone
        secondary_hue="blue",   # Medical blue as secondary color
        neutral_hue="slate"     # Gray neutral tone for professional look
    )

    with gr.Blocks(theme=theme) as demo:
        with gr.Column():
            gr.Markdown(
                """
                # 🔬 EndoAgent - Endoscopy Intelligent Assistant
                ## Endoscopic Image Analysis & Report Generation System

                *Supporting lesion detection, lesion segmentation, lesion classification, endoscopic image analysis, endoscopic image generation, and report generation.*
                """
            )

            with gr.Row():
                # Left side chat area
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(
                        [],
                        height=700,
                        container=True,
                        show_label=True,
                        elem_classes="chat-box",
                        type="messages",
                        label="Endoscopy Analysis Assistant",
                        avatar_images=(
                            None,
                            "assets/endoagent_logo.png",
                        ),
                    )
                    with gr.Row():
                        with gr.Column(scale=4):
                            txt = gr.Textbox(
                                show_label=False,
                                placeholder="Enter instructions, e.g.: analyze this endoscopy image, detect lesions, generate report...",
                                container=False,
                                lines=2
                            )
                        with gr.Column(scale=1, min_width=100):
                            send_btn = gr.Button("Send 📤", variant="primary")

                    # Quick command buttons
                    with gr.Row():
                        gr.Markdown("### Quick Commands")
                    
                    # First row: Image analysis and lesion classification
                    with gr.Row():
                        analyze_btn = gr.Button("📊 Analyze Image", size="sm", scale=1)
                        classify_btn = gr.Button("🏷️ Classify Lesion", size="sm", scale=1)
                    
                    # Second row: Detect and segment lesions
                    with gr.Row():
                        detect_btn = gr.Button("🔍 Detect Lesions", size="sm", scale=1)
                        segment_btn = gr.Button("✂️ Segment Lesion", size="sm", scale=1)
                    
                    # Third row: Generate polyp and remove polyp
                    with gr.Row():
                        generate_polyp_btn = gr.Button("🚨 Generate Polyp", size="sm", scale=1)
                        generate_normal_btn = gr.Button("🛡️ Remove Polyp", size="sm", scale=1)

                    # Fourth row: Generate report (occupying entire row) 
                    with gr.Row():
                        report_btn = gr.Button("📝 Generate Report", size="sm", scale=2)

                # Right side image area
                with gr.Column(scale=3):
                    image_display = gr.Image(
                        label="Endoscopy Image",
                        type="filepath", 
                        height=550, 
                        container=True,
                    )
                    
                    # Image upload area
                    with gr.Row():
                        upload_button = gr.UploadButton(
                            "📷 Upload Endoscopy Image",
                            file_types=["image"],
                            variant="primary"
                        )
                        # dicom_upload = gr.UploadButton(
                            # "📄 Upload DICOM File",
                            # file_types=["file"],
                        # )
                        
                    # Status and control area
                    with gr.Row():
                        clear_btn = gr.Button("🧹 Clear Chat")
                        new_thread_btn = gr.Button("🔄 New Session")

                    # Add system information area
                    with gr.Row():
                        gr.Markdown(
                            """
                            ### System Information
                            * Supports common endoscopy image formats: JPG, PNG
                            * Features: lesion detection, lesion segmentation, lesion classification, lesion analysis, image generation, and report generation
                            * Classification categories: normal, hyperplastic, adenomatous, and malignant lesions
                            * AI assists diagnosis; final diagnosis should rely on physician judgment
                            """
                        )

        # Event handlers - keep original logic
        def clear_chat():
            interface.original_file_path = None
            interface.display_file_path = None
            return [], None

        def new_thread():
            interface.current_thread_id = str(time.time())
            return [], interface.display_file_path

        def handle_file_upload(file):
            return interface.handle_upload(file.name)

        # Quick button functionality implementation
        def set_analyze_prompt():
            image_path = str(EXAMPLE_IMAGES_DIR / "vqa.png")
            return "Please analyze the visible features and structures in this endoscopy image", image_path if os.path.exists(image_path) else None
                    
        def set_classify_prompt():
            image_path = str(EXAMPLE_IMAGES_DIR / "classify.jpg")
            return "Please classify the lesion in this endoscopy image", image_path if os.path.exists(image_path) else None
                    
        def set_detect_prompt():
            image_path = str(EXAMPLE_IMAGES_DIR / "detect.jpg") 
            return "Detect lesions in this endoscopy image", image_path if os.path.exists(image_path) else None
                    
        def set_segment_prompt():
            image_path = str(EXAMPLE_IMAGES_DIR / "segment.jpg")
            return "Segment the lesion in this endoscopy image", image_path if os.path.exists(image_path) else None

        def set_generate_polyp_prompt():
            image_path = str(EXAMPLE_IMAGES_DIR / "generate_polyp.jpg")
            return "Generate a realistic lesion in this endoscopy image", image_path if os.path.exists(image_path) else None
                    
        def set_generate_normal_prompt():
            image_path = str(EXAMPLE_IMAGES_DIR / "generate_normal.jpg")
            return "Remove the lesion from this endoscopy image and generate a normal tissue image", image_path if os.path.exists(image_path) else None

        def set_report_prompt():
            image_path = str(EXAMPLE_IMAGES_DIR / "mrg.jpg")
            return "Generate a detailed report based on the analysis of this endoscopy image", image_path if os.path.exists(image_path) else None
    
        def load_preset(prompt, image_path):
            """处理预设的提示词和图像路径"""
            display_path = None
            if image_path and os.path.exists(image_path):
                # 将图像复制到临时目录使其可用
                temp_dir = Path("temp")
                temp_dir.mkdir(exist_ok=True)
                timestamp = int(time.time())
                filename = f"preset_{timestamp}{Path(image_path).suffix}"
                saved_path = temp_dir / filename
                shutil.copy2(image_path, saved_path)
                display_path = str(saved_path)
                
                # 设置为原始文件路径，这样其他工具可以使用
                interface.original_file_path = display_path
                interface.display_file_path = display_path
                
                print(f"已加载预设图像: {image_path} -> {display_path}")
            else:
                print(f"预设图像不存在: {image_path}")
            return prompt, display_path
            
        # Debug function
        def debug_input(message, display_image):
            print(f"Debug - Received message: '{message}'")
            print(f"Debug - Display image: '{display_image}'")
            return message

        # Add debug before binding events
        txt.submit(debug_input, inputs=[txt, image_display], outputs=[])
        send_btn.click(debug_input, inputs=[txt, image_display], outputs=[])

        # Keep the original message processing logic
        chat_msg = txt.submit(
            interface.add_message, inputs=[txt, image_display, chatbot], outputs=[chatbot, txt]
        ).then(
            interface.process_message,
            inputs=[txt, image_display, chatbot],
            outputs=[chatbot, image_display, txt],
        ).then(lambda: gr.Textbox(interactive=True), None, [txt])
        
        # Send button binds to the same processing logic
        send_btn.click(
            interface.add_message, inputs=[txt, image_display, chatbot], outputs=[chatbot, txt]
        ).then(
            interface.process_message,
            inputs=[txt, image_display, chatbot],
            outputs=[chatbot, image_display, txt],
        ).then(lambda: gr.Textbox(interactive=True), None, [txt])

        # Quick button bindings
        analyze_btn.click(set_analyze_prompt, outputs=[txt, image_display]).then(
            load_preset, inputs=[txt, image_display], outputs=[txt, image_display]
        )
        classify_btn.click(set_classify_prompt, outputs=[txt, image_display]).then(
            load_preset, inputs=[txt, image_display], outputs=[txt, image_display]
        )
        detect_btn.click(set_detect_prompt, outputs=[txt, image_display]).then(
            load_preset, inputs=[txt, image_display], outputs=[txt, image_display]
        )
        segment_btn.click(set_segment_prompt, outputs=[txt, image_display]).then(
            load_preset, inputs=[txt, image_display], outputs=[txt, image_display]
        )
        generate_polyp_btn.click(set_generate_polyp_prompt, outputs=[txt, image_display]).then(
            load_preset, inputs=[txt, image_display], outputs=[txt, image_display]
        )
        generate_normal_btn.click(set_generate_normal_prompt, outputs=[txt, image_display]).then(
            load_preset, inputs=[txt, image_display], outputs=[txt, image_display]
        )
        report_btn.click(set_report_prompt, outputs=[txt, image_display]).then(
            load_preset, inputs=[txt, image_display], outputs=[txt, image_display]
        )

        # File upload
        upload_button.upload(handle_file_upload, inputs=upload_button, outputs=image_display)
        # dicom_upload.upload(handle_file_upload, inputs=dicom_upload, outputs=image_display)

        # Clear and new session
        clear_btn.click(clear_chat, outputs=[chatbot, image_display])
        new_thread_btn.click(new_thread, outputs=[chatbot, image_display])

    return demo