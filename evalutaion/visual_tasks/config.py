from vlmeval.vlm import *
from vlmeval.api import *
from functools import partial

api_models = {
    # GPT
    "GPT4o_20241120": partial(
        GPT4V,
        model="gpt-4o-2024-11-20",
        temperature=0,
        img_size=-1,
        img_detail="high",
        retry=10,
        verbose=False,
    ),

    # Gemini
    "GeminiPro2-5-OpenAI": partial(
        GeminiPro25,
        temperature=0,
        retry=10,
        api_key=None,  # Set via GEMINI_API_KEY environment variable
        api_base="https://api2.aigcbest.top/v1"  # Your custom endpoint
    ),
    
    # Qwen-VL
    "QwenVLPlus": partial(QwenVLAPI, model="qwen-vl-plus", temperature=0, retry=10),

    # Step
    "Step1o": partial(
        GPT4V,
        model="step-1o-vision-32k",
        api_base="https://api.stepfun.com/v1/chat/completions",
        temperature=0,
        retry=10,
        img_size=-1,
        img_detail="high",
    ),

    # Yi-Vision
    "Yi-Vision": partial(
        GPT4V,
        model="yi-vision",
        api_base="https://api.lingyiwanwu.com/v1/chat/completions",
        temperature=0,
        retry=10,
    ),

    # Claude
    "Claude3-5V_Sonnet": partial(
        Claude3V,
        model="claude-3-5-sonnet-20240620",
        temperature=0,
        retry=10,
        verbose=False,
    ),

    # grok
    "grok-vision-beta": partial(
        GPT4V,
        model="grok-vision-beta",
        api_base="https://api.x.ai/v1/chat/completions",
        temperature=0,
        retry=10,
    ),
    "grok-2-vision-1212": partial(
        GPT4V,
        model="grok-2-vision",
        api_base="https://api.x.ai/v1/chat/completions",
        temperature=0,
        retry=10,
    ),

    # EndoAgentReflexion 多轮智能决策与自我优化
    "EndoAgentReflexion": partial(
        EndoAgentReflexionWrapper,
        model_name="gpt-4o",
        temperature=0.7,
        top_p=0.95,
        max_tokens=2048,
        retry=5,
        verbose=True,
        max_reflexion_rounds=3,
        prompt_file="EndoAgent/endoagent/docs/system_prompts.txt",
        model_weights_dir="EndoAgent/endoagent/models",
        temp_dir="temp/endoagent_reflexion",
        device="cuda"
    ),

    "EndoAgentReflexion_2": partial(
        EndoAgentReflexionWrapper,
        model_name="gpt-4o",
        temperature=0.7,
        top_p=0.95,
        max_tokens=2048,
        retry=5,
        verbose=True,
        max_reflexion_rounds=2,
        prompt_file="EndoAgent/endoagent/docs/system_prompts.txt",
        model_weights_dir="EndoAgent/endoagent/models",
        temp_dir="temp/endoagent_reflexion",
        device="cuda"
    ),

    # ColonGPT for endoscopy image analysis
    "ColonGPT": partial(
        ColonGPTAPI,
        # model_path='endoagent/tools/Config/IntelliScope/cache/checkpoint/ColonGPT-phi1.5-siglip-lora-stg2',
        model_path='EndoAgent/endoagent/tools/Config/IntelliScope/cache/checkpoint/ColonGPT-phi1.5-siglip-stg1',
        model_base='EndoAgent/endoagent/tools/Config/IntelliScope/cache/downloaded-weights/phi-1.5',
        model_type='phi-1.5',
        device="cuda",
        max_new_tokens=512,
        temperature=0.2
    ),
}

supported_VLM = {}

model_groups = [
    api_models.keys(),
]

for grp in model_groups:
    supported_VLM.update(grp)
