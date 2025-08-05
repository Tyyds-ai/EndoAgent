import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import json
import time
import random
import logging
import warnings
from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv
import sys
import traceback

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
_ = load_dotenv()

# Import EndoAgentReflexion
from vlmeval.api.endoagent_reflexion import EndoAgentReflexionWrapper

def load_entries(input_file):
    entries = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line)
                entries.append(entry)
            except Exception:
                continue
    return entries

def save_results(results, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

def process_entry(agent, entry):
    # 构造输入
    inputs = []
    if "image_path" in entry:
        inputs.append({"type": "image", "value": entry["image_path"]})
    if "question" in entry:
        inputs.append({"type": "text", "value": entry["question"]})
    # 多轮反思机制
    try:
        ret_code, response, log = agent.generate_inner(inputs)
        entry["answer_endoagent_reflexion"] = response
        entry["reflexion_log"] = log
        entry["ret_code"] = ret_code
    except Exception as e:
        entry["answer_endoagent_reflexion"] = f"Error: {str(e)}"
        entry["reflexion_log"] = traceback.format_exc()
        entry["ret_code"] = 1
    return entry

def main():
    input_file = "data/endoscopy_language_tasks_with_answers.jsonl"
    output_file = "data/endoscopy_language_tasks_with_endoagent_reflexion.jsonl"
    entries = load_entries(input_file)
    print(f"Loaded {len(entries)} entries.")

    # 加载已处理结果，跳过已完成条目
    processed_ids = set()
    results = []
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    result = json.loads(line)
                    qid = result.get("question_id")
                    if qid is not None:
                        processed_ids.add(qid)
                        results.append(result)
                except Exception:
                    continue
        print(f"已加载 {len(processed_ids)} 条已处理结果，将跳过这些条目。")

    # 初始化 EndoAgentReflexion
    agent = EndoAgentReflexionWrapper(
        prompt_file="EndoAgent/endoagent/docs/system_prompts.txt",
        model_weights_dir="EndoAgent/endoagent/models",
        temp_dir="temp_endoagent",
        device="cuda",
        model_name="gpt-4o",
        max_reflexion_rounds=3,
        verbose=True
    )

    # 处理未完成条目，每处理一个就追加保存
    with open(output_file, "a", encoding="utf-8") as f:
        for entry in tqdm(entries, desc="Processing"):
            qid = entry.get("question_id")
            if qid in processed_ids:
                continue
            result = process_entry(agent, entry)
            results.append(result)
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"全部处理完成，结果已保存到 {output_file}")

if __name__ == "__main__":
    main()