import os
import json
import time
from typing import List, Dict, Any

import requests

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'ai_training', 'language_model', 'data')
OUTPUT_PATH = os.path.join(DATA_DIR, 'collab.jsonl')

CONFIG = {
    'server_url': os.environ.get('CHAT_SERVER_URL', 'http://localhost:3000'),
    'models': ['gpt-oss-20', 'llama3.2', 'qwen2.5'],
    'system_prompt': 'You are a helpful, precise assistant.',
    'input_file': os.path.join(DATA_DIR, 'train.txt'),
    'batch_sleep_s': 0.2,
}


def read_prompts(path: str) -> List[str]:
    prompts: List[str] = []
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(line)
    return prompts


def call_multi_chat(server_url: str, models: List[str], messages: List[Dict[str, str]]) -> Dict[str, Any]:
    url = f"{server_url}/api/multi-chat"
    resp = requests.post(url, json={'models': models, 'messages': messages}, timeout=60)
    resp.raise_for_status()
    return resp.json()


def choose_best(results: List[Dict[str, Any]]) -> str:
    candidates = [r.get('text', '') for r in results if r.get('ok') and r.get('text')]
    candidates = [c for c in candidates if c.strip()]
    if not candidates:
        return ''
    # Simple heuristic: longest non-empty response
    return sorted(candidates, key=lambda x: len(x), reverse=True)[0]


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    prompts = read_prompts(CONFIG['input_file'])
    total = len(prompts)
    print(f"Loaded {total} prompts from {CONFIG['input_file']}")

    with open(OUTPUT_PATH, 'w', encoding='utf-8') as out:
        for i, prompt in enumerate(prompts, 1):
            messages = [
                {'role': 'system', 'content': CONFIG['system_prompt']},
                {'role': 'user', 'content': prompt},
            ]
            try:
                data = call_multi_chat(CONFIG['server_url'], CONFIG['models'], messages)
                results = data.get('results', [])
                best = data.get('best') or choose_best(results)
                record = {
                    'prompt': prompt,
                    'responses': [{'model': r.get('model'), 'text': r.get('text'), 'provider': r.get('provider'), 'ok': r.get('ok'), 'ms': r.get('ms')} for r in results],
                    'best': best,
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"[{i}/{total}] Error: {e}")
            else:
                print(f"[{i}/{total}] Collected {len(results)} responses; best length={len(best)}")
            time.sleep(CONFIG['batch_sleep_s'])

    print(f"Wrote collab dataset to {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
