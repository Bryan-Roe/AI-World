import sys
import json
from typing import List, Dict

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def semantic_rank(prompt: str, responses: List[Dict]) -> str:
    """
    Rank responses by semantic similarity to prompt.
    Returns the text of the best response.
    """
    if not HAS_TRANSFORMERS:
        # Fallback: return longest
        valid = [r for r in responses if r.get('ok') and r.get('text')]
        if not valid:
            return ''
        return max(valid, key=lambda x: len(x.get('text', '')))['text']

    # Load mini model for speed
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    valid = [r for r in responses if r.get('ok') and r.get('text') and r['text'].strip()]
    if not valid:
        return ''
    
    prompt_emb = model.encode(prompt, convert_to_numpy=True)
    best_text = ''
    best_score = -1.0
    
    for r in valid:
        text = r['text']
        text_emb = model.encode(text, convert_to_numpy=True)
        score = cosine_similarity(prompt_emb, text_emb)
        if score > best_score:
            best_score = score
            best_text = text
    
    return best_text


def main():
    if len(sys.argv) < 2:
        print(json.dumps({'error': 'Usage: semantic_rank.py <json_payload>'}))
        sys.exit(1)
    
    try:
        payload = json.loads(sys.argv[1])
        prompt = payload.get('prompt', '')
        responses = payload.get('responses', [])
        
        best = semantic_rank(prompt, responses)
        print(json.dumps({'best': best}))
    except Exception as e:
        print(json.dumps({'error': str(e)}))
        sys.exit(1)


if __name__ == '__main__':
    main()
