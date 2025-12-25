import os
import json
import sys
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'ai_training', 'language_model', 'data')
COLLAB_PATH = os.path.join(DATA_DIR, 'collab.jsonl')


def analyze_dataset():
    if not os.path.exists(COLLAB_PATH):
        print("No dataset found.")
        return
    
    records = []
    with open(COLLAB_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    
    if not records:
        print("Dataset is empty.")
        return
    
    print(f"\n{'='*60}")
    print(f"DATASET ANALYSIS: {len(records)} records")
    print(f"{'='*60}\n")
    
    # Aggregator distribution
    agg_counts = defaultdict(int)
    for rec in records:
        agg = rec.get('aggregator', 'unknown')
        agg_counts[agg] += 1
    
    print("📊 Aggregator Distribution:")
    for agg, count in sorted(agg_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / len(records)
        print(f"  {agg:20s} {count:4d} ({pct:5.1f}%)")
    
    # Model performance
    model_stats = defaultdict(lambda: {'count': 0, 'total_ms': 0, 'errors': 0})
    for rec in records:
        for result in rec.get('results', []):
            model = result.get('model', 'unknown')
            model_stats[model]['count'] += 1
            model_stats[model]['total_ms'] += result.get('ms', 0)
            if not result.get('ok'):
                model_stats[model]['errors'] += 1
    
    print("\n⚡ Model Performance:")
    print(f"  {'Model':20s} {'Queries':>8s} {'Avg Time':>10s} {'Errors':>8s}")
    print(f"  {'-'*20} {'-'*8} {'-'*10} {'-'*8}")
    for model, stats in sorted(model_stats.items(), key=lambda x: -x[1]['count']):
        avg_ms = stats['total_ms'] / stats['count'] if stats['count'] > 0 else 0
        print(f"  {model:20s} {stats['count']:8d} {avg_ms:8.0f}ms {stats['errors']:8d}")
    
    # Prompt/response length stats
    prompt_lens = [len(r.get('prompt', '')) for r in records]
    best_lens = [len(r.get('best', '')) for r in records]
    
    print("\n📏 Length Statistics:")
    print(f"  Prompt avg: {sum(prompt_lens)/len(prompt_lens):.0f} chars")
    print(f"  Prompt range: {min(prompt_lens)}-{max(prompt_lens)} chars")
    print(f"  Best response avg: {sum(best_lens)/len(best_lens):.0f} chars")
    print(f"  Best response range: {min(best_lens)}-{max(best_lens)} chars")
    
    # Token estimation
    total_chars = sum(prompt_lens) + sum(best_lens)
    est_tokens = total_chars // 4  # Rough estimate
    print(f"\n🔢 Estimated tokens: ~{est_tokens:,}")
    print(f"   (Training context: ~{est_tokens} tokens)")
    
    # Sample prompts
    print("\n📝 Sample Prompts:")
    for i, rec in enumerate(records[:3], 1):
        prompt = rec.get('prompt', '')[:100]
        best_preview = rec.get('best', '')[:80]
        print(f"  {i}. {prompt}...")
        print(f"     → {best_preview}...\n")
    
    print(f"{'='*60}\n")


if __name__ == '__main__':
    analyze_dataset()
