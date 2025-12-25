import os
import json
import subprocess
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'ai_training', 'language_model', 'data')
COLLAB_PATH = os.path.join(DATA_DIR, 'collab.jsonl')
MODEL_DIR = os.path.join(BASE_DIR, 'ai_training', 'language_model', 'models', 'distill_student')

CONFIG = {
    'auto_train_threshold': 50,  # Auto-train when dataset reaches this size
    'python_cmd': sys.executable,
    'training_script': os.path.join(BASE_DIR, 'multi_llm_distill.py'),
}


def count_records():
    if not os.path.exists(COLLAB_PATH):
        return 0
    with open(COLLAB_PATH, 'r', encoding='utf-8') as f:
        return sum(1 for line in f if line.strip())


def should_train():
    count = count_records()
    return count >= CONFIG['auto_train_threshold']


def trigger_training():
    print(f"Dataset reached {count_records()} records. Triggering auto-distillation...")
    try:
        result = subprocess.run(
            [CONFIG['python_cmd'], CONFIG['training_script']],
            capture_output=True,
            text=True,
            timeout=3600
        )
        if result.returncode == 0:
            print("✓ Training completed successfully")
            print(result.stdout)
            return True
        else:
            print("✗ Training failed")
            print(result.stderr)
            return False
    except subprocess.TimeoutExpired:
        print("✗ Training timed out (>1 hour)")
        return False
    except Exception as e:
        print(f"✗ Training error: {e}")
        return False


def main():
    print(f"Checking dataset: {COLLAB_PATH}")
    count = count_records()
    print(f"Current records: {count}")
    print(f"Threshold: {CONFIG['auto_train_threshold']}")
    
    if should_train():
        trigger_training()
    else:
        print(f"Not enough records yet. Need {CONFIG['auto_train_threshold'] - count} more.")


if __name__ == '__main__':
    main()
