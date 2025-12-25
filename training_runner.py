#!/usr/bin/env python
"""
Training runner for local AI modules.

Usage:
  python training_runner.py --module language_model --config-b64 <base64-json>
"""

import argparse
import base64
import json
import sys
from copy import deepcopy


def decode_config(config_b64):
    if not config_b64:
        return {}
    try:
        raw = base64.b64decode(config_b64.encode("utf-8")).decode("utf-8")
        return json.loads(raw)
    except Exception as exc:
        print(f"Failed to decode config overrides: {exc}")
        return {}


def merge_config(base, overrides):
    merged = deepcopy(base)
    for key, value in overrides.items():
        merged[key] = value
    return merged


def normalize_custom_nn_overrides(overrides):
    if "hidden_layers" in overrides and isinstance(overrides["hidden_layers"], str):
        parts = [p.strip() for p in overrides["hidden_layers"].split(",") if p.strip()]
        try:
            overrides["hidden_layers"] = [int(p) for p in parts]
        except ValueError:
            pass
    return overrides


def run_language_model(overrides):
    import language_model as mod
    config = merge_config(mod.CONFIG, overrides)
    trainer = mod.LanguageModelTrainer(config)
    trainer.train()


def run_image_classifier(overrides):
    import image_classifier as mod
    config = merge_config(mod.CONFIG, overrides)
    trainer = mod.ImageClassifier(config)
    trainer.train()


def run_game_ai(overrides):
    import game_ai as mod
    config = merge_config(mod.CONFIG, overrides)
    trainer = mod.GameAITrainer(config)
    trainer.train()


def run_custom_nn(overrides):
    import custom_nn as mod
    overrides = normalize_custom_nn_overrides(dict(overrides))
    preset_name = overrides.pop("preset", None)
    base_config = mod.CONFIG
    if preset_name:
        preset = mod.PRESETS.get(preset_name)
        if preset:
            base_config = merge_config(base_config, preset)
        else:
            print(f"Unknown preset '{preset_name}', using default config.")
    config = merge_config(base_config, overrides)
    trainer = mod.CustomNNTrainer(config)
    trainer.summary()
    trainer.train()


def run_world_generator(overrides):
    import world_generator_train as mod
    mod.CONFIG.update(overrides)
    mod.train_world_generator()


def main():
    parser = argparse.ArgumentParser(description="Run local training modules.")
    parser.add_argument("--module", required=True, help="Training module name.")
    parser.add_argument("--config-b64", default="", help="Base64 JSON config overrides.")
    args = parser.parse_args()

    module = args.module.strip()
    overrides = decode_config(args.config_b64)

    runners = {
        "language_model": run_language_model,
        "image_classifier": run_image_classifier,
        "game_ai": run_game_ai,
        "custom_nn": run_custom_nn,
        "world_generator": run_world_generator,
    }

    runner = runners.get(module)
    if not runner:
        print(f"Unknown module '{module}'.")
        sys.exit(1)

    runner(overrides)


if __name__ == "__main__":
    main()
