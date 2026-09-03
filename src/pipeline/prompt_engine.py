"""Builds structured Chain-of-Thought prompts and Few-shot injections."""
import yaml
from pathlib import Path
from typing import List, Dict, Any


class PromptEngine:
    def __init__(self, config_path: str = "./configs/prompt_templates.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            return {}
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def build_system_prompt(self) -> str:
        return self.config.get("system_prompt", "")

    def build_chat_messages(self, user_instruction: str, few_shot_examples: List[Dict[str, Any]] = None) -> List[Dict[str, str]]:
        messages = [{"role": "system", "content": self.build_system_prompt()}]

        if few_shot_examples:
            for ex in few_shot_examples:
                messages.append({"role": "user", "content": ex["user"]})
                messages.append({"role": "assistant", "content": ex["assistant"]})

        messages.append({"role": "user", "content": user_instruction})
        return messages
