"""Custom PyTorch Dataset for Financial Chart Visual Instruction Tuning."""
import json
from pathlib import Path
from typing import Dict, Any, List
from PIL import Image
from torch.utils.data import Dataset


class FinancialChartDataset(Dataset):
    """
    Dataset nạp ảnh biểu đồ tài chính và chuỗi suy luận (CoT annotations)
    từ các file splits (.jsonl).
    """

    def __init__(self, split_file: str, image_root: str = "", transform=None):
        self.split_file = Path(split_file)
        self.image_root = Path(image_root)
        self.transform = transform
        self.data: List[Dict[str, Any]] = []
        self._load_data()

    def _load_data(self):
        if not self.split_file.exists():
            return
        with open(self.split_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        image_path = self.image_root / item["image_path"]
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return {
            "id": item.get("id", str(idx)),
            "image": image,
            "instruction": item.get("instruction", "Phân tích biểu đồ này."),
            "cot_reasoning": item.get("cot_reasoning", ""),
            "action": item.get("action", ""),
        }
