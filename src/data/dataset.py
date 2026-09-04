"""Custom PyTorch Dataset for Financial Chart Visual Instruction Tuning."""
import json
from pathlib import Path
from typing import Dict, Any, List
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    Image = None

try:
    from torch.utils.data import Dataset
except ImportError:
    class Dataset:
        pass



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

    def get_conversation(self, idx: int) -> List[Dict[str, Any]]:
        """
        Định dạng bản ghi thành cấu trúc hội thoại đa phương thức chuẩn của Qwen2.5-VL:
        User: [Image + Instruction] -> Assistant: [CoT Reasoning + Action].
        """
        item = self.data[idx]
        image_path = self.image_root / item["image_path"] if self.image_root else Path(item["image_path"])

        # Load image if exists, else keep path
        if image_path.exists():
            image = Image.open(image_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
        else:
            image = None

        instruction = item.get("instruction", "Hãy phân tích biểu đồ kỹ thuật này theo quy chuẩn CMT 4 bước và đề xuất chiến lược xử lý.")
        cot = item.get("cot_reasoning", "").strip()
        action = item.get("action", "").strip()

        # Ghép phản hồi của assistant
        assistant_content = cot
        if action and action not in cot:
            assistant_content += f"\n\nKhuyến nghị cuối cùng: {action}"

        user_content = []
        if image is not None:
            user_content.append({"type": "image", "image": image})
        elif image_path.exists():
            user_content.append({"type": "image", "image": str(image_path)})
        user_content.append({"type": "text", "text": instruction})

        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
        return messages

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        image_path = self.image_root / item["image_path"] if self.image_root else Path(item["image_path"])
        image = None
        if image_path.exists():
            image = Image.open(image_path).convert("RGB")
            if self.transform:
                image = self.transform(image)

        messages = self.get_conversation(idx)

        return {
            "id": item.get("id", str(idx)),
            "image": image,
            "image_path": str(image_path),
            "instruction": item.get("instruction", "Phân tích biểu đồ này."),
            "cot_reasoning": item.get("cot_reasoning", ""),
            "action": item.get("action", ""),
            "messages": messages,
        }

