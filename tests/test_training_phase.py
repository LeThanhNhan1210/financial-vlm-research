"""
Unit & Integration Tests for Phase 2 Training Components (Dataset, Collator, DriveSyncCallback).
"""
import sys
import json
import shutil
import tempfile
import unittest
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import FinancialChartDataset
from src.training.callbacks import DriveSyncCallback
from src.training.collator import QwenVLDataCollator


class MockTokenizer:
    def __init__(self):
        self.pad_token_id = 0
        self.vocab = {"<|im_start|>": 1, "assistant": 2, "user": 3, "<|im_end|>": 4}

    def encode(self, text, add_special_tokens=False):
        if text == "<|im_start|>assistant":
            return [1, 2]
        return [5, 6, 7]


class MockProcessor:
    def __init__(self):
        self.tokenizer = MockTokenizer()

    def apply_chat_template(self, msg, tokenize=False, add_generation_prompt=False):
        return f"<|im_start|>user\n{msg[0]['content']}<|im_end|>\n<|im_start|>assistant\n{msg[1]['content']}<|im_end|>"

    def __call__(self, text=None, images=None, videos=None, padding=True, return_tensors="pt"):
        # Giả lập tensors đầu ra nếu không có PyTorch thật
        class MockTensor:
            def __init__(self, data):
                self.data = [list(row) for row in data]

            def clone(self):
                return MockTensor([list(r) for r in self.data])

            def __getitem__(self, idx):
                return self.data[idx]

            def __setitem__(self, key, val):
                if isinstance(key, tuple) and isinstance(key[1], slice):
                    row_idx, s = key
                    start = s.start or 0
                    stop = s.stop if s.stop is not None else len(self.data[row_idx])
                    for k in range(start, stop):
                        self.data[row_idx][k] = val
                elif isinstance(key, tuple):
                    self.data[key[0]][key[1]] = val
                elif isinstance(key, slice):
                    pass
                # Bỏ qua nếu key là boolean mask giả lập
                elif isinstance(key, (bool, MockTensor)):
                    pass
                else:
                    self.data[key] = val

            def __eq__(self, other):
                return False

            def __len__(self):
                return len(self.data)


        # Token sequence: [prompt tokens...] + [1, 2, 198] + [cot tokens...]
        mock_seq = [10, 11, 12, 1, 2, 198, 20, 21, 22]
        return {
            "input_ids": MockTensor([mock_seq]),
            "attention_mask": MockTensor([[1] * len(mock_seq)]),
        }


class TestTrainingPhase(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.test_jsonl = Path(self.test_dir) / "test_train.jsonl"
        sample_records = [
            {
                "id": "train_001",
                "image_path": "dummy.png",
                "symbol": "VCB",
                "instruction": "Phân tích biểu đồ kỹ thuật này.",
                "cot_reasoning": "1. Xu hướng: Uptrend.\n2. Khối lượng: Cao.\n3. Hỗ trợ: 100.\n4. Hành động: BUY",
                "action": "BUY",
            }
        ]
        with open(self.test_jsonl, "w", encoding="utf-8") as f:
            for r in sample_records:
                f.write(json.dumps(r) + "\n")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_dataset_get_conversation(self):
        """Kiểm tra format tin nhắn đa phương thức của FinancialChartDataset."""
        ds = FinancialChartDataset(self.test_jsonl)
        self.assertEqual(len(ds), 1)

        conv = ds.get_conversation(0)
        self.assertEqual(len(conv), 2)
        self.assertEqual(conv[0]["role"], "user")
        self.assertEqual(conv[1]["role"], "assistant")
        self.assertIn("BUY", conv[1]["content"])


    def test_drive_sync_callback(self):
        """Kiểm tra callback tự động sao lưu checkpoint sang Google Drive folder."""
        drive_dir = Path(self.test_dir) / "drive_checkpoints"
        output_dir = Path(self.test_dir) / "local_output"
        checkpoint_dir = output_dir / "checkpoint-50"
        checkpoint_dir.mkdir(parents=True)

        # Tạo file giả lập adapter weights
        (checkpoint_dir / "adapter_model.safetensors").write_text("fake_weights")
        (checkpoint_dir / "adapter_config.json").write_text("fake_config")

        class MockState:
            global_step = 50

        class MockArgs:
            pass

        mock_args = MockArgs()
        mock_args.output_dir = str(output_dir)

        callback = DriveSyncCallback(drive_backup_dir=str(drive_dir))
        callback.on_save(mock_args, MockState(), None)

        # Kiểm tra file đã được copy sang drive_dir / checkpoint-50
        backup_cp = drive_dir / "checkpoint-50"
        self.assertTrue(backup_cp.exists())
        self.assertTrue((backup_cp / "adapter_model.safetensors").exists())
        self.assertEqual(len(callback.synced_checkpoints), 1)

    def test_loss_masking_collator(self):
        """Kiểm tra cơ chế Loss Masking gán nhãn -100 cho prompt tokens."""
        processor = MockProcessor()
        collator = QwenVLDataCollator(processor)

        sample_item = {
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "Prompt"}]},
                {"role": "assistant", "content": "Assistant answer"},
            ]
        }
        batch = collator([sample_item])
        labels = batch["labels"]

        # Vị trí trước assistant [0, 1, 2, 3, 4, 5] phải là -100
        for pos in range(6):
            self.assertEqual(labels[0][pos], -100)

        # Vị trí assistant [6, 7, 8] phải giữ nguyên giá trị token (20, 21, 22)
        self.assertEqual(labels[0][6], 20)
        self.assertEqual(labels[0][7], 21)
        self.assertEqual(labels[0][8], 22)


if __name__ == "__main__":
    unittest.main()
