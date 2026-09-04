import csv
import random
import logging
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any, Union

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

logger = logging.getLogger(__name__)


class AuditSampler:
    """
    Module lấy mẫu phân tầng 30% bộ dữ liệu đã gán nhãn CoT để chuyên gia rà soát chất lượng.
    Phân tầng theo nhóm tài sản (asset_class) bằng thư viện chuẩn Python (stdlib),
    hỗ trợ tương thích cả pandas khi có sẵn trên môi trường Colab.
    """

    def __init__(self, sample_rate: float = 0.30, random_state: int = 42):
        self.sample_rate = sample_rate
        self.random_state = random_state

    def sample_records(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Lấy mẫu phân tầng từ danh sách các bản ghi dictionary (chuẩn stdlib).
        """
        if not records:
            return []

        rng = random.Random(self.random_state)

        # Gom nhóm theo asset_class
        grouped = defaultdict(list)
        for r in records:
            asset = r.get("asset_class", "default")
            grouped[asset].append(r)

        sampled_records = []
        for asset, items in grouped.items():
            k = max(1, int(round(len(items) * self.sample_rate)))
            k = min(k, len(items))
            # Shuffled deterministic sample
            chosen = rng.sample(items, k)
            for item in chosen:
                item_copy = dict(item)
                item_copy["audit_status"] = "pending"
                item_copy["expert_action"] = ""
                item_copy["expert_soundness_score"] = ""
                item_copy["expert_notes"] = ""
                sampled_records.append(item_copy)

        logger.info(
            f"Stratified audit sampling: selected {len(sampled_records)}/{len(records)} samples ({len(sampled_records)/len(records)*100:.1f}%)"
        )
        return sampled_records

    def export_audit_csv(
        self,
        data: Union[List[Dict[str, Any]], Any],
        output_path: Union[str, Path],
    ) -> str:
        """
        Lấy mẫu và xuất file CSV phục vụ kiểm định chuyên gia dùng chuẩn csv stdlib.
        """
        if HAS_PANDAS and isinstance(data, pd.DataFrame):
            records = data.to_dict(orient="records")
        else:
            records = list(data)

        sampled = self.sample_records(records)
        if not sampled:
            return ""

        cols_order = [
            "id",
            "symbol",
            "asset_class",
            "timeframe",
            "image_path",
            "action",
            "cot_reasoning",
            "audit_status",
            "expert_action",
            "expert_soundness_score",
            "expert_notes",
        ]

        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        # Lấy tất cả key hiện có
        all_keys = list(sampled[0].keys())
        fieldnames = [c for c in cols_order if c in all_keys] + [
            c for c in all_keys if c not in cols_order
        ]

        with open(out_p, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in sampled:
                writer.writerow(row)

        logger.info(f"Exported HITL audit file to {out_p}")
        return str(out_p)


