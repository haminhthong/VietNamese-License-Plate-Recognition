"""Kiểm thử gom nhóm và chia tập dataset chống rò rỉ dữ liệu (Data Leakage)."""

import pandas as pd

from src.dataset import _merge_groups, find_group_safe_split, normalize_plate_identity


def test_normalize_plate_identity():
    """Kiểm tra chuẩn hóa identity biển số."""
    assert normalize_plate_identity("51F-123.45") == "51F12345"
    assert normalize_plate_identity(" 30A 9999 ") == "30A9999"
    assert normalize_plate_identity("") == ""
    assert normalize_plate_identity(None) == ""


def test_merge_groups_by_identity_and_md5():
    """Kiểm tra gom nhóm các ảnh có cùng plate_identity hoặc cùng MD5 vào cùng một group_id."""
    df = pd.DataFrame(
        {
            "group_id": ["cam1_01", "cam2_05", "cam3_10"],
            "original_split": ["train", "val", "test"],
            "md5": ["hash1", "hash2", "hash3"],
            "plate_identity": ["51F12345", "51F12345", "30A99999"],
        }
    )
    merged = _merge_groups(df)
    # Hai dòng đầu có cùng plate_identity nên phải được gộp về cùng 1 group_id
    assert merged.loc[0, "group_id"] == merged.loc[1, "group_id"]
    # Dòng thứ 3 có identity khác nên có group_id khác
    assert merged.loc[0, "group_id"] != merged.loc[2, "group_id"]


def test_find_group_safe_split_prevents_leakage():
    """Kiểm tra GroupShuffleSplit không để bất kỳ group_id nào xuất hiện ở nhiều split khác nhau."""
    # Tạo 10 nhóm nguồn khác nhau, mỗi nhóm có 3 ảnh
    records = []
    for g in range(10):
        group_name = f"group_{g}"
        for i in range(3):
            records.append(
                {
                    "image_id": f"{group_name}_{i}",
                    "group_id": group_name,
                    "plate_identity": f"PLATE_{g}",
                }
            )
    manifest = pd.DataFrame(records)
    split_df = find_group_safe_split(manifest, seed=42)

    # Đảm bảo không có group nào giao giữa các split
    crossing = split_df.groupby("group_id")["split"].nunique()
    assert (crossing > 1).sum() == 0

    # Đảm bảo có đủ 3 split
    splits_present = set(split_df["split"].unique())
    assert {"train", "val", "test"}.issubset(splits_present)
