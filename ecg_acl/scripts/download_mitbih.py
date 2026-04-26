from __future__ import annotations

import argparse
from pathlib import Path


MITBIH_RECORDS = [
    "100", "101", "102", "103", "104", "105", "106", "107", "108", "109",
    "111", "112", "113", "114", "115", "116", "117", "118", "119", "121",
    "122", "123", "124", "200", "201", "202", "203", "205", "207", "208",
    "209", "210", "212", "213", "214", "215", "217", "219", "220", "221",
    "222", "223", "228", "230", "231", "232", "233", "234",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download MIT-BIH Arrhythmia records from PhysioNet.")
    parser.add_argument("--out", default="ecg_acl/data/mitbih/raw", help="Output directory.")
    parser.add_argument("--records", nargs="*", default=MITBIH_RECORDS, help="Record ids to download.")
    return parser.parse_args()


def main() -> None:
    try:
        import wfdb
    except ImportError as exc:
        raise RuntimeError("Install wfdb first: pip install wfdb") from exc

    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for record in args.records:
        print(f"downloading {record}")
        wfdb.dl_database("mitdb", dl_dir=str(out), records=[record])
    print(f"done: {out}")


if __name__ == "__main__":
    main()
