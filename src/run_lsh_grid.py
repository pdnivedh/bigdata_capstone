"""Regenerate every LSH configuration reported in the README so the
Recall@100 table can be reproduced from the repository alone.

Run from the repository root:  python src/run_lsh_grid.py
"""
import subprocess
import sys

GRID = [(3, 2), (3, 4), (3, 8), (5, 4)]   # (top-K genres, hash tables)

for k, t in GRID:
    print(f"\n=== top-{k} genres, {t} hash tables ===")
    subprocess.run([sys.executable, "src/build_features.py", "--top-genres", str(k)], check=True)
    subprocess.run([sys.executable, "src/lsh_candidates.py", "--top-genres", str(k), "--tables", str(t)], check=True)

# leave the final configuration's features in place
print("\nDone. Now run: python src/evaluate_lsh.py")
