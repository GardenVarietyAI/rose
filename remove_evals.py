#!/usr/bin/env python3
"""Remove all eval-related files."""

import os
import shutil

# Remove the entire evals directory from rose_worker
evals_dir = "src/rose_worker/evals"
if os.path.exists(evals_dir):
    shutil.rmtree(evals_dir)
    print(f"✓ Removed {evals_dir}")
else:
    print(f"✗ {evals_dir} not found")

print("\nEval worker code removed. Next steps:")
print("1. Remove eval API endpoints")
print("2. Check dependencies before removing stores/DB tables")
