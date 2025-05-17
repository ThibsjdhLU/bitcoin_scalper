import os
import subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))

errors = []
for folder, _, files in os.walk(ROOT):
    for file in files:
        if file.endswith(".py"):
            path = os.path.join(folder, file)
            print(f"Checking {path} ...")
            result = subprocess.run(["python", "-m", "py_compile", path], capture_output=True, text=True)
            if result.returncode != 0:
                errors.append((path, result.stderr.strip()))

if errors:
    print("\n=== Import/Compilation errors found ===")
    for path, err in errors:
        print(f"\nFile: {path}\n{err}")
else:
    print("All Python files compiled successfully (no import errors)!")