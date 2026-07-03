import subprocess
import time
import sys
from datetime import datetime, timezone, timedelta

IST = timezone(timedelta(hours=5, minutes=30))

def log(msg):
    t = datetime.now(IST).strftime("%I:%M:%S %p")
    print(f"[{t} IST] {msg}")

def run_script(cmd):
    log(f"Running: {cmd}")
    start = time.time()
    result = subprocess.run(cmd, shell=True)
    end = time.time()
    if result.returncode != 0:
        log(f"Command failed with code {result.returncode}")
        sys.exit(1)
    log(f"Finished in {int(end - start)} seconds.")

if __name__ == '__main__':
    log("Starting 12-month chained run...")
    run_script("python backend/scripts/run_12m_chained.py")
    
    log("Starting 6-month automated run...")
    run_script("python backend/scripts/run_6m_chained.py")
    
    log("Starting 3-month automated run...")
    run_script(r"python C:\Users\Admin\.gemini\antigravity-ide\brain\90b11e21-1568-4262-9476-5a8bc3d8ce4c\scratch\automated_run_3m.py")
    
    log("All runs completed successfully!")
