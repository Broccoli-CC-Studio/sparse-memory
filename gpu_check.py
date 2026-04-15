"""GPU health check — run before any GPU task"""
import subprocess
import sys


def check_gpu():
    """Check GPU state and warn about stale processes."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
            capture_output=True, text=True
        )
        lines = [l.strip() for l in result.stdout.strip().split("\n") if l.strip()]

        if not lines:
            print("GPU: clean, no processes")
            return True

        print(f"GPU: {len(lines)} process(es) using VRAM:")
        total_mb = 0
        stale = []
        for line in lines:
            parts = line.split(",")
            pid = parts[0].strip()
            mem = parts[1].strip()
            mb = int(mem.replace("MiB", "").strip())
            total_mb += mb

            # check if process exists
            try:
                with open(f"/proc/{pid}/cmdline", "r") as f:
                    cmd = f.read().replace("\x00", " ")[:80]
                print(f"  PID {pid}: {mem} — {cmd}")
            except FileNotFoundError:
                print(f"  PID {pid}: {mem} — STALE (process dead, CUDA context leaked)")
                stale.append(pid)

        print(f"  Total: {total_mb} MiB")

        if stale:
            print(f"\nWARNING: {len(stale)} stale CUDA context(s) detected!")
            print("These need container restart to clear.")
            print("To prevent: always use 'with MSAEngine(...) as engine:' pattern")
            return False

        return True

    except FileNotFoundError:
        print("nvidia-smi not found")
        return False


if __name__ == "__main__":
    ok = check_gpu()
    sys.exit(0 if ok else 1)
