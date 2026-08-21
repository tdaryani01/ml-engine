# utils/thread_monitor.py
import threading
import time
import os
import psutil

class ThreadActivityMonitor:
    def __init__(self, interval_ms: float = 50.0):
        self.interval = interval_ms / 1000.0
        self.running = False
        self._thread = None
        self.proc = psutil.Process(os.getpid())
        self.samples = []

    def _sample_loop(self):
        # Prime psutil CPU times
        self.proc.cpu_percent(interval=None)
        
        while self.running:
            try:
                threads = self.proc.threads()
                cpu_user = self.proc.cpu_times().user
                cpu_system = self.proc.cpu_times().system
                active_native_threads = len(threads)
                
                # Check user vs kernel time distribution across native threads
                user_heavy = sum(1 for t in threads if t.user_time > 0)
                
                self.samples.append({
                    "total_threads": active_native_threads,
                    "user_threads": user_heavy,
                    "user_time": cpu_user,
                    "sys_time": cpu_system
                })
            except Exception:
                pass
            time.sleep(self.interval)

    def start(self):
        self.running = True
        self.samples.clear()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop_and_report(self):
        self.running = False
        if self._thread:
            self._thread.join(timeout=1.0)
            
        if not self.samples:
            print("[Thread Monitor] No telemetry samples recorded.")
            return

        total_samples = len(self.samples)
        avg_threads = sum(s["total_threads"] for s in self.samples) / total_samples
        max_threads = max(s["total_threads"] for s in self.samples)
        
        total_user_time = self.samples[-1]["user_time"] - self.samples[0]["user_time"]
        total_sys_time = self.samples[-1]["sys_time"] - self.samples[0]["sys_time"]
        total_time = max(1e-5, total_user_time + total_sys_time)
        
        sys_pct = (total_sys_time / total_time) * 100.0
        user_pct = (total_user_time / total_time) * 100.0

        print("\n" + "=" * 70)
        print("          LIVE PROCESS NATIVE THREADING TELEMETRY")
        print("=" * 70)
        print(f"Total Samples Collected       : {total_samples} (every {self.interval*1000:.0f} ms)")
        print(f"Native OS Threads in Process  : Avg {avg_threads:.1f} | Peak {max_threads}")
        print(f"CPU Time in User Mode (Compute): {user_pct:.2f}%")
        print(f"CPU Time in Kernel/Sys Mode   : {sys_pct:.2f}%")
        print("-" * 70)
        if sys_pct > 15.0:
            print("DIAGNOSIS: High Kernel/System CPU time detected (>15%).")
            print("-> Threads are spending excessive time in OS mutex locks/context switches.")
        elif user_pct > 85.0 and avg_threads > 8:
            print("DIAGNOSIS: Heavy User-Space contention.")
            print("-> Worker threads (vcomp140.dll / OpenBLAS) are active-spinning on barrier loops.")
        print("=" * 70 + "\n")