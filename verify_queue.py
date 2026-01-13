
import asyncio
import sys
import os
import random

# Add app to path
sys.path.append(os.getcwd())

from app.services.job_manager import JobManager
from app.models.job import JobInfo, JobStatus
from app.config import Settings
from app.services.model_manager import ModelMode

async def blocker_task():
    print("  [BLOCKER] Started. Holding the line for 3 seconds...")
    await asyncio.sleep(3.0)
    print("  [BLOCKER] Finished.")
    return "blocked"

async def tiny_task(idx):
    return f"done_{idx}"

async def test_intense_queue():
    print("=== STARTING INTENSE QUEUE TEST ===")
    settings = Settings()
    manager = JobManager(settings)
    manager.start()

    # 1. Start a blocker so the worker is busy
    print("\n1. Submitting BLOCKER job...")
    await manager.submit_job("BLOCKER", ModelMode.IMAGE, blocker_task)
    
    # Allow worker to pick it up
    await asyncio.sleep(0.1)
    
    # Verify blocker is processing
    if manager.get_queue_position("BLOCKER") is None:
        print("  -> BLOCKER is successfully processing (not in queue).")
    else:
        print("  -> FAILURE: BLOCKER is still in queue!")
        return

    # 2. Submit 50 jobs properly
    COUNT = 50
    print(f"\n2. Rapidly submitting {COUNT} jobs...")
    
    start_time = asyncio.get_running_loop().time()
    for i in range(1, COUNT + 1):
        job_id = f"job_{i}"
        await manager.submit_job(job_id, ModelMode.IMAGE, tiny_task, idx=i)
    
    duration = asyncio.get_running_loop().time() - start_time
    print(f"  -> Submitted {COUNT} jobs in {duration:.4f}s")

    # 3. Verify ALL positions
    print("\n3. Verifying queue positions for ALL 50 jobs...")
    errors = 0
    for i in range(1, COUNT + 1):
        job_id = f"job_{i}"
        pos = manager.get_queue_position(job_id)
        
        # Expected position is 'i' because they were submitted in order 1..50
        # and job_1 is at the front of the queue (pos 1) waiting for BLOCKER
        if pos != i:
            print(f"  [ERROR] {job_id} expected pos {i}, got {pos}")
            errors += 1
            
    if errors == 0:
        print(f"  -> SUCCESS: All {COUNT} jobs have correct sequential positions (1-{COUNT}).")
    else:
        print(f"  -> FAILURE: Found {errors} position errors.")

    # 4. Wait for blocker to finish and see shifts
    print("\n4. Waiting for BLOCKER to finish and worker to pick up job_1...")
    # Blocker sleeps 3s total, we already waited ~0.1s + submission time.
    # Let's wait another 3.5s to be safe.
    await asyncio.sleep(3.5)
    
    # Now:
    # BLOCKER -> Completed
    # job_1 -> Processing (None)
    # job_2 -> Pos 1
    # job_50 -> Pos 49
    
    print("  -> Checking shifting logic...")
    
    pos_job_1 = manager.get_queue_position("job_1")
    if pos_job_1 is None:
        # It might be processing or completed (tiny task is fast)
        # If it's completed, job_2 might be processing or completed...
        # Tiny tasks happen instantly. The worker might have chewed through many of them 
        # in the time we slept if we are not careful.
        # But wait, tiny_task is NOT async sleep, it just returns. 
        # But the worker loop has overhead.
        
        # Let's check status directly
        j1_status = manager.get_job("job_1").status
        print(f"  -> job_1 status: {j1_status} (Queue pos: {pos_job_1})")
    else:
        print(f"  -> job_1 is unexpectedly still in queue at pos {pos_job_1}")

    # Let's check the tail
    job_last = f"job_{COUNT}"
    pos_last = manager.get_queue_position(job_last)
    
    # If the worker is fast, it might have finished everything.
    # If the worker overhead is non-zero, some might still be there.
    remaining = len(manager._pending_jobs)
    print(f"  -> Remaining jobs in queue: {remaining}")
    
    if remaining == 0:
        print("  -> Worker successfully processed all jobs!")
    else:
        print(f"  -> Worker is still chugging. Last job ({job_last}) pos: {pos_last}")
        # Validate that the last job's position equals the queue length
        if pos_last == remaining:
             print("  -> FIFO integrity maintained (Last job pos == Queue length).")
        else:
             print(f"  -> FAILURE: Queue integrity mismatch. Last pos {pos_last} != {remaining}")

    manager.stop()
    print("=== TEST COMPLETED ===")

if __name__ == "__main__":
    asyncio.run(test_intense_queue())
