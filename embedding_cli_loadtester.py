"""
Load testing the 'embed' endpoint.

Measure runtime accurately in multi-treaded program:
https://superfastpython.com/benchmark-time-monotonic/
"""
import time
import threading

import requests


URL = "http://127.0.0.1:3001/embed"
data = {"inputs": "What is Deep Learning?"}
headers = {"Content-Type": "application/json"}


def make_request(thread_id, results):
    """Send 'embed' request and gather results."""
    req_start = time.monotonic()

    try:
        response = requests.post(URL, json=data, headers=headers, timeout=(1, 6))
        status = response.status_code
        vector_dim = len(response.json()[0])
    except Exception as err:
        status = "Error"
        vector_dim = None
        print(f"ERROR occurred on thread {thread_id}: {err}")

    results[thread_id] = {
        "status": status,
        "vector_dim": vector_dim,
        "time": time.monotonic() - req_start
    }
    print(
        f"Thread {thread_id}: Status={status}, Vector dim="
        f"{vector_dim}, Time={results[thread_id]['time']:.4f}s"
    )

# Single request timing
start = time.monotonic()
response = requests.post(URL, json=data, headers=headers, timeout=(1, 6))
end = time.monotonic()

print("Response Status Code:", response.status_code)
print("Response Vector dim:", len(response.json()[0]))
print(f"Single request time: {end - start:.4f} seconds")

# Parallel load testing using threading
NUM_REQUESTS = 10
threads = []
results = {}

print("\nStarting parallel load test with threading...")
load_start = time.monotonic()

for i in range(NUM_REQUESTS):
    thread = threading.Thread(target=make_request, args=(i + 1, results))
    threads.append(thread)
    thread.start()
    # To maintain the desired request rate
    time.sleep(0.5)

# Wait for all threads to complete
for thread in threads:
    thread.join()

# Calculate throughput
total_run_time = time.monotonic() - load_start
throughput = NUM_REQUESTS / total_run_time

print(f"\nTotal time for {NUM_REQUESTS} parallel requests: {total_run_time:.4f}s")
print(f"Throughput: {throughput:.2f} requests/second")
