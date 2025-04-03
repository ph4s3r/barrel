import requests
import time
import threading

url = "http://127.0.0.1:3001/embed"
data = {"inputs": "What is Deep Learning?"}
headers = {"Content-Type": "application/json"}

# Function to perform a single request and record its result
def make_request(thread_id, results):
    req_start = time.time()
    try:
        response = requests.post(url, json=data, headers=headers)
        status = response.status_code
        vector_dim = len(response.json()[0])
    except Exception as e:
        status = "Error"
        vector_dim = None
        print(f"Thread {thread_id}: Exception occurred - {e}")
    req_end = time.time()
    results[thread_id] = {
        "status": status,
        "vector_dim": vector_dim,
        "time": req_end - req_start
    }
    print(f"Thread {thread_id}: Status={status}, Vector dim={vector_dim}, Time={req_end - req_start:.4f}s")

# Single request timing
start = time.time()
response = requests.post(url, json=data, headers=headers)
end = time.time()

print("Response Status Code:", response.status_code)
print("Response Vector dim:", len(response.json()[0]))
print(f"Single request time: {end - start:.4f} seconds")

# Parallel load testing using threading
num_requests = 10
threads = []
results = {}

print("\nStarting parallel load test with threading...")
load_start = time.time()

for i in range(num_requests):
    thread = threading.Thread(target=make_request, args=(i+1, results))
    threads.append(thread)
    thread.start()
    # To maintain the desired request rate
    time.sleep(1.0 / 2)  # requests_per_second = 2

# Wait for all threads to complete
for thread in threads:
    thread.join()

load_end = time.time()

# Calculate throughput
total_load_time = load_end - load_start
throughput = num_requests / total_load_time

print(f"\nTotal time for {num_requests} parallel requests: {total_load_time:.4f}s")
print(f"Throughput: {throughput:.2f} requests/second")