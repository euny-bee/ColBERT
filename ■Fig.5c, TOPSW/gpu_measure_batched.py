"""GPU 배치(batch) 처리 실측 -- 쿼리 B개를 한 번에 묶어서 처리, 쿼리당 에너지/latency 재측정.
실제 서비스에서는 GPU에 쿼리를 1개씩 안 던지고 배치로 묶는 게 표준이라, batch=1(비교 불공정) 대비 공정한 비교용.
"""
import torch, time, threading
import numpy as np
import pandas as pd
import pynvml

BASE = r'C:\Users\nmdl-khb\ColBERT\centroidset_vector 크기 조절\06_newq_margin'

def load_qc():
    Q_df = pd.read_excel(f'{BASE}/[clip99.9]query_embs_96x128.xlsx', index_col=0)
    C_df = pd.read_excel(f'{BASE}/[clip99.9]centroids_100x128.xlsx', index_col=0)
    Q = Q_df.values.astype(np.float32)[:32]   # 쿼리 1개 = 32 토큰
    C = C_df.values.astype(np.float32)
    return Q, C

def measure_batched(gpu_index, n_cand, batch_size, duration_s=4.0, warmup=100):
    dev = torch.device(f'cuda:{gpu_index}')
    torch.cuda.set_device(dev)

    Q_np, C_np = load_qc()
    # 쿼리 B개를 배치 차원으로 쌓음 (같은 32토큰 쿼리를 복제 -- 실제 서비스에서는 서로 다른 쿼리들이지만, 연산량/텐서 크기는 동일)
    Q_batch_np = np.tile(Q_np[None, :, :], (batch_size, 1, 1)).reshape(batch_size*32, 128)
    Q = torch.tensor(Q_batch_np, device=dev)          # (B*32,128)
    C = torch.tensor(C_np, device=dev)                # (100,128)
    rng = np.random.default_rng(0)
    D_np = rng.normal(0, Q_np.std(), size=(n_cand,128)).astype(np.float32)
    D = torch.tensor(D_np, device=dev)

    def workload():
        coarse = Q @ C.T                 # (B*32,100)
        _ = coarse.topk(2, dim=1)
        score = Q @ D.T                  # (B*32,n_cand)
        maxsim = score.view(batch_size, 32, -1).max(dim=2).values.sum()
        return maxsim

    for _ in range(warmup):
        workload()
    torch.cuda.synchronize(dev)

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
    power_samples = []
    stop_flag = threading.Event()
    def poll_power():
        while not stop_flag.is_set():
            power_samples.append(pynvml.nvmlDeviceGetPowerUsage(handle)/1000.0)
            time.sleep(0.02)
    t = threading.Thread(target=poll_power)
    t.start()

    torch.cuda.synchronize(dev)
    t0 = time.perf_counter()
    n_batches = 0
    STEP = 20
    while time.perf_counter() - t0 < duration_s:
        for _ in range(STEP):
            workload()
        torch.cuda.synchronize(dev)
        n_batches += STEP
    t1 = time.perf_counter()

    stop_flag.set()
    t.join()
    pynvml.nvmlShutdown()

    total_time = t1 - t0
    n_queries = n_batches * batch_size
    latency_per_query = total_time / n_queries
    latency_per_batch = total_time / n_batches
    avg_power = np.mean(power_samples)
    energy_per_query = avg_power * latency_per_query

    return dict(gpu=torch.cuda.get_device_name(gpu_index), n_cand=n_cand, batch_size=batch_size,
                n_batches=n_batches, n_queries=n_queries, total_time=total_time,
                latency_per_batch_us=latency_per_batch*1e6, latency_per_query_us=latency_per_query*1e6,
                avg_power_W=avg_power, energy_per_query_nJ=energy_per_query*1e9,
                n_power_samples=len(power_samples))

if __name__ == '__main__':
    results = []
    for gpu_idx in [0, 1]:
        for batch_size in [32, 256]:
            for label, n_cand in [('OptionA_cand184', 184), ('OptionC_cand193', 193)]:
                r = measure_batched(gpu_idx, n_cand, batch_size)
                r['label'] = label
                results.append(r)
                print(r)
    df = pd.DataFrame(results)
    df.to_csv('gpu_measure_batched_results.csv', index=False)
    print("\nSaved gpu_measure_batched_results.csv")
