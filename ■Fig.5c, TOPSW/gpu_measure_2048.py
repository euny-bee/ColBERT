"""GPU 실측 -- 시스템 레벨(2,048 centroid, Table2와 동일 스케일) coarse search + scoring."""
import torch, time, threading
import numpy as np
import pandas as pd
import pynvml

CBASE = r'C:\Users\nmdl-khb\ColBERT\centroidset'

def load_qc():
    Q_all = torch.load(f'{CBASE}/scale_query_embs_255x32x128.pt', weights_only=False)
    Q = Q_all[0].numpy().astype(np.float32)      # 쿼리 1개 = 32 토큰
    C = np.load(f'{CBASE}/scale_centroids_2k.npy').astype(np.float32)   # (2048,128)
    return Q, C

def measure(gpu_index, n_cand, duration_s=4.0, warmup=200):
    dev = torch.device(f'cuda:{gpu_index}')
    torch.cuda.set_device(dev)

    Q_np, C_np = load_qc()
    Q = torch.tensor(Q_np, device=dev)          # (32,128)
    C = torch.tensor(C_np, device=dev)          # (2048,128)
    rng = np.random.default_rng(0)
    D_np = rng.normal(0, Q_np.std(), size=(n_cand,128)).astype(np.float32)
    D = torch.tensor(D_np, device=dev)

    def workload():
        coarse = Q @ C.T          # (32,2048)
        _ = coarse.topk(2, dim=1)
        score = Q @ D.T           # (32,n_cand)
        maxsim = score.max(dim=1).values.sum()
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
    n_iters = 0
    BATCH = 50
    while time.perf_counter() - t0 < duration_s:
        for _ in range(BATCH):
            workload()
        torch.cuda.synchronize(dev)
        n_iters += BATCH
    t1 = time.perf_counter()

    stop_flag.set()
    t.join()
    pynvml.nvmlShutdown()

    total_time = t1 - t0
    latency_per_query = total_time / n_iters
    avg_power = np.mean(power_samples)
    energy_per_query = avg_power * latency_per_query

    return dict(gpu=torch.cuda.get_device_name(gpu_index), n_cand=n_cand, n_iters=n_iters, total_time=total_time,
                latency_us=latency_per_query*1e6, avg_power_W=avg_power,
                energy_per_query_uJ=energy_per_query*1e6, n_power_samples=len(power_samples))

if __name__ == '__main__':
    results = []
    for gpu_idx in [0, 1]:
        for label, n_cand in [('OptionA_cand10988', 10988), ('OptionC_v4_cand10292', 10292)]:
            r = measure(gpu_idx, n_cand)
            r['label'] = label
            results.append(r)
            print(r)
    df = pd.DataFrame(results)
    df.to_csv('gpu_measure_2048_results.csv', index=False)
    print("\nSaved gpu_measure_2048_results.csv")
