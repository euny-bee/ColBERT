"""RTX 3080Ti / RTX 3070Ti 실측: coarse search + scoring(MaxSim) 워크로드, per-query 에너지 측정.
coarse search: 32 query tokens x 100 centroids (실제 [clip99.9] 데이터)
scoring: 32 query tokens x candidates(Option A=184, Option C=193), 128-dim MaxSim
"""
import torch, time, threading, sys
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

def measure(gpu_index, n_cand, duration_s=4.0, warmup=500):
    dev = torch.device(f'cuda:{gpu_index}')
    torch.cuda.set_device(dev)

    Q_np, C_np = load_qc()
    Q = torch.tensor(Q_np, device=dev)          # (32,128)
    C = torch.tensor(C_np, device=dev)          # (100,128)
    rng = np.random.default_rng(0)
    D_np = rng.normal(0, Q_np.std(), size=(n_cand,128)).astype(np.float32)  # 후보 문서(candidate) 임베딩 -- 실제 통계와 일치하는 합성 데이터
    D = torch.tensor(D_np, device=dev)

    def workload():
        # coarse search: Q(32,128) x C(100,128)^T -> (32,100), top-nprobe 선택 (여기선 전체 스캔 자체가 연산비용)
        coarse = Q @ C.T
        _ = coarse.topk(2, dim=1)
        # scoring (MaxSim): Q(32,128) x D(n_cand,128)^T -> (32,n_cand), token별 max, 합산
        score = Q @ D.T
        maxsim = score.max(dim=1).values.sum()
        return maxsim

    # warmup
    for _ in range(warmup):
        workload()
    torch.cuda.synchronize(dev)

    # pynvml power 폴링 (백그라운드 스레드)
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
    power_samples = []
    stop_flag = threading.Event()
    def poll_power():
        while not stop_flag.is_set():
            p = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW -> W
            power_samples.append(p)
            time.sleep(0.02)
    t = threading.Thread(target=poll_power)
    t.start()

    # 고정 시간(duration_s) 동안 최대한 back-to-back으로 워크로드 반복 실행 (GPU가 지속적으로 바쁜 상태 유지)
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
                energy_per_query_nJ=energy_per_query*1e9, n_power_samples=len(power_samples))

def measure_idle(gpu_index, duration_s=2.0):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
    samples = []
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < duration_s:
        samples.append(pynvml.nvmlDeviceGetPowerUsage(handle)/1000.0)
        time.sleep(0.02)
    pynvml.nvmlShutdown()
    return np.mean(samples)

if __name__ == '__main__':
    results = []
    for gpu_idx in [0, 1]:
        idle_p = measure_idle(gpu_idx)
        print(f"GPU {gpu_idx} idle power = {idle_p:.2f} W")
        for label, n_cand in [('OptionA_cand184', 184), ('OptionC_cand193', 193)]:
            r = measure(gpu_idx, n_cand)
            r['label'] = label
            r['idle_power_W'] = idle_p
            r['energy_per_query_incl_idle_nJ'] = r['energy_per_query_nJ']
            r['energy_per_query_net_nJ'] = max(0, r['avg_power_W']-idle_p) * (r['latency_us']*1e-6) * 1e9
            results.append(r)
            print(r)
    df = pd.DataFrame(results)
    df.to_csv('gpu_measure_results.csv', index=False)
    print("\nSaved gpu_measure_results.csv")
