hardware_params = {
    # https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf
    # NOTICE: V100 not support INT8 in tensor core, so INT8 performance is not good


    "rockchip_rk3588": {
        "mem_bandwidth": 51.2e9, #or 51.2 GB/s if LRDDR5
        "FP32": 3e12,
        "FP16": 6e12,
        "INT8": 12e12,
        "onchip_buffer": 384e3*3,
        #"network": 300e9,
        "num": 1,
    },


    # "rockchip_rk3588": {
    #     "mem_bandwidth": 17.8,
    #     "FP32": 3,
    #     "FP16": 6,
    #     "INT8": 12,
    #     "onchip_buffer": 0.384*3,
    #     #"network": 300e9,
    #     "num": 1,
    # },
    "nvidia_V100": {
        "mem_bandwidth": 900e9,
        "FP32": 14e12,
        "FP16": 112e12,
        "INT8": 62e12,
        "onchip_buffer": 20480e3,
        "network": 300e9,
        "num": 4,
    },
    # https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.pdf
    "nvidia_A6000": {
        "mem_bandwidth": 768e9,
        "FP16": 309.677e12 / 2,
        "INT8": 309.7e12,
        "onchip_buffer": 21504e3,
        "network": 112e9,
        "num": 2,
    },
    # https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf
    # Ampere's SM has 256KB RF, max 164KB Shared Mem
    "nvidia_A100": {
        "mem_bandwidth": 1555e9,
        "FP16": 312e12,
        "INT8": 624e12,
        "onchip_buffer": 27648e3,
        "network": 600e9,
        "num": 4,
    },  # use 40G data
    "nvidia_A100_40G": {
        "mem_bandwidth": 1555e9,
        "FP16": 312e12,
        "INT8": 624e12,
        "onchip_buffer": 27648e3,
        "network": 600e9,
        "num": 4,
    },
    "nvidia_A100_80G": {
        "mem_bandwidth": 2039e9,
        "FP16": 312e12,
        "INT8": 624e12,
        "onchip_buffer": 27648e3,
        "network": 600e9,
        "num": 4,
    },
    "nvidia_A800_80G_SXM": {
        "mem_bandwidth": 2039e9,
        "FP16": 312e12,
        "INT8": 624e12,
        "onchip_buffer": 27648e3,
    },
    "nvidia_A40": {
        "mem_bandwidth": 696e9,
        "FP16": 149.7e12,
        "INT8": 299.3e12,
        "onchip_buffer": 21504e3,
    },
    # https://resources.nvidia.com/en-us-tensor-core/gtc22-whitepaper-hopper
    "nvidia_H100": {
        "mem_bandwidth": 3072e9,
        "FP16": 1979e12 / 2,
        "INT8": 3958e12 / 2,
        "onchip_buffer": 33792e3,
        "network": 900e9,
        "num": 4,
    },  # use SXM data
    "nvidia_H100_SXM": {
        "mem_bandwidth": 3072e9,
        "FP16": 1979e12 / 2,
        "INT8": 3958e12 / 2,
        "onchip_buffer": 33792e3,
        "network": 900e9,
        "num": 4,
    },
    "nvidia_H100_PCIe": {
        "mem_bandwidth": 2048e9,
        "FP16": 1513e12 / 2,
        "INT8": 3026e12 / 2,
        "onchip_buffer": 29184e3,
        "network": 32e9,
        "num": 4,
    },
    # https://images.nvidia.com/aem-dam/Solutions/Data-Center/l4/nvidia-ada-gpu-architecture-whitepaper-v2.1.pdf
    # Ada SM has 256 KB Register File, and 128 KB of L1/Shared Memory
    "nvidia_L40": {
        "mem_bandwidth": 864e9,
        "FP16": 181e12,
        "INT8": 362e12,
        "onchip_buffer": 36352e3,
        "network": 32e9,
        "num": 4,
    },
    "nvidia_T40": {
        "mem_bandwidth": 864e9,
        "FP16": 181e12,
        "INT8": 362e12,
        "onchip_buffer": 36352e3,
        "network": 32e9,
        "num": 4,
    },
    "nvidia_L4": {
        "mem_bandwidth": 300e9,
        "FP32": 30.3e12,
        "FP16": 121e12,
        "INT8": 485e12,
        "onchip_buffer": 48000e3,
        "network": 32e9,
        "num": 1,
    },
    "nvidia_T4": {
        "mem_bandwidth": 320e9,
        "FP32": 8.1e12,
        "FP16": 64.8e12,
        "INT8": 130e12,
        "INT4": 260e12,
        "onchip_buffer": 4000e3,
        "network": 32e9,
        "num": 1,
    },
    # Intel Skylake-X (Skylake-X, Cascade Lake) Intel Xeon Phi (Knights Landing, Knights Mill) Intel Ice Lake, Tiger Lake and Rocket Lake
    # support AVX-512 & FMA (512-bit), they has throughput of 1 cycle
    # https://www.intel.com/content/www/us/en/products/sku/230496/intel-core-i913900k-processor-36m-cache-up-to-5-80-ghz/specifications.html
    "intel_13900k": {
        "mem_bandwidth": 89.6e9,
        "FP16": 8 * 5.4e9 * (512 / 16),
        "onchip_buffer": 36e6,
    },
}

GPU_type = {
    "A100": {"code": 1},
    "L4": {"code": 2},
    "T4": {"code": 3},
    "T4": {"code": 3},
}


avaliable_hardwares = [_ for _ in hardware_params.keys()]


def roofline_analyze(mem_bandwidth, max_OPS, OPs, memory_access):
    # mem_bandwidth is bytes/s
    # memory_access in byte
    # x axis is OPS/byte
    # y axis is OPS/s
    y_max = max_OPS
    memory_access_bytes = memory_access
    turning_point = y_max / mem_bandwidth
    arithmetic_intensity = OPs / memory_access_bytes
    if arithmetic_intensity < turning_point:
        bound = "memory"
        performance = arithmetic_intensity * mem_bandwidth
    else:
        bound = "compute"
        performance = y_max
    if performance == 0:
        1 == 1
        pass
    return arithmetic_intensity, performance, bound
