# fnos-maxwell-ort

为飞牛 fnOS 相册 AI 构建的 ONNX Runtime 1.19.0，带 **NVIDIA Maxwell (sm_50)** GPU 支持。

Build of ONNX Runtime 1.19.0 that enables NVIDIA Maxwell (compute capability 5.0 / 5.2) GPUs for the fnOS photo app's AI inference.

---

## 为什么需要它 / Why

飞牛相册的智能识别与以文搜图由 `/usr/trim/bin/ai_manager` 拉起子进程，通过 `PYTHONPATH` 加载 `@sysappmeta/ai-manager/python/*/site-packages/onnxruntime`。这里有两个问题：

**1. 官方 wheel 不含 sm_50 kernel**

官方预编译包只包含 sm_60 及以上架构，Maxwell 加载模型时报 `cudaErrorNoKernelImageForDevice`，直接回退 CPU。

**2. 2 GB 显存被 arena 策略吃满**

`CUDAExecutionProviderInfo` 默认 `arena_extend_strategy = kNextPowerOfTwo`，arena 按 2 的幂预分配。img2vec-large 在 2 GB 卡上会占到 1984 MiB，只剩十几 MiB，首次 `session.run()` 创建 cuBLAS 句柄时抛 `CUBLAS_STATUS_ALLOC_FAILED`。

> **关键点**：这个参数在 ORT 里有两套取值路径。
> 显式传 `providers=[('CUDAExecutionProvider', {...})]` 走 `FromProviderOptions()`，用头文件里的 in-class 默认值；
> 裸传 `providers=['CUDAExecutionProvider']` 走 `else` 分支，用 `onnxruntime_pybind_state_common.cc` 里的**全局变量**。
> 只改头文件对 fnOS 无效——ai_manager 传的正是裸字符串。两个补丁必须同时打。

## 支持的显卡 / Supported GPUs

判断标准是 `compute_cap` 为 **5.0 或 5.2**，显存 **≥ 2 GB**。

```bash
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv
```

| 架构 | 核心 | 算力 | 桌面 | 笔记本 |
|---|---|---|---|---|
| Maxwell 1.0 | GM107 / GM108 | 5.0 | GTX 745 / 750 / 750 Ti | 920M / 930M / 940M / GTX 850M / 860M / 950M / 960M |
| Maxwell 2.0 | GM204 / GM206 / GM200 | 5.2 | GTX 950 / 960 / 970 / 980 / 980 Ti / Titan X | GTX 965M / 970M / 980M / 980（笔记本） |

专业卡：Quadro K620 / K1200 / K2200 / K2200M / M1000M / M2000M（5.0），M3000M / M4000M / M5000M / M4000 / M5000 / M6000（5.2），Tesla M4 / M6 / M40 / M60（5.2）。

**不支持**：Kepler（算力 3.x）、Pascal 及以上（官方版即支持）、AMD / Intel 显卡。

本仓库按 `sm_50` 编译。CUDA 在同一大版本内向上二进制兼容，5.2 的卡也能跑；实测只在 940M（5.0）上完成。

## 构建 / Build

```bash
# 1. 构建镜像
docker build -t fnos-maxwell-ort:1.19.0 .

# 2. 首次全量编译（1-3 小时，建议 ≥ 16 GB 内存 + 10 核）
docker run --rm -it \
  -v "$PWD/build:/build" -v "$PWD/dist:/dist" \
  --shm-size=2g -m 16g \
  fnos-maxwell-ort:1.19.0 \
  bash -c "bash /build/../scripts/build.sh"

# 3. 取出四个运行时文件到 dist/
docker run --rm -it \
  -v "$PWD/build:/build" -v "$PWD/dist:/dist" \
  fnos-maxwell-ort:1.19.0 \
  bash -c "bash /build/../scripts/collect.sh /dist"
```

由于脚本在镜像里没有挂载，实际使用时把仓库挂到 `/repo` 更直接：

```bash
docker run --rm -it \
  -v "$PWD:/repo" -v "$PWD/build:/build" -v "$PWD/dist:/dist" \
  --shm-size=2g -m 16g \
  fnos-maxwell-ort:1.19.0 \
  bash -c "bash /repo/scripts/build.sh && bash /repo/scripts/collect.sh /dist"
```

增量重编（ccache 已就绪）通常 20-40 秒。

### 关键编译参数

```
--use_cuda
--cuda_home /usr/local/cuda
--cudnn_home /usr
--cuda_version 12.6
--cmake_extra_defines CMAKE_CUDA_ARCHITECTURES=50     # ← Maxwell，不可省略
--enable_pybind --build_shared_lib --build_wheel
--compile_no_warning_as_error --allow_running_as_root --skip_tests
--parallel $(nproc) --nvcc_threads 1                  # nvcc_threads 过大会爆内存
```

## 产物 / Artifacts

| 文件 | 说明 |
|---|---|
| `onnxruntime_pybind11_state.so` | Python 扩展，**静态链接了 ORT 核心**（25 MB）。`libonnxruntime.so` 实际不会被加载 |
| `libonnxruntime_providers_cuda.so` | CUDA EP，含 sm_50 kernel（120 MB） |
| `libonnxruntime_providers_shared.so` | provider 共享符号 |
| `libonnxruntime.so.1.19.0` | CPU 核心，pybind 静态链接后通常不加载，保留以兼容其他调用方式 |

## 补丁 / Patches

| 补丁 | 文件 | 作用 | 必要性 |
|---|---|---|---|
| `0001-cuda-ep-info-arena.patch` | `core/providers/cuda/cuda_execution_provider_info.h` | arena 默认值改为 `kSameAsRequested` | 必需（显式传 options 的路径） |
| `0002-pybind-globals-arena.patch` | `python/onnxruntime_pybind_state_common.cc` | 全局变量 arena 改为 `kSameAsRequested` | **必需（裸传 providers 的路径，fnOS 走这条）** |
| `0003-disable-mem-pattern.patch` | `core/framework/session_options.h` | `enable_mem_pattern = false` | 可选，缩短会话初始化 0.7~1.5 s |
| `0004-moe-kernel-sm50.patch` | `contrib_ops/cuda/moe/ft_moe/moe_kernel.cu` | 修 sm_50 下未使用参数的编译错误 | 必需（否则编译失败） |

### 实测效果（GeForce 940M, 2 GB）

| 模型 | 默认策略 | 打补丁后 |
|---|---|---|
| img2vec-large | 1984 MiB → `CUBLAS_STATUS_ALLOC_FAILED` | **1256 MiB**，推理正常 |
| txt2vec-large | 1984 MiB → 失败 | **约 1.65 GB**，推理正常 |

推理后稳定在 1300 MiB，GPU 利用率 90-100%。

## 在 fnOS 上使用 / Install on fnOS

**方式一：手动替换**（先备份）

```bash
SITE=$(ls -d /vol*/@sysappmeta/ai-manager/python/*/lib/python3*/site-packages | head -1)
CAP="$SITE/onnxruntime/capi"

cp -a "$CAP" "${CAP}.bak.$(date +%Y%m%d)"
cp -a dist/*.so* "$CAP/"
rm -rf "$SITE/onnxruntime/__pycache__"
systemctl restart ai_manager
```

**方式二：打包成 FPK**（推荐分发用）

FNOS 应用包需要 `manifest` + `cmd/` 生命周期脚本 + `overlay-amd64/app/`。可用 [fn-fpk-builder-skill](https://github.com/) 的 `fpk.py` 构建：

```bash
fpk.py build --project ./packages/ort-maxwell --out ./dist \
  --arch amd64 --overlay-amd64 ./overlay-amd64
```

### 验证是否生效

```bash
# 触发一次智能识别后，观察显存落点
nvidia-smi --query-gpu=memory.used --format=csv,noheader -lms 500
```

- 生效：加载后稳定 **1256~1300 MiB**，无 CUDA 报错
- 未生效：冲到 **1984 MiB** 后归零，`dmesg`/子进程日志出现 `CUBLAS_STATUS_ALLOC_FAILED`

也可以直接量一次：

```bash
python3 -c "
import onnxruntime as ort
s = ort.InferenceSession('$SITE/../../models/img2vec-large-model-1.0.0/img2vec.onnx',
                         ort.SessionOptions(), providers=['CUDAExecutionProvider'])
print(s.get_providers())"
```

## 已验证环境

```
OS        Ubuntu 22.04.5（构建）/ fnOS（运行）
GPU       GeForce 940M, 2 GB, compute 5.0
CUDA      12.6.85
cuDNN     9.5.1
Python    3.11.17（deadsnakes）
ORT       v1.19.0 (26250ae)
cmake     3.28.3 / gcc 11.4
```

注意：ai_manager 的推理子进程是 **Nuitka 打包的原生二进制**（`Py_NoSiteFlag`），不读 `.pth`，因此只能整目录替换 `site-packages/onnxruntime`，不要试图用 `.pth` 注入。

## 已知限制

1. 系统/模型升级可能覆盖 `site-packages/onnxruntime`，需要重新安装
2. 2 GB 显存是底线；与其它吃显存的服务（视频硬解等）同时运行可能相互挤占
3. `libonnxruntime.so` 通常不被加载（pybind 静态链接核心），替换主库不会改变行为
4. 仅 x86_64 + NVIDIA Maxwell

## License

本仓库的补丁与脚本：MIT。
ONNX Runtime 本体：MIT，版权归 Microsoft 及贡献者所有，见 [上游仓库](https://github.com/microsoft/onnxruntime)。
