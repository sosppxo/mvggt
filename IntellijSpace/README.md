# IntellijSpace：MVGGT 前后端应用

上传多视角图片或视频，通过自然语言进行三维目标分割、移除和家具替换，预览并下载 GLB。
前端使用 React、Vite 和 React Three Fiber，后端使用 FastAPI。

## 目录与模型版本

- backend/：HTTP API、任务编排、指令解析、几何编辑。
- frontend-react/：上传、任务状态、Three.js 预览与变换交互。
- mvggt/：本应用使用的模型源码快照。
- example/、glb/、image/：示例图片、家具 GLB、素材图片。
- app.py：独立 Gradio 演示及 predict_remote 接口实现。
- backend_workspace/、models/、ckpts/：运行时生成或自行准备，不提交。

本应用的 mvggt/models/mvggt_training.py 与仓库根目录模型版本不同，且多出
mvggt/models/mvggt.py。因此保留完整应用模型源码，不覆盖仓库原有训练模型。
请从本目录运行命令，或使用 run_backend.py 的文件路径启动，避免导入错误版本。

## 安装

以下命令从仓库根目录执行，已创建环境时可以跳过 conda create：

    cd IntellijSpace
    conda create -n mvggt python=3.10 -y
    conda activate mvggt
    python -m pip install -r requirements.txt
    Copy-Item .env.example .env

编辑 .env，填入 MVGGT_LLM_API_KEY。默认通过 Qwen 解析自然语言。
不提供有效密钥时，默认配置的指令解析无法正常调用。
如只需目标分割，可把 MVGGT_LLM_API_URL 留空，此时原始文本直接作为分割目标，
不支持自然语言识别 REMOVE / REPLACE。
真实 .env 已被忽略，不要提交密钥。

requirements_demo.txt 保留原始模型和 Gradio 依赖版本；
requirements.txt 在其基础上补充 Web 后端依赖。
本地 GPU 推理需要匹配驱动的 CUDA 版 PyTorch（本项目固定 torch 2.5.1 /
torchvision 0.20.1）。安装后检查：

    python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

## 启动前后端

在本目录运行后端：

    conda activate mvggt
    python run_backend.py

API 文档：http://127.0.0.1:8001/docs
健康检查：http://127.0.0.1:8001/api/v1/health

另开终端，在本目录运行：

    cd frontend-react
    npm ci
    npm run dev

浏览器打开 http://localhost:5173。Vite 将 /api/v1 代理到本机 8001 端口。
Node.js 版本要求：20.19+（20.x）或 22.12+；具体以锁定的 Vite engines 为准。

生产静态文件可用 npm run build 生成；Vite 开发代理不属于生产部署，
部署时需另外配置 /api/v1 反向代理或设置 VITE_API_BASE。
前端 API 地址也支持 URL 参数 api_base 和 localStorage.MVGGT_API_BASE。

## 推理与模型准备

- local：本地 PyTorch 推理，需足够显存及完整模型权重。
- hf_api：调用配置的 Hugging Face Space；依赖远端运行状态、额度及接口一致性。
- auto：后端根据本地 GPU 状态选择。

即使使用远程模式，当前后端仍需安装模型相关 Python 依赖。
本地权重优先读取 MVGGT_LOCAL_MODEL_PATH；不存在时从配置的 Hugging Face
仓库下载到模型缓存目录。Tokenizer 优先读取 MVGGT_TOKENIZER_PATH，
不存在时使用 roberta-base。模型文本编码器也可能需要下载 Hugging Face 文件。
本仓库不包含权重；首次运行可能需要网络和较长下载时间。
后端检测到本地 GPU 时会尝试启动预热，可能触发下载；预热失败会记录警告。

前端支持 result / nomask / mask 结果切换。
替换素材的旋转、缩放在浏览器实时预览，下载时调用 dynamic 接口导出。

## 独立 Gradio 演示

    python app.py

此入口独立于 FastAPI，启动时加载模型，推理需要 CUDA。
它保留原演示的模型下载及路径行为，不读取 FastAPI 的 .env 配置；
请从本目录启动。predict_remote 是配套远程调用接口。

## 提交范围

提交源码、依赖清单、前端锁文件和小型示例素材。
不提交 node_modules、dist、模型权重、上传文件、任务结果、Python 缓存和密钥。
此应用当前使用内存任务记录和后台线程，适合演示；重启后无法恢复任务查询。