# DITP-Analysis

DITP-Analysis 是一个用于 SPM 轨迹聚类和平台长度分析的 Streamlit 网页工具。
上传数据后即可在浏览器中完成分析、查看交互图表并下载结果。

## 核心功能

- 支持上传无表头 CSV、XLSX 和对应的 ZIP 压缩文件，输入文件相邻两列为一条轨迹的 `x`、`y` 数据；
- 自动识别轨迹中的平台-噪音边界；
- 支持设置聚类数量 K 和位移/电导 ROI；
- 完成 KMeans++ 聚类、平台长度计算和交互式结果展示；
- 支持下载聚类结果和平台长度 CSV。

## 本地部署

需要安装 Git 和 Python 3.11 或兼容版本。

### macOS / Linux

```bash
git clone https://github.com/song123abc/DITP-Analysis.git
cd DITP-Analysis

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m streamlit run webapp/app.py
```

### Windows PowerShell

```powershell
git clone https://github.com/song123abc/DITP-Analysis.git
cd DITP-Analysis

py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run webapp/app.py
```

启动后访问终端显示的本地地址，通常为 `http://localhost:8501`。

## 在线演示

[https://ditp-analysis.streamlit.app/](https://ditp-analysis.streamlit.app/)
