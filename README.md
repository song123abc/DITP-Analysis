# DITP-Analysis

独立的 SPM 轨迹聚类与平台长度分析网页服务。用户上传无表头 CSV、XLSX、XLS、CSV.GZ 或 ZIP
文件后，一次完成 AllTrace 平台-噪音边界识别、KMeans++ 聚类和平台长度统计。

## 当前方法

- 聚类 ROI：`x = 0..2 nm`，`log10(G/G0) = -5..-2`。
- 每条轨迹转换为 `28 x 28` 二维直方图，L1 归一化后逐元素开平方。
- KMeans++：`K=3`、`n_init=10`、随机种子 `42`；不使用 PCA。
- Cluster 标签按平均电导从高到低重排为 Cluster 1、2、3。
- AllTrace 一维电导投影自动找到低电导背景峰、平台峰及两者之间的低谷。
- 平台起点为首次进入不高于 `0.1 G0` 的位置；终点为起点后边界窗口内的最右侧点。
- 两侧各裁掉 10 个点，使用 OLS 拟合，`|slope| < 8` 才接受。
- 长度为裁剪区间的 `x_end - x_start`，不做 snapback 修正。

ZIP 压缩包必须只包含一个 CSV、XLSX 或 XLS 数据文件；GZIP 压缩文件必须是
`CSV.GZ`。压缩文件只在内存中解压，不写入长期存储。对于较大的 CSV，优先使用
`CSV.GZ` 上传以减少网络传输量。

## 本地运行

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run webapp/app.py
```

浏览器打开 Streamlit 输出的本地地址即可。

## 部署到 Streamlit Community Cloud

1. 将本目录作为 GitHub 仓库 `DITP-Analysis` 推送。
2. 登录 [Streamlit Community Cloud](https://share.streamlit.io/)。
3. 选择 GitHub 仓库、分支和文件 `webapp/app.py`。
4. Python 版本由 `runtime.txt` 指定，依赖由 `requirements.txt` 安装。

应用仅在当前进程内处理上传数据，不写入长期 `outputs/` 目录。网页只生成 Plotly
图表和可下载的 CSV，不生成论文 PNG/PDF。
