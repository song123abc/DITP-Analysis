# DITP-Analysis

DITP-Analysis 是一个面向 SPM 轨迹数据的网页分析工具。用户上传数据后，可以在
一次分析中完成 AllTrace 平台-噪音边界识别、三类轨迹聚类和平台长度统计，并在
浏览器中查看交互图表和下载结果。

## 在线演示

- 在线演示：[DITP-Analysis](https://ditp-analysis.streamlit.app/)
- 源代码：[github.com/song123abc/DITP-Analysis](https://github.com/song123abc/DITP-Analysis)

## 主要功能

- 支持无表头 CSV、XLSX、XLS、CSV.GZ 和单文件 ZIP 数据；
- 自动识别 AllTrace 平台-噪音边界，也允许手动调整后重算平台长度；
- 使用 L1 归一化二维直方图平方根特征和 KMeans++ 完成三类聚类；
- 展示 AllTrace、Cluster、代表轨迹和平台长度等交互结果；
- 提供聚类结果和平台长度 CSV 下载；
- 不长期保存用户上传的数据和分析结果。

输入数据应由交替排列的 `x/y` 列组成，每两列表示一条轨迹。ZIP 压缩包只能包含
一个 CSV、XLSX 或 XLS 文件；CSV.GZ 和 ZIP 只在内存中解压。

## 本地运行

需要 Python 3.11 或兼容版本。

```bash
git clone https://github.com/song123abc/DITP-Analysis.git
cd DITP-Analysis

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m streamlit run webapp/app.py
```

Windows 激活虚拟环境使用：

```powershell
.venv\Scripts\activate
```

启动后，在浏览器中打开终端显示的本地地址，通常为
`http://localhost:8501`。

## 部署

本项目可以直接部署到
[Streamlit Community Cloud](https://share.streamlit.io/)：

1. 登录 Streamlit Community Cloud 并连接 GitHub；
2. 选择仓库 `song123abc/DITP-Analysis`；
3. 选择分支 `main`；
4. 将入口文件设置为 `webapp/app.py`；
5. 点击 Deploy，等待依赖安装和服务启动。

后续推送到 GitHub `main` 分支的更新会由 Streamlit Cloud 自动重新部署。

## 数据与隐私

应用在当前会话中处理上传文件，不创建长期 `outputs/` 目录，也不生成论文
PNG/PDF。部署者仍应根据数据来源和所在机构的要求配置访问权限，并提醒用户不要
上传无权交由第三方云服务处理的敏感数据。

## License

本项目使用 [MIT License](LICENSE)。
