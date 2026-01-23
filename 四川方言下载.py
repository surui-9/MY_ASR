from huggingface_hub import hf_hub_download
import zipfile

# 下载 ZIP 文件
zip_path = hf_hub_download(
    repo_id="wanghaikuan/sichuan",
    filename="sichuan.zip",
    repo_type="dataset"
)

# 查看 ZIP 内容
with zipfile.ZipFile(zip_path, 'r') as z:
    print(z.namelist())