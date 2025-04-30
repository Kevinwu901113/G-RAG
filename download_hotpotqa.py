import os
import requests

# 创建数据目录
data_dir = "data"
os.makedirs(data_dir, exist_ok=True)

# 定义数据集文件及其下载链接
datasets = {
    "hotpot_train_v1.1.json": "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json",
    "hotpot_dev_distractor_v1.json": "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json",
    "hotpot_dev_fullwiki_v1.json": "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_fullwiki_v1.json"
}

# 下载并保存文件
for filename, url in datasets.items():
    file_path = os.path.join(data_dir, filename)
    if not os.path.exists(file_path):
        print(f"Downloading {filename}...")
        response = requests.get(url)
        if response.status_code == 200:
            with open(file_path, "wb") as f:
                f.write(response.content)
            print(f"Saved {filename} to {file_path}")
        else:
            print(f"Failed to download {filename}. Status code: {response.status_code}")
    else:
        print(f"{filename} already exists. Skipping download.")
