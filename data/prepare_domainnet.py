import os
import requests
import zipfile
"""
This file is used to configure the cross-domain dataset: DomainNet, \
and the configuration process follows the guidelines of FedBN.
FedBN: https://github.com/med-air/FedBN/tree/master?tab=readme-ov-file

Before running this file, please download the "domainnet_dataset.zip" file provided by Link: \
https://mycuhk-my.sharepoint.com/:u:/g/personal/1155149226_link_cuhk_edu_hk/EUTZ_Dr9YnxLm_cGvjXJGvEBJtKUn_LxpFs9DZ2ZVS-eaw?e=N8ajKz \
and extract it to `./data/DomainNet`, renaming the extracted folder to `split`.

After that, you can run this file. Upon completion, the folder structure of `./data/DomainNet` should appear as follows:
.
├── DomainNet
│   ├── clipart
│   │   ├── aircraft_carrier
│   │   │   ├── xxx.jpg
│   │   │   ├── yyy.jpg
│   │   │   ├── ...
│   │   ├── ...
│   ├── infograph
│   │   ├── aircraft_carrier
│   │   │   ├── xxx.jpg
│   │   │   ├── yyy.jpg
│   │   │   ├── ...
│   │   ├── ...
│   ├── painting
│   │   ├── aircraft_carrier
│   │   │   ├── xxx.jpg
│   │   │   ├── yyy.jpg
│   │   │   ├── ...
│   │   ├── ...
│   ├── quickdraw
│   │   ├── aircraft_carrier
│   │   │   ├── xxx.jpg
│   │   │   ├── yyy.jpg
│   │   │   ├── ...
│   │   ├── ...
│   ├── real
│   │   ├── aircraft_carrier
│   │   │   ├── xxx.jpg
│   │   │   ├── yyy.jpg
│   │   │   ├── ...
│   │   ├── ...
│   ├── sketch
│   │   ├── aircraft_carrier
│   │   │   ├── xxx.jpg
│   │   │   ├── yyy.jpg
│   │   │   ├── ...
│   │   ├── ...
│   ├── split
│   │   ├── clipart_test.pkl
│   │   ├── clipart_train.pkl
│   │   ├── ...
"""
data_dir = os.path.join(os.getcwd(), "data")
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
    print(f"Created directory: {data_dir}")

domainnet_dir = os.path.join(data_dir, "DomainNet")
if not os.path.exists(domainnet_dir):
    os.makedirs(domainnet_dir)
    print(f"Created directory: {domainnet_dir}")

domain_list = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
for domain in domain_list:
    if domain == 'clipart' or domain == 'painting':
        current_url = f'http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/{domain}.zip'
    else: 
        current_url = f'http://csr.bu.edu/ftp/visda/2019/multi-source/{domain}.zip'
    zip_path = os.path.join(domainnet_dir, f"{domain}.zip")
    if not os.path.exists(zip_path):
        print(f"Downloading {domain}.zip...")
        response = requests.get(current_url, stream=True)
        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        print(f"Downloaded {domain}.zip to {zip_path}")
    # unzip 
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(domainnet_dir)
        print(f"Extracted {domain}.zip to {domainnet_dir}")
    # remove zip
    os.remove(zip_path)
    print(f"Removed zip file: {zip_path}")

