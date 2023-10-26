import os
import requests
from tqdm import tqdm

from deforum import default_cache_folder

def fetch_and_download_model(modelId:str, destination:str=default_cache_folder):
    # Fetch model details
    response = requests.get(f"https://civitai.com/api/v1/models/{modelId}")
    response.raise_for_status()
    model_data = response.json()



    download_url = model_data['modelVersions'][0]['downloadUrl']
    filename = model_data['modelVersions'][0]['files'][0]['name']

    print(download_url)

    print(filename)
    dir_path = destination
    print(dir_path)

    os.makedirs(dir_path, exist_ok=True)
    filepath = os.path.join(dir_path, filename)
    print(filepath)

    # Check if file already exists
    if os.path.exists(filepath):
        print(f"File {filename} already exists in models/checkpoints/")
        return

    # Download file in chunks with progress bar
    print(f"Downloading {filename}...")
    response = requests.get(download_url, stream=True, headers={'Content-Disposition': 'attachment'})
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    t = tqdm(total=total_size, unit='iB', unit_scale=True)
    with open(filepath, 'wb') as f:
        for data in response.iter_content(block_size):
            t.update(len(data))
            f.write(data)
    t.close()

    if total_size != 0 and t.n != total_size:
        print("ERROR: Something went wrong while downloading the file.")
    else:
        print(f"{filename} downloaded successfully!")

    return filename