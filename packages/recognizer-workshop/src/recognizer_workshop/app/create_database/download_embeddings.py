# helpers to download embeddings for building hoplite databases
# from the old workshop

from pathlib import Path
import logging
import requests
import zipfile
from tqdm import tqdm
import ipywidgets as widgets
from IPython.display import display

log = logging.getLogger(__name__)

def download_embeddings(dataset_name, 
                        embeddings_dir, 
                        download_from_gcp: bool = False):
    """
    Downloads a zip file from a url based on the dataset_name and extracts it to the embeddings_dir
    Shows progress for both download and extraction
    """

    def has_embeddings(directory):
        directory = Path(directory)
        # Return True as soon as we find any .pt file
        try:
            next(directory.glob('**/*.parquet'))
            return True
        except StopIteration:
            return False
     
    def do_download():
        
        embeddings_dir.mkdir(parents=True, exist_ok=True)

        base_url = (
            'https://storage.googleapis.com/chirp-public-bucket/esa-2024'
            if download_from_gcp
            else 'https://api.ecosounds.org/system/esa2024'
        )

        #dowload config.json
        url = f'{base_url}/embedding_config.json'
        config_path = Path(embeddings_dir) / "config.json"
        response = requests.get(url)
        with open(config_path, 'wb') as file:
            file.write(response.content)
        
        url = f'{base_url}/{dataset_name}/embeddings.zip'
        
        download_file(url, zip_path, description=f'Downloading {dataset_name} to {embeddings_dir}')

    def do_unzip():

        # Extract with progress bar
        with zipfile.ZipFile(zip_path) as zf:
            for member in tqdm(zf.infolist(), desc=f'Extracting {dataset_name}'):
                zf.extract(member, embeddings_dir)
        
        # Optionally remove zip after extraction
        zip_path.unlink()

    def do_download_and_unzip():
        do_download()
        do_unzip()

    def download_again_button():
        button1 = widgets.Button(description="Download again and overwrite existing embeddings?",
                                 layout=widgets.Layout(width='auto'))
        button1.on_click(lambda b: do_download_and_unzip())
        display(button1)

    def unzip_again_button(description="Unzip existing downloaded file and overwrite embeddings?"):
        button2 = widgets.Button(description=description,
                                 layout=widgets.Layout(width='auto'))
        button2.on_click(lambda b: do_unzip())
        display(button2)
        
    embeddings_dir = Path(embeddings_dir)
    zip_path = Path(embeddings_dir) / f"{dataset_name}.zip"

    already_unzipped = has_embeddings(embeddings_dir)
    already_downloaded = zip_path.exists()

    if already_unzipped:

      log.info(f"Embeddings already downloaded and extracted at {embeddings_dir.resolve()}.")
      download_again_button()

      if already_downloaded:
        # might happen if they unterrupt during unzip 
        # or if they interrupt during download after already downloadin and unzipping
        # (in this 2nd case thy probably don't want to unzip without downloading again)
        unzip_again_button()

    elif already_downloaded:
      # This can happen if they interrupt during download. 
      # (in this case they probably don't want to unzip without downloading)
      # TODO: (phil) check if the zip is complete, e.g. comparing file size to the response headers
      log.info(f"Embeddings already downloaded to {zip_path} but not unzipped.")
      download_again_button()
      unzip_again_button("Unzip existing downloaded file?")

    else:
      do_download()
      do_unzip()


def download_file(url: str, dest: str | Path, max_retries: int = 5, description: str = None) -> bool:
    """
    Download a file from a URL to a destination path with progress bar and resume capability.
    
    Args:
        url: URL to download from
        dest: Destination path (string or Path object)
        max_retries: Maximum number of retry attempts (default: 5)
        description: Optional description for the progress bar. If None, uses filename
    
    Returns:
        bool: True if download was successful
    
    Raises:
        Exception: If download fails after all retries
    """

    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    from tqdm import tqdm
    import time

    dest = Path(dest)
    description = description or dest.name
    
    # Set up retry strategy
    retry_strategy = Retry(
        total=max_retries,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    
    # Configure session
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': '*/*',
        'Accept-Encoding': 'gzip, deflate'
    }
    
    for attempt in range(max_retries):
        try:
            # Check file size and resume support
            head = session.head(url, headers=headers)
            total_size = int(head.headers.get('content-length', 0))
            supports_resume = 'accept-ranges' in head.headers
            
            existing_size = 0
            if dest.exists():
                existing_size = dest.stat().st_size
                if existing_size == total_size:
                    log.info(f"File already completely downloaded ({existing_size} bytes)")
                    return True
                elif supports_resume and existing_size < total_size:
                    log.info(f"Resuming download from byte {existing_size}")
                    headers['Range'] = f'bytes={existing_size}-'
                else:
                    log.info("Cannot resume download - starting from beginning")
                    existing_size = 0
            
            dest.parent.mkdir(parents=True, exist_ok=True)
            
            response = session.get(
                url,
                stream=True,
                headers=headers,
                timeout=30
            )
            response.raise_for_status()
            
            mode = 'ab' if existing_size else 'wb'
            chunk_size = 8 * 1024 * 1024  # 8MB chunks
            
            with open(dest, mode) as file, tqdm(
                desc=description,
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
                initial=existing_size
            ) as pbar:
                for data in response.iter_content(chunk_size=chunk_size):
                    size = file.write(data)
                    pbar.update(size)
            
            # Verify complete download
            if dest.stat().st_size == total_size:
                log.info("Download completed successfully")
                return True
            else:
                log.warning(f"Download incomplete, retrying... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(2)
                continue
                
        except (requests.exceptions.ChunkedEncodingError,
                requests.exceptions.ConnectionError,
                requests.exceptions.ReadTimeout) as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                log.warning(f"Download failed, retrying in {wait_time} seconds... (Attempt {attempt + 1}/{max_retries})")
                log.warning(f"Error: {str(e)}")
                time.sleep(wait_time)
                continue
            else:
                raise Exception(f"Failed to download after {max_retries} attempts: {str(e)}")
    
    raise Exception("Download failed after all retry attempts")


