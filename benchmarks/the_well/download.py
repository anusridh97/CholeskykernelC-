"""Download and verify the pinned 14 GB The Well HDF5 source."""
import argparse
import hashlib
import json
from pathlib import Path

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',type=Path,default=Path('data/the_well/post_neutron_star_merger'));args=parser.parse_args()
    from huggingface_hub import hf_hub_download
    dataset=json.loads(Path(__file__).with_name('protocol.json').read_text())['dataset']
    path=Path(hf_hub_download(repo_id=dataset['repository'],repo_type='dataset',filename=dataset['relative_path'],revision=dataset['revision'],local_dir=str(args.output_dir)))
    if path.stat().st_size!=dataset['bytes']:raise ValueError('raw file size mismatch')
    digest=hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda:handle.read(8<<20),b''):digest.update(chunk)
    if digest.hexdigest()!=dataset['sha256']:raise ValueError('raw file SHA-256 mismatch')
    print('Pinned dataset size and SHA-256 verified')
if __name__=='__main__':main()
