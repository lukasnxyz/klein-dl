import argparse, urllib.request, os
from pathlib import Path

dir_home = str(Path.home())
dir_home_cache = os.path.join(dir_home, '.cache/')
data = ['mnist.npz', 'test1', 'test2']

urls = {'mnist.npz': 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz'}

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--download', choices=data, required=True, help='Data set to download')
  parser.add_argument('--path', default=dir_home_cache, help='Data set to download')
  args = parser.parse_args()

  dl_filename = vars(args)['download']
  dl_path = vars(args)['path']
  print('downloading', dl_filename, 'from', urls[dl_filename])

  # check/assert path is correct if not default
  if os.path.exists(os.path.join(dl_path, dl_filename)): print('data already downloaded!')
  else: urllib.request.urlretrieve(urls[dl_filename], os.path.join(dl_path, dl_filename))