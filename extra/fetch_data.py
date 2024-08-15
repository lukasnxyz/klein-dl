import argparse, urllib.request, os

data = ['mnist.npz', 'test1', 'test2']

urls = {'mnist.npz': 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz'}

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--download', choices=data, required=True, help='Data set to download')
  parser.add_argument('--download_path', default='$HOME/.cache', help='Data set to download')
  # parser download path
  args = parser.parse_args()

  dl_filename:str = vars(args)['download']
  dl_path:str = vars(args)['download_path']
  print('downloading', dl_filename, 'from', urls[dl_filename])

  # default download path is $HOME/.cache
  # change to dl path
  # check if already exists

  # doesn't work currently because of '$HOME'
  urllib.request.urlretrieve(urls[dl_filename], os.path.join(dl_path, dl_filename))