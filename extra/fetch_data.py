import argparse, urllib.request

data = ['mnist.npz', 'test1', 'test2']

urls = {'mnist.npz': 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz'}

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--download', choices=data, required=True, help='Data set to download')
  # parser download path
  args = parser.parse_args()

  dl_filename:str = vars(args)['download']
  print('downloading', dl_filename, 'from', urls[dl_filename])

  # change to dl path
  # check if already exists
  urllib.request.urlretrieve(urls[dl_filename], dl_filename)