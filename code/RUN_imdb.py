import torch_main
from tensorflow import keras


#imdb
imdb = keras.datasets.imdb
data = imdb.load_data(num_words=10000)
classes = 2

torch_main.main_process( data, classes )
