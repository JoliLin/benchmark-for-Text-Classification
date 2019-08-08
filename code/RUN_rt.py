import torch_main
import rt_data as rt

#rt-polarity
path = 'dataset/rt-polarity.all'
data = rt.rt_data(path).load_data()
classes = 2
	
torch_main.main_process(data, classes)
