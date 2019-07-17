import argparse

def process_command():
	parser = argparse.ArgumentParser(prog='Training', description='Arguments')
	parser.add_argument('--gpu', '-g', default=0, help='-1=cpu, 0, 1,...= gpt', type=int)
	#parser.add_argument('--model', '-model', default='./model', help='path of model')
	parser.add_argument('--epoch', '-epoch', default=300, type=int)
	parser.add_argument('--batch', '-batch', default=64, help='batch size', type=int)
	

	return parser.parse_args()
