
#ag_news
training = '/home/zllin/torch_exp_flow/dataset/Text-Classification-Benchmark/data/preprocessed/ag_news_train.txt'
testing = '/home/zllin/torch_exp_flow/dataset/Text-Classification-Benchmark/data/preprocessed/ag_news_test.txt'
training_ag = rt.rt_data(training, splitted=False, split_token='\t').load_data()
testing_ag = rt.rt_data(testing, splitted=False, split_token='\t').load_data()
data = (training_ag, testing_ag)
classes = 4


