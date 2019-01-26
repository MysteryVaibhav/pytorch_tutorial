class Evaluator:
    def __init__(self, params, utils):
        self.params = params
        self.utils = utils
        self.log_time = {}
        self.pte = self.utils.get_pre_trained_embeddings()

    def evaluate_lstm(self):
        config = 'lstm_no_pre_trained_emb'
        print('-----------{}-------------'.format(config))
        training_time = self.utils.train(pretrained_emb=None, save_plots_as=config)
        self.log_time[config] = training_time
        print('-----------------------------------------')
        config = 'lstm_use_pre_trained_emb'
        print('-----------{}-------------'.format(config))
        training_time = self.utils.train(pretrained_emb=self.pte, save_plots_as=config)
        self.log_time[config] = training_time
        print('-----------------------------------------')
