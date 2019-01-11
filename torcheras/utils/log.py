
from collections import defaultdict
from tensorboardX import SummaryWriter

from .. import metrics

class MetricsLog:
    def __init__(self, metrics, loss_fn, multi_tasks=[], custom_objects={}):

        self.metrics = metrics
        self.multi_tasks = multi_tasks
        self.custom_objects = custom_objects
        self.writer = SummaryWriter()
        
        if len(self.multi_tasks) == 0:
            self.metrics_log = defaultdict(lambda: 0)
        else:
            self.metrics_log = defaultdict(lambda: defaultdict(lambda: 0))
            self.metrics_log['loss'] = 0
     
        self.loss_fn = loss_fn
        self.steps = 0
    
    def evaluate(self, y_pred, y_true, if_metric=True):
        loss = self.loss_fn(y_pred, y_true)

        self.steps += 1
        if not if_metric:
            return loss, None

        if len(self.multi_tasks) == 0:  
            # single task
            result_metrics = {}          
            self.metrics_log['loss'] += loss.item()
            result_metrics['loss'] = loss.item()
            for metric, value in self._make_result_metrics(y_pred, y_true).items():
                self.metrics_log[metric] += value.item()
                result_metrics[metric] = value.item()
        else:
            # multi tasks
            result_metrics = defaultdict(lambda: defaultdict(lambda: 0))
            for i_task, task in enumerate(self.multi_tasks):
                self.metrics_log['loss'] += loss.item()
                result_metrics['loss'] = loss.item()
                # predictions
                if type(y_pred) == list or type(y_pred) is tuple:
                    _y_pred = y_pred[i_task]
                else:
                    _y_pred = y_pred[:, i_task]
                # ground truth  
                if type(y_true) is list or type(y_true) is tuple:
                    _y_true = y_true[i_task]
                else:
                    _y_true = y_true[:, i_task]
                
                for metric, value in self._make_result_metrics(_y_pred, _y_true).items():
                    self.metrics_log[task][metric] += value.item()
                    result_metrics[task][metric] = value.item()
        return loss, result_metrics

    def _make_result_metrics(self, y_pred, y_true):
        metrics_dic = {}     
        # for unified computation
        topk = ()
        
        for metric in self.metrics:
            if metric == 'loss':
                continue
            elif metric == 'binary_acc':
                metrics_dic[metric] = metrics.binary_accuracy(y_true, y_pred)
            elif metric == 'categorical_acc':
                metrics_dic[metric] = metrics.categorical_accuracy(y_true, y_pred)
            elif metric.startswith('top'):
                # get the k in 'top1, top2, ...'
                topk = topk + (int(metric[3:]), ) 
            else:
                # custom metrics
                metrics_dic[metric] = self.custom_objects[metric](y_true, y_pred)
        
        # for unified computation
        if len(topk) != 0:
            topk_metrics = metrics.topk(y_true, y_pred, topk)
            for i, k in enumerate(topk):
                metrics_dic['top' + str(k)] = topk_metrics[i]
            
        return metrics_dic
    
    def average(self, steps=0):
        if steps == 0:
            steps = self.steps
        
        if len(self.multi_tasks) == 0:
            # single task
            average_metrics = {}
            for metric, value in self.metrics_log.items():
                average_metrics[metric] = value / steps
        else:
            # multi tasks
            average_metrics = {'loss': self.metrics_log['loss'] / steps}
            for task in self.multi_tasks:
                average_metrics[task] = {}
                for metric, value in self.metrics_log[task].items():
                    average_metrics[task][metric] = value / steps

        return average_metrics