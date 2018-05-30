import os
import json
import shutil
import traceback
import time
from datetime import datetime

import torch
# from torch.autograd import Variable
# from visualdl import LogWriter
    
import random

from . import metrics

class Model:
    def __init__(self, model, logdir, for_train = True):
        self.model = model
        
        self.logdir = logdir
        self.for_train = for_train
        
        self.notes = {
            'optimizer': '',
            'description':'',
            'params': {},
            'epochs': [],
            'metrics':[]
        }
            
        self._custom_objects = {}
            
    def set_description(self, description):
        self.notes['description'] = description
    
    def compile(self, loss_function, optimizer, metrics = [], multi_tasks = [''], custom_objects = {}, device = torch.device('cpu')):
        self.loss_function = loss_function
        
        self.optimizer = optimizer
        self.metrics = ['loss'] + metrics
        self.multi_tasks = multi_tasks
        self.custom_objects = custom_objects
        self.device = device
        
        # notes
        self.notes['optimizer'] = self.optimizer.__class__.__name__
        self.notes['metrics'] = self.metrics
        self.notes['multi_tasks'] = self.multi_tasks
        
        # send model to the device
        self.model.to(self.device)
        
        # optimizer parameters
        param_groups_num = 0
        param_groups = self.optimizer.param_groups if 'param_groups' in dir(self.optimizer) else self.optimizer.optimizer.param_groups
        for param_group in param_groups:
            if 'module_name' in param_group:
                self.notes['params'][param_group['module_name']] = {}
            else:
                continue               
            param_groups_num += 1
            for k, v in param_group.items():
                if k != 'params' and k != 'module_name':
                    self.notes['params'][param_group['module_name']][k] = v
                    
        if param_groups_num == 0 and len(param_groups) == 1:
            for k, v in param_groups[0].items():
                if k != 'params':
                    self.notes['params'][k] = v

    def load_model(self, sub_folder, epoch=1):
        self.model.load_state_dict(torch.load(os.path.join(self.logdir, sub_folder, 'model_' + str(epoch)+ '.pth')))
        
    def predict(self, X):
        return self.model.predict(X)
    
    def parameters(self):
        return self.model.parameters()

    def fit(self, train_data, val_data, epochs = 200):
        if not self.for_train:
            raise AttributeError('This model is not for training.')
        try:
            if not os.path.exists(self.logdir):
                os.mkdir(self.logdir)
            now = datetime.now()
            folder = '_'.join([str(i) for i in [now.year, now.month, now.day, now.hour, now.minute, now.second, now.microsecond]])
            self.logdir = os.path.join(self.logdir, folder)

            if not os.path.exists(self.logdir):
                os.mkdir(self.logdir)
        
            # self.logger = LogWriter(self.logdir, sync_cycle = 50)

            # if self.logdir != '':
            #     train_scalars, test_scalars = self._init_logger()
                
            print(self.logdir)
            print(self.notes['optimizer'])
            print(self.notes['params'])
            
            for epoch in range(epochs):
                # ========== train ==========
                self.model.train()
                train_metrics = []
                for i_batch, sample_batched in enumerate(train_data):
                    self.model.zero_grad()
                    x,y = self._variable_data(sample_batched)
                    
                    result_metrics = self._evaluate(x, y)
                    
                    loss = result_metrics[0]
                    loss.backward()
                    self.optimizer.step()

                    train_metrics = self._add_metrics(result_metrics, train_metrics)

                    print_string, print_data = self._make_print(epoch, i_batch, result_metrics)
                    print(print_string % print_data, end='')
                    print('\r', end='')

                # train metrics
                train_metrics = self._average_metrics(train_metrics, i_batch)
                print_string, print_data = self._make_print(epoch, i_batch, train_metrics)
                print(print_string % print_data) 

                # self._logger_add_record(epoch+1, train_scalars, train_metrics)
                    
                # ========== test ==========
                with torch.no_grad():
                    test_metrics = []
                    for i_batch, sample_batched in enumerate(val_data):
                        X,y = self._variable_data(sample_batched)
                        result_metrics = self._evaluate(X, y)
                        test_metrics = self._add_metrics(result_metrics, test_metrics)
                
                    # test metrics
                    test_metrics = self._average_metrics(test_metrics, i_batch)
                    print_string, print_data = self._make_print(epoch, i_batch, test_metrics)
                    print(print_string % print_data)

                # self._logger_add_record(epoch+1, test_scalars, test_metrics)
                    
                torch.save(self.model.state_dict(), os.path.join(self.logdir, 'model_' + str(epoch + 1) + '.pth'))
                
                self.notes['epochs'].append([train_metrics, test_metrics])
            
            # save notes
            self._save_notes()

        except KeyboardInterrupt:
            try:
                time.sleep(0.5)
                print("Is early stopped? please input 'y' or 'n'")
                answer = input()
                if answer == 'n':
                    self._del_logdir()
                elif answer == 'y':
                    self._save_notes()
                    print('Notes saved')
            except KeyboardInterrupt:
                self._del_logdir()
        except Exception:
            traceback.print_exc()
            self._del_logdir()
                
    # === private ===
    def _variable_data(self, sample_batched):
        x = sample_batched[0]
        y = sample_batched[1]
        if type(x) is list or type(x) is tuple:
            for i, _x in enumerate(x):
                x[i] = x[i].to(self.device)
        else:
            x = x.to(self.device)
        if type(y) is list or type(y) is tuple:
            for i, _y in enumerate(y):
                y[i] = y[i].to(self.device)
        else:
            y = y.to(self.device)
        
        return x, y

    def _evaluate(self, X, y):
        output = self.model.forward(X)
        loss = self.loss_function(output, y)
        result_metrics = [loss]     
        
        if self.multi_tasks[0] == '':
            # single task
            result_metrics.extend(self._make_result_metrics(output, y))
        else:
            # multi tasks
            multi_tasks_length = len(self.multi_tasks)
            
            # for output (predictions)
            multi_outputs = []
            
            if type(output) is tuple or type(output) is list:
                for i in range(multi_tasks_length):
                    multi_outputs.append(output[i])
            else:
                for i in range(multi_tasks_length):
                    multi_outputs.append(output[:,i])
            
            # for y (true labels)
            multi_y = []
            if type(y) is tuple or type(y) is list:
                for i in range(multi_tasks_length):
                    multi_y.append(y[i])
            else:
                for i in range(multi_tasks_length):
                    multi_y.append(y[:,i])                    

            for _output, _y in zip(multi_outputs, multi_y):
                result_metrics.extend(self._make_result_metrics(_output, _y))
                        
        return result_metrics
    
    def _add_metrics(self, result_metrics, metrics):
        if len(metrics) == 0:
            for metric in result_metrics:
                metrics.append(0.0)
                
        for i, result in enumerate(result_metrics):  
            metrics[i] = metrics[i] + result.item()
            
        return metrics   

    def _average_metrics(self, metrics, i_batch):
        for i in range(len(metrics)):
            metrics[i] = metrics[i] / (i_batch + 1)
        return metrics   
    
    def _make_result_metrics(self, output, y):
        result_metrics = []
        
        metrics_dic = {}     
        # for unified computation
        topk = ()
        
        for metric in self.metrics:
            if metric == 'loss':
                continue
            elif metric == 'binary_acc':
                metrics_dic[metric] = metrics.binary_accuracy(y, output)
            elif metric == 'categorical_acc':
                metrics_dic[metric] = metrics.categorical_accuracy(y, output)
            elif metric.startswith('top'):
                # get the k in 'top1, top2, ...'
                topk = topk + (int(metric[3:]), ) 
            else:
                # custom metrics
                metrics_dic[metric] = self.custom_objects[metric](y, output)
        
        # for unified computation
        if len(topk) != 0:
            topk_metrics = metrics.topk(y, output, topk)
            for i, k in enumerate(topk):
                metrics_dic['top' + str(k)] = topk_metrics[i]
                
        # build result_metrics
        for metric in self.metrics[1:]:
            result_metrics.append(metrics_dic[metric])
            
        return result_metrics
        
    def _make_print(self, epoch, i_batch, metric_values):
        print_string = '[%d, %5d] loss: %.5f'
        print_data = (epoch + 1, i_batch + 1, metric_values[0])
            
        metric_values_index = 1
        for output in self.multi_tasks:      
            if output != '':
                print_string = print_string + ', ' + output + ': '
            else:
                print_string = print_string + ', '
                
            for metric, value in zip(self.metrics[1:], metric_values[metric_values_index:]):
                print_string  = print_string + metric + ': %.5f, '
                print_data = print_data + (value, )
                    
                metric_values_index += 1
                
            # delete last ', ' from the print string
            print_string = print_string[: -2]

        return print_string, print_data
    
    def _init_logger(self):
        train_scalars = []
        test_scalars = []
        
        with self.logger.mode('train'):
            if 'outputs' in self.multi_tasks:
                train_scalars.append(self.logger.scalar('train/loss'))
                for output in self.multi_tasks['outputs']:
                    for metric in self.metrics[1:]:
                        train_scalars.append(self.logger.scalar('train/' + output +'/' + metric))
            else:
                for metric in self.metrics:
                    train_scalars.append(self.logger.scalar('train/' + metric))
                    
        with self.logger.mode('test'):
            if 'outputs' in self.multi_tasks:
                test_scalars.append(self.logger.scalar('test/loss'))
                for output in self.multi_tasks['outputs']:
                    for metric in self.metrics[1:]:
                        test_scalars.append(self.logger.scalar("test/" + output + '/' + metric))
            else:
                for metric in self.metrics:
                    test_scalars.append(self.logger.scalar("test/" + metric))
                    
        return train_scalars, test_scalars
    
    def _logger_add_record(self, step, scalars, metrics):
        for scalar, metric in zip(scalars, metrics):
            scalar.add_record(step, metric)
            
    def _save_notes(self):
        with open(self.logdir + '/notes.json',"w") as f:
            json.dump(self.notes, f)
    
    def _del_logdir(self):
        # del self.logger
        shutil.rmtree(self.logdir)
        print(self.logdir + ' Checkpoint deleted')
            