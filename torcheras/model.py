import os
import json
import shutil
import traceback
import time
from datetime import datetime

import torch
from torch.autograd import Variable
from visualdl import LogWriter
import random

from . import metrics

class Model:
    def __init__(self, model, logdir = '', for_train = True):
        self.model = model
        
        self.logdir = logdir
        
        self.for_train = for_train
        if for_train == True:
            if logdir != '':
                if not os.path.exists(self.logdir):
                    os.mkdir(self.logdir)
                now = datetime.now()
                folder = '_'.join([str(i) for i in [now.year, now.month, now.day, now.hour, now.minute, now.second, now.microsecond]])
                self.logdir = os.path.join(self.logdir, folder)

                if not os.path.exists(self.logdir):
                    os.mkdir(self.logdir)
        
            self.logger = LogWriter(self.logdir, sync_cycle = 50)

            self.notes = {
                'optimizer': '',
                'description':'',
                'params': {},
                'epochs': [],
                'metrics':[]
            }
            
    def setDescription(self, decription):
        self.notes['description'] = description
    
    def compile(self, loss_function, optimizer, metrics = [], use_cuda = False):
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.metrics = ['loss'] + metrics
        
        self.use_cuda = use_cuda
        if self.use_cuda:
            # self.cuda()
            self.model.cuda()
                
        # notes
        self.notes['optimizer'] = self.optimizer.__class__.__name__
        self.notes['metrics'] = self.metrics
        
        # optimizer parameters
        param_groups_num = 0
        for param_group in self.optimizer.param_groups:       
            if 'module_name' in param_group:
                self.notes['params'][param_group['module_name']] = {}
            else:
                continue               
            param_groups_num += 1
            for k, v in param_group.items():
                if k != 'params' and k != 'module_name':
                    self.notes['params'][param_group['module_name']][k] = v
                    
        if param_groups_num == 0 and len(optimizer.param_groups) == 1:
            for k, v in optimizer.param_groups[0].items():
                if k != 'params':
                    self.notes['params'][k] = v

    def loadModel(self, sub_folder, epoch=1):
        self.model.load_state_dict(torch.load(os.path.join(self.logdir, sub_folder, 'model_' + str(epoch)+ '.pth')))
        
    def predict(self, X):
        return self.model.predict(X)
    
    def parameters(self):
        return self.model.parameters()

    def fit(self, train_data, val_data, epochs = 200):
        if not self.for_train:
            raise AttributeError('This model is not for training.')
        try:
            if self.logdir != '':
                train_scalars, test_scalars = self.initLogger()
                
            print(self.logdir)
            print(self.notes['optimizer'])
            print(self.notes['params'])
            
            for epoch in range(epochs):
                # ========== train ==========
                train_metrics = []
                for i_batch, sample_batched in enumerate(train_data):
                    self.optimizer.zero_grad()
                    X,y = self.variableData(sample_batched)
                    result_metrics = self.evaluate(X, y)
                    loss = result_metrics[0]
                    loss.backward()
                    self.optimizer.step()

                    train_metrics = self.step(result_metrics, train_metrics)

                    print_string, print_data = self.makePrint(epoch, i_batch, result_metrics)
                    print(print_string % print_data, end='')
                    print('\r', end='')

                # train metrics
                train_metrics = self.averageMetrics(train_metrics, i_batch)
                print_string, print_data = self.makePrint(epoch, i_batch, train_metrics, if_batch=False)
                print(print_string % print_data) 

                if self.logdir != '':
                    self.loggerAddRecord(epoch+1, train_scalars, train_metrics)
                    
                # ========== test ==========
                test_metrics = []
                for i_batch, sample_batched in enumerate(val_data):
                    X,y = self.variableData(sample_batched)
                    result_metrics = self.evaluate(X, y)
                    test_metrics = self.step(result_metrics, test_metrics)
                
                # test metrics
                test_metrics = self.averageMetrics(test_metrics, i_batch)
                print_string, print_data = self.makePrint(epoch, i_batch, test_metrics, if_batch=False)
                print(print_string % print_data)
                
                if self.logdir != '':
                    self.loggerAddRecord(epoch+1, test_scalars, test_metrics)
                    
                torch.save(self.model.state_dict(), os.path.join(self.logdir, 'model_' + str(epoch + 1) + '.pth'))
                
                self.notes['epochs'].append([train_metrics, test_metrics])
            
            # save notes
            self.saveNotes()

        except KeyboardInterrupt:
            time.sleep(0.2)
            print("Is early stopped? please input 'y' or 'n'")
            answer = input()
            if answer == 'n':
                del self.logger
                shutil.rmtree(self.logdir)
                print(self.logdir + ' Checkpoint deleted')
            elif answer == 'y':
                self.saveNotes()
                print('Notes saved')
                
        except Exception as e:
            traceback.print_exc()
            shutil.rmtree(self.logdir)
            print(self.logdir + ' Checkpoint deleted')
                
    # === private ===
    def variableData(self, sample_batched):
        X = sample_batched[0]
        y = sample_batched[1]
        X = Variable(X.cuda() if self.use_cuda else X)
        y = Variable(y.cuda() if self.use_cuda else y, requires_grad=False)
        return X, y

    def evaluate(self, X, y):
        output = self.model.forward(X)
        loss = self.loss_function(output, y)
        result_metrics = [loss]
        
        metrics_dic = {}
        
        # for unified computation
        topk = ()
        
        for metric in self.metrics:
            if metric == 'binary_acc':
                metrics_dic[metric] = metrics.binary_accuracy(y, output)
                # result_metrics.append(metrics.binary_accuracy(y, output))
            if metric == 'categorical_acc':
                metrics_dic[metric] = metrics.categorical_accuracy(y, output)
                # result_metrics.append(metrics.categorical_accuracy(y, output))
            if metric.startswith('top'):
                topk = topk + (int(metric[3:]), )
        
        # for unified computation
        if len(topk) != 0:
            topk_metrics = metrics.topk(y, output, topk)
            for i, k in enumerate(topk):
                metrics_dic['top' + str(k)] = topk_metrics[i]
                
        # build result_metrics
        for metric in self.metrics[1:]:
            result_metrics.append(metrics_dic[metric])
        
        return result_metrics
                    
    def averageMetrics(self, metrics, i_batch):
        for i in range(len(metrics)):
            metrics[i] = metrics[i] / (i_batch + 1)
        return metrics
        
    def makePrint(self, epoch, i_batch, metric_values, if_batch=True):
        print_string = '[%d, %5d]'
        print_data = (epoch + 1, i_batch + 1)
        
        for metric, value in zip(self.metrics, metric_values):
            print_string  = print_string + ', ' + metric + ': %.5f'
            if if_batch == False:
                print_data = print_data + (value, )
            else:
                print_data = print_data + (value.data[0], )
        return print_string, print_data

    def step(self, result_metrics, metrics):
        if len(metrics) == 0:
            for metric in self.metrics:
                metrics.append(0.0)
                
        for i, (metric, result) in enumerate(zip(self.metrics, result_metrics)):  # self.metrics (metric name)
            metrics[i] = metrics[i] + result.data[0]
            
        return metrics

    def initLogger(self):
        train_scalars = []
        test_scalars = []
        with self.logger.mode('train'):
            for metric in self.metrics:
                scalar = self.logger.scalar('train/'+metric)
                train_scalars.append(scalar)
        with self.logger.mode('test'):
            for metric in self.metrics:
                test_scalars.append(self.logger.scalar("test/" + metric))
        return train_scalars, test_scalars
    
    def loggerAddRecord(self, step, scalars, metrics):
        for scalar, metric in zip(scalars, metrics):
            scalar.add_record(step, metric)
            
    def saveNotes(self):
        with open(self.logdir + '/notes.json',"w") as f:
            json.dump(self.notes, f)
            