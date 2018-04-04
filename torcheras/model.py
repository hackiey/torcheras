import os
import json
import shutil
import torch
from torch.autograd import Variable
from visualdl import LogWriter
import random
from datetime import datetime
from . import metrics

class Model:
    def __init__(self, model, logdir = '', for_train = True):
        self.model = model

        self.if_submodel = False
        self.submodel_names = []
        self.submodels = []
        
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
                'params': {},
                'epochs': [],
                'metrics':[]
            }
    
    def compile(self, loss_function, optimizer, metrics, use_cuda = False):
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.metrics = ['loss'] + metrics
        
        self.use_cuda = use_cuda
        if self.use_cuda:
            self.cuda()
                
        # notes
        self.notes['optimizer'] = self.optimizer.__class__.__name__
        self.notes['metrics'] = self.metrics
        for submodel_name, param_group in zip(self.submodel_names, self.optimizer.param_groups):
            self.notes['params'][submodel_name] = {}
            for k, v in param_group.items():
                if k!= 'params':
                    self.notes['params'][submodel_name][k] = v
                    
    def cuda(self):
        for submodel in self.submodels:
            submodel = submodel.cuda()

    def variableData(self, sample_batched):
        X = sample_batched[0]
        y = sample_batched[1]
        X = Variable(X.cuda() if self.use_cuda else X)
        y = Variable(y.cuda() if self.use_cuda else y)
        return X, y

    def evaluate(self, X, y):
        output = self.model.forward(X)
        loss = self.loss_function(output, y)
        result_metrics = [loss]
        for metric in self.metrics:
            if metric == 'binary_acc':
                result_metrics.append(metrics.binary_accuracy(y, output))
            if metric == 'categorical_acc':
                result_metrics.append(metrics.categorical_accuracy(y, output))
        return result_metrics

    def loadModel(self, sub_folder, epoch=1):
        if self.if_submodel:
            for submodel_name, submodel in zip(self.submodel_names, self.submodels):
                submodel.load_state_dict(torch.load(os.path.join(self.logdir, sub_folder, submodel_name+'_'+str(epoch)+ '.pth')))
        else:
            self.model.load_state_dict(torch.load(os.path.join(self.logdir, sub_folder, 'model_' + str(epoch)+'.pth')))
        
    def predict(self, X):
        return self.model.predict(X)

    def setSubmodels(self, submodels):
        self.if_submodel = True

        for submodel in submodels:
            self.submodel_names.append(submodel[0])
            self.submodels.append(submodel[1])

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

                # save checkpoints
                if self.if_submodel:
                    for submodel_name, submodel in zip(self.submodel_names, self.submodels):
                        torch.save(submodel.state_dict(),os.path.join(self.logdir,submodel_name+'_'+str(epoch + 1)+'.pth'))
                else:
                    torch.save(self.model.state_dict(), os.path.join(self.logdir, 'model_' + str(epoch + 1) + '.pth'))
                
                self.notes['epochs'].append([train_metrics, test_metrics])
            
            # save notes
            self.saveNotes()

        except KeyboardInterrupt:
            print("If early stopped? please input 'y' or 'n'")
            answer = input()
            if answer == 'n':
                shutil.rmtree(self.logdir)
                print(self.logdir + ' Checkpoint deleted')
            elif answer == 'y':
                self.saveNotes()
                print('Notes saved')
                    
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
            