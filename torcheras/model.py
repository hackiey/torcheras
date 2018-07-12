import os
import json
import shutil
import traceback
import time
from datetime import datetime

import torch
from torch import Tensor
    
import random

from . import utils
from .utils.log import MetricsLog

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
    
    def compile(self, loss_fn, optimizer, metrics = [], multi_tasks = [], custom_objects = {}, device = torch.device('cpu')):
        self.loss_fn = loss_fn
        
        self.optimizer = optimizer
        self.metrics = metrics
        self.multi_tasks = multi_tasks
        self.custom_objects = custom_objects
        self.device = device
        
        # notes
        self.notes['optimizer'] = self.optimizer.__class__.__name__ \
                if 'param_groups' in dir(self.optimizer) else self.optimizer.optimizer.__class__.__name__
        self.notes['metrics'] = self.metrics
        self.notes['multi_tasks'] = self.multi_tasks
        
        # send model to the device
        self.model.to(self.device)
        
        # optimizer parameters
        param_groups_num = 0
       
        if 'param_groups' in dir(self.optimizer):
            self.optimizer = self.optimizer
            self.scheduler = None
        else:
            self.scheduler = self.optimizer
            self.optimizer = self.scheduler.optimizer
            
        param_groups = self.optimizer.param_groups

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

        # count parameters
        utils.model.count_parameters(self.model)

    def load_model(self, sub_folder, epoch=1, ema=False):
        if ema:
            ema_shadow = torch.load(os.path.join(self.logdir, sub_folder, 'ema_'+str(epoch)+'.pth'))
            state_dict = self.model.state_dict()
            state_dict.update(ema_shadow)
            self.model.load_state_dict(state_dict)
        else:
            self.model.load_state_dict(torch.load(os.path.join(self.logdir, sub_folder, 'model_' + str(epoch)+ '.pth')))
        
    def predict(self, X):
        return self.model.predict(X)
    
    def parameters(self):
        return self.model.parameters()

    def fit(self, train_data, test_data=None, 
            epochs = 200, 
            ema_decay=None,
            grad_clip=None, 
            save_state=True,
            batch_callback=None, 
            epoch_callback=None):        

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
                
            print(self.logdir)
            print(self.notes['optimizer'])
            print(self.notes['params'])

            # expenential moving average
            if ema_decay:
                ema = utils.train.EMA(ema_decay, self.model.named_parameters())
            
            for epoch in range(epochs):
                # ========== train ==========
                self.model.train()
                train_metrics = MetricsLog(self.metrics, self.loss_fn, 
                                            multi_tasks=self.multi_tasks, custom_objects=self.custom_objects)
                for i_batch, sample_batched in enumerate(train_data):
                    self.model.zero_grad()
                    x, y_true = self._variable_data(sample_batched)
                    y_pred = self.model.forward(x)
                    loss, batch_metrics = train_metrics.evaluate(y_pred, y_true)
                    loss.backward()

                    # grad clip
                    if grad_clip:
                        torch.nn.utils.clip_grad_norm(self.model.parameters(), grad_clip)

                    # scheduler optimizer
                    if self.scheduler:
                        self.scheduler.step()
                    self.optimizer.step()
                    
                    if ema_decay:
                        ema(self.model.named_parameters())

                    # batch callback
                    if batch_callback:
                        batch_callback(epoch, i_batch, batch_metrics)

                    print_string, print_data = self._make_print(epoch, i_batch, batch_metrics)
                    print(print_string % print_data, end='')
                    print('\r', end='')

                # train metrics
                train_metrics_averaged = train_metrics.average()
                print_string, print_data = self._make_print(epoch, i_batch, train_metrics_averaged)
                print(print_string % print_data) 

                # epoch callback
                if epoch_callback:
                    epoch_callback(epoch, train_metrics_averaged)
                
                # ========== test ==========
                if test_data:
                    self.model.eval()
                    test_metrics_averaged = self.evaluate(test_data, batch_num=len(test_data))

                # ema                
                if ema_decay:
                    torch.save(ema.shadow, os.path.join(self.logdir, 'ema_'+str(epoch+1)+'.pth'))

                # save model
                if save_state:
                    torch.save(self.model.state_dict(), os.path.join(self.logdir, 'model_' + str(epoch + 1) + '.pth'))
                else:
                    torch.save(self.model, os.path.join(self.logdir, 'model_' + str(epoch + 1) + '.pt'))
                            
                # save notes
                if test_data:
                    self.notes['epochs'].append([json.dumps(train_metrics_averaged), json.dumps(test_metrics_averaged)])
                else:
                    self.notes['epochs'].append([json.dumps(train_metrics_averaged)])
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

    def evaluate(self, test_data, batch_num = 100):
        with torch.no_grad():
            test_metrics = MetricsLog(self.metrics, self.loss_fn, 
                                        multi_tasks=self.multi_tasks, custom_objects=self._custom_objects)
            for i_batch, sample_batched in enumerate(test_data):
                x, y_true = self._variable_data(sample_batched)
                y_pred = self.model.forward(x)
                test_metrics.evaluate(y_pred, y_true)
                if test_metrics.steps > batch_num:
                    break    

            # test metrics
            test_metrics_averaged = test_metrics.average()
            print_string, print_data = self._make_print('test', i_batch, test_metrics_averaged)
            print(print_string % print_data)
            
            return test_metrics_averaged
                
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

    def _make_print(self, epoch, i_batch, metric_values):
        if type(epoch) == str:
            print_string = '[%s, %5d] loss: %.5f'
            print_data = (epoch, i_batch + 1, metric_values['loss'])
        else:
            print_string = '[%d, %5d] loss: %.5f'
            print_data = (epoch + 1, i_batch + 1, metric_values['loss'])
        
        if len(self.multi_tasks) == 0:
            print_string = print_string + ', '
            for metric in self.metrics:
                print_string = print_string + metric + ': %.5f, '
                print_data = print_data + (metric_values[metric], )
            # delete last ', ' from the print string
            print_string = print_string[: -2]
        else:
            for task in self.multi_tasks:
                print_string = print_string + ', ' + task + ': '
                for metric in self.metrics:
                    print_string = print_string + metric + ': %.5f, '
                    print_data = print_data + (metric_values[task][metric], )              
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
