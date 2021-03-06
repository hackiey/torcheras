
import os
import json
import shutil
import traceback
import torch
import time

from tqdm import tqdm
from datetime import datetime

from tensorboardX import SummaryWriter

from . import utils


class Module(torch.nn.Module):
    def __init__(self):
        super(Module, self).__init__()
        
        self._metrics = []
        self._multi_tasks = []
        self._custom_objects = {}
        
    def forward(self, *inputs):
        raise NotImplementedError
        
    def compile(self,
                optimizer: torch.optim.Optimizer,
                loss_fn,
                metrics=None,
                multi_tasks=None,
                lr_scheduler=None,
                custom_objects=None):

        self._optimizer = optimizer
        self._loss_fn = loss_fn
        self._metrics = metrics or self._metrics
        self._multi_tasks = multi_tasks or self._multi_tasks
        self._custom_objects = custom_objects or self._custom_objects
        self._lr_scheduler = lr_scheduler
        
        total_parameters, trainable_parameters = utils.model.count_parameters(self)
        print('total parameters', total_parameters)
        print('trainable parameters', trainable_parameters)

    def load_model(self, logdir, sub_folder, suffix=1, ema=False):
        if ema:
            ema_shadow = torch.load(
                os.path.join(logdir, sub_folder, 'ema_'+str(suffix)+'.pth'))
            state_dict = self.state_dict()
            state_dict.update(ema_shadow)
            self.load_state_dict(state_dict)
        else:
            self.load_state_dict(torch.load(
                os.path.join(logdir, sub_folder, 'ckpt_'+str(suffix)+'.pth')))

    def fit(self,
            train_dataloader,
            test_dataloader=None,
            epochs=1,
            logdir='',
            sub_folder='',
            save_checkpoint='epoch',
            save_ema='epoch',
            ema_decay=None,
            fp16=False,
            grad_clip=5.0,
            batch_callback=None,
            epoch_callback=None,
            **kwargs):

        if logdir != '' and not os.path.exists(logdir):
            os.makedirs(logdir, exist_ok=True)

        now = datetime.now()
        if sub_folder == '':
            now = datetime.now()
            sub_folder = now.strftime('%Y_%m_%d_%H_%M_%S_%f')
        logdir = os.path.join(logdir, sub_folder)

        if not os.path.exists(logdir):
            # os.makedirs(logdir)
            os.makedirs(os.path.join(logdir, 'checkpoints'), exist_ok=True)
        print(logdir)

        # tensorboard
        writer = SummaryWriter(logdir=logdir)

        try:    
            if ema_decay:
                ema = utils.train.EMA(ema_decay, self.named_parameters())
            self.train_dataloader = train_dataloader
            self.test_dataloader = test_dataloader
            global_step = 0
            for epoch in range(epochs):
                timer = utils.train.Timer(len(self.train_dataloader)) # Timer
                self.train()
                train_metrics = utils.log.MetricsLog(self._metrics,
                                                     self._loss_fn,
                                                     multi_tasks=self._multi_tasks,
                                                     custom_objects=self._custom_objects)

                for step, batch in enumerate(self.train_dataloader):
                    self._optimizer.zero_grad()

                    inputs, labels = self._variable_data(batch, fp16=fp16)

                    if isinstance(inputs, (list, tuple)):
                        preds = self.forward(*inputs)
                    else:
                        preds = self.forward(inputs)
                    loss, batch_metrics = train_metrics.evaluate(preds, labels)
                    # TODO: gradient accumulation
                    if fp16:
                        try:
                            self._optimizer.backward(loss)
                        except Exception as e:
                            print('You may need install nvidia/apex for fp16 training')
                    else:
                        loss.backward()
                    global_step += 1

                    if grad_clip:
                        torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip)
                        
                    # TODO: lr_scheduler
                    self._optimizer.step()

                    # TODO: data parallel ema
                    if ema_decay:
                        ema(self.named_parameters())

                    # batch callback
                    if batch_callback:
                        batch_callback(self, epoch, global_step, batch_metrics, writer, **kwargs)
                        
                    it_per_second, spent_time, left_time = timer()
                    print_string, print_data = self._make_print(
                        epoch, step, batch_metrics, self._metrics, self._multi_tasks,
                        it_per_second, spent_time, left_time)
                    
                    print(print_string % print_data, end='')
                    print('\r', end='')

                # train metrics
                train_metrics_averaged = train_metrics.average()
                print_string, print_data = self._make_print(
                    epoch, step, train_metrics_averaged, self._metrics, self._multi_tasks,
                    it_per_second, spent_time, left_time)
                print(print_string % print_data)
                self.write_tensorboard(writer, epoch, 'train', train_metrics_averaged)

                # test
                if self.test_dataloader:
                    self.eval()
                    test_metrics_averaged = self.evaluate(self.test_dataloader, 
                                                          batch_num=len(self.test_dataloader),
                                                          loss_fn=self._loss_fn,
                                                          metrics=self._metrics,
                                                          multi_tasks=self._multi_tasks,
                                                          custom_objects=self._custom_objects)
                    self.write_tensorboard(writer, epoch, 'test', test_metrics_averaged)
                    
                # epoch callback
                if epoch_callback:
                    epoch_callback(self, epoch, train_metrics_averaged, writer, **kwargs)

                # save ema
                if ema_decay and type(save_ema) == str:
                    if save_ema == 'epoch':
                        torch.save(ema.shadow, os.path.join(logdir, 'checkpoints', 'ema_'+str(epoch+1)+'.pth'))

                # save checkpoint
                if type(save_checkpoint) == str:
                    if save_checkpoint == 'epoch':
                        torch.save(self.state_dict(), os.path.join(logdir, 'checkpoints', 
                                                                   'ckpt_'+str(global_step+1)+'.pth'))

                # TODO: save params
                
        except KeyboardInterrupt:
            try:
                time.sleep(0.5)
                print("\nIs early stopped? please input 'y' or 'n(delete this model)'")
                answer = input()
                if answer == 'n':
                    self._del_logdir(logdir)
                elif answer == 'y':
                    pass
                    # self._save_notes()
                    print('Training completed!')
            except KeyboardInterrupt:
                self._del_logdir(logdir)
                
        except Exception:
            traceback.print_exc()
            self._del_logdir(logdir)

    def evaluate(self, test_dataloader, batch_num=None, loss_fn=None, metrics=None, multi_tasks=None, custom_objects={}):
        batch_num = batch_num or len(test_dataloader) 
        
        metrics = metrics or self._metrics
        multi_takss = multi_tasks or self._multi_tasks
        custom_objects = custom_objects or self._custom_objects
        
        if_metric = len(metrics) > 0 or len(multi_tasks) > 0
        
        with torch.no_grad():
            test_metrics = utils.log.MetricsLog(metrics, loss_fn, multi_tasks, custom_objects)
            for step, batch in enumerate(test_dataloader):
                inputs, labels = self._variable_data(batch)
                if isinstance(inputs, (list, tuple)):
                    preds = self.forward(*inputs)
                else:
                    preds = self.forward(inputs)
                test_metrics.evaluate(preds, labels, if_metric)
                if test_metrics.steps > batch_num:
                    break

            # test metrics
            test_metrics_averaged = test_metrics.average()
            print_string, print_data = self._make_print('test', step, test_metrics_averaged, metrics, multi_tasks)
            print(print_string % print_data)

            return test_metrics_averaged

    def _variable_data(self, batch, fp16=False):
        device = next(self.parameters()).device
        inputs, labels = batch
        if type(inputs) is list or type(inputs) is tuple:
            for i, _inputs in enumerate(inputs):
                inputs[i] = inputs[i].to(device)
        else:
            inputs = inputs.to(device)
        if type(labels) is list or type(labels) is tuple:
            for i, _labels in enumerate(labels):
                labels[i] = labels[i].to(device)
        else:
            labels = labels.to(device)
            
        if fp16:
            return inputs.half(), labels
        return inputs, labels

    def _make_print(self, epoch, step, metric_values, metrics, multi_tasks, 
                    it_per_second=None, spent_time=None, left_time=None):
        if len(metric_values) == 0:
            return 'no metrics', []
        if type(epoch) == str:
            print_string = '[%s,%5d] loss: %.3f'
            print_data = (epoch, step + 1, metric_values['loss'])
        else:
            print_string = '[%d,%5d, %s<%s, %s] loss: %.3f'
            print_data = (epoch + 1, step + 1, spent_time, left_time, it_per_second, metric_values['loss'])

        if len(multi_tasks) == 0:
            print_string = print_string + ', '
            for metric in metrics:
                print_string = print_string + metric + ': %.3f, '
                print_data = print_data + (metric_values[metric],)
            # delete last ', ' from the print string
            print_string = print_string[: -2]
        else:
            for task in multi_tasks:
                print_string = print_string + ', ' + task + ': '
                for metric in metrics:
                    print_string = print_string + metric + ': %.3f, '
                    print_data = print_data + (metric_values[task][metric],)
                # delete last ', ' from the print string
                print_string = print_string[: -2]

        return print_string, print_data
    
    def write_tensorboard(self, writer, step, mode, metrics_result):
        for k1, v1 in metrics_result.items():
            if type(v1) is dict:
                for k2, v2 in v1.items():
                    writer.add_scalar(f'{mode}/{k1}/{k2}', v2, step)
            else:
                writer.add_scalar(f'{mode}/{k1}', v1, step)
                    
    def _del_logdir(self, logdir):
        shutil.rmtree(logdir)
        print(logdir + ' model deleted!')
