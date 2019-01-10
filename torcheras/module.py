
import os
import json
import shutil
import traceback
import torch

from datetime import datetime

from . import utils


class Module(torch.nn.Module):
    def __init__(self):
        super(Module, self).__init__()

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
        self._metrics = metrics or []
        self._multi_tasks = multi_tasks or []

        self._lr_scheduler = lr_scheduler
        self._custom_objects = custom_objects or {}

        total_parameters, trainable_parameters = utils.model.count_parameters(self)
        print('total parameters', total_parameters)
        print('trainable parameters', trainable_parameters)

    def load_model(self, sub_folder, suffix=1, ema=False):
        if ema:
            ema_shadow = torch.load(
                os.path.join(self._logdir, sub_folder, 'ema_'+str(suffix)+'.pth'))
            state_dict = self.state_dict()
            state_dict.update(ema_shadow)
            self.load_state_dict(state_dict)
        else:
            self.load_state_dict(torch.load(
                os.path.join(self._logdir, sub_folder, 'ckpt_'+str(suffix)+'.pth')))

    def fit(self,
            train_dataloader,
            test_dataloader=None,
            epochs=1,
            logdir='',
            sub_folder='',
            save_checkpoint='epoch',
            save_ema='epoch',
            ema_decay=None,
            grad_clip=5.0,
            batch_callback=None,
            epoch_callback=None):

        if logdir != '' and not os.path.exists(logdir):
            os.mkdir(logdir)

        now = datetime.now()
        if sub_folder == '':
            now = datetime.now()
            sub_folder = now.strftime('%Y_%m_%d_%H_%M_%S_%f')
        logdir = os.path.join(logdir, sub_folder)

        if not os.path.exists(logdir):
            os.mkdir(logdir)

        print(logdir)
        
        try:
            if ema_decay:
                ema = utils.train.EMA(ema_decay, self.parameters())

            global_step = 0
            for epoch in range(epochs):
                self.train()
                train_metrics = utils.log.MetricsLog(self._metrics,
                                                     self._loss_fn,
                                                     multi_tasks=self._multi_tasks,
                                                     custom_objects=self._custom_objects)

                for step, batch in enumerate(train_dataloader):
                    self._optimizer.zero_grad()

                    inputs, labels = self._variable_data(batch)
                    preds = self.forward(*inputs)
                    loss, batch_metrics = train_metrics.evaluate(preds, labels)
                    # TODO: gradient accumulation
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
                        batch_callback(self, epoch, global_step, batch_metrics)

                    print_string, print_data = self._make_print(epoch, step, batch_metrics)
                    print(print_string % print_data, end='')
                    print('\r', end='')

                # train metrics
                train_metrics_averaged = train_metrics.average()
                print_string, print_data = self._make_print(epoch, step, train_metrics_averaged)
                print(print_string % print_data)

                # epoch callback
                if epoch_callback:
                    epoch_callback(epoch, train_metrics_averaged)

                # test
                if test_dataloader:
                    self.eval()
                    test_metrics_averaged = self.evaluate(test_dataloader, 
                                                          batch_num=len(test_dataloader),
                                                          loss_fn=self._loss_fn,
                                                          metrics=self._metrics,
                                                          multi_tasks=self._multi_tasks,
                                                          custom_objects=self._custom_objects)

                # save ema
                if ema_decay and type(save_ema) == str:
                    if save_ema == 'epoch':
                        torch.save(ema.shadow, os.path.join(logdir, 'ema_'+str(epoch+1)+'.pth'))

                # save checkpoint
                if type(save_checkpoint) == str:
                    if save_checkpoint == 'epoch':
                        torch.save(self.state_dict(), os.path.join(logdir, 'ckpt_'+str(global_step+1)+'.pth'))

                # save notes
                # TODO: tensorboard
                
        except KeyboardInterrupt:
            try:
                time.sleep(0.5)
                print("Is early stopped? please input 'y' or 'n(delete this model)'")
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

    def evaluate(self, test_dataloader, batch_num, loss_fn=None, metrics=None, multi_tasks=None, custom_objects=None):
        with torch.no_grad():
            test_metrics = utils.log.MetricsLog(metrics, loss_fn, multi_tasks, custom_objects)
            for step, batch in enumerate(test_dataloader):
                inputs, labels = self._variable_data(batch)
                preds = self.forward(*inputs)
                test_metrics.evaluate(preds, labels)
                if test_metrics.steps > batch_num:
                    break

            # test metrics
            test_metrics_averaged = test_metrics.average()
            print_string, print_data = self._make_print('test', step, test_metrics_averaged)
            print(print_string % print_data)

            return test_metrics_averaged

    def _variable_data(self, batch):
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

        return inputs, labels

    def _make_print(self, epoch, step, metric_values):
        if type(epoch) == str:
            print_string = '[%s, %5d] loss: %.5f'
            print_data = (epoch, step + 1, metric_values['loss'])
        else:
            print_string = '[%d, %5d] loss: %.5f'
            print_data = (epoch + 1, step + 1, metric_values['loss'])

        if len(self._multi_tasks) == 0:
            print_string = print_string + ', '
            for metric in self._metrics:
                print_string = print_string + metric + ': %.5f, '
                print_data = print_data + (metric_values[metric],)
            # delete last ', ' from the print string
            print_string = print_string[: -2]
        else:
            for task in self._multi_tasks:
                print_string = print_string + ', ' + task + ': '
                for metric in self._metrics:
                    print_string = print_string + metric + ': %.5f, '
                    print_data = print_data + (metric_values[task][metric],)
                    # delete last ', ' from the print string
                print_string = print_string[: -2]

        return print_string, print_data
    
    def _del_logdir(self, logdir):
        shutil.rmtree(logdir)
        print(logdir + ' model deleted!')
