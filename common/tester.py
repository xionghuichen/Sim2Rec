#!/usr/bin/env python
# coding=utf-8

# Author      :   Xionghui Chen
# Created     :   2017-11-12
# Modified    :   2017-11-12
# Version     :   1.0


from common import logger
import pickle
import time
import os

import datetime
import baselines.common.tf_util as U
from common.time_step import time_step_holder
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import os.path as osp
import pandas
import seaborn as sns
from baselines.bench import monitor
from collections import defaultdict, namedtuple


sns.set_style('darkgrid', {'legend.frameon':True})


class Tester(object):
    def __init__(self):

        self.__ipaddr = None
        self.hyper_param = {}
        self.session = None
        self.strftims = None
        self.post_file_id = None
        self.file = None
        self.saver = None
        self.save_object = None
        self.last_record_time = None
        self.record_period = None
        self.add_record_to_pkl = True
        self.hyper_param_record = []
        self.time_step_holder = time_step_holder

    def configure(self, prefix_dir, run_file, record_date='', session=None, info=None, save_object=True, record_period=30,
                  add_record_to_pkl=True, run_metadata_mode=False, do_save_checkpoint=True,
                  root='../', max_to_keep=3):
        """
        Tester is a class can store all of record and parameter in your experiment.
        It will be saved as a pickle. you can load it and plot those records to curve picture.

        Parameters
        ----------
        episodes : int
            repeat time per evalate test.

        period : int
            test period


        time_step_holder: common.variable.TimeStepHolder
            glabal time step holder

        file: string
            file to store pickle
        """

        self.__custom_recorder = {}
        self.__custom_data = {}

        self.run_metadata_mode = run_metadata_mode
        self.log_root = root
        self.metadata_list = []
        self.do_save_checkpoint = do_save_checkpoint
        self.session = session
        self.add_record_to_pkl = add_record_to_pkl
        if record_date == '':
            self.record_date = datetime.datetime.now()
        else:
            self.record_date = datetime.datetime.strptime(record_date, '%Y/%m/%d/%H-%M-%S-%f')
            _, file_found = self.load_checkpoint_from_date(record_date, prefix_dir=prefix_dir, log_root=root)
            if file_found != '':
                file_found = file_found.split(' ')
                self.__ipaddr = file_found[1]
                info = " ".join(file_found[2:])

        if info is None:
            info = self.auto_parse_info()
            info = '&' + info
        self.info = info

        _, code_file = self.__create_file_directory(root + "code/" + prefix_dir, '/')
        _, log_file = self.__create_file_directory(root + "log/" + prefix_dir, '.log')
        self.pkl_dir, self.pkl_file = self.__create_file_directory(root + 'result_tester/' + prefix_dir, '.pkl')
        self.checkpoint_dir, _ = self.__create_file_directory(root + 'checkpoint/' + prefix_dir, is_file=False)
        self.results_dir, _ = self.__create_file_directory(root + 'results/' + prefix_dir, is_file=False)
        self.log_file = log_file
        self.code_file = code_file
        logger.info("store file %s" % self.pkl_file)
        logger.configure(log_file, ['stdout', 'log', 'tensorboard', 'csv'])
        self.time_step_holder.set_time(0)

        import tensorflow as tf
        for fmt in logger.Logger.CURRENT.output_formats:
            if isinstance(fmt, logger.TensorBoardOutputFormat):
                self.writer = fmt.writer
        # self.writer = logger. # tf.summary.FileWriter(log_file)
        self.save_object = save_object
        self.last_record_time = time.time()
        self.record_period = record_period
        self.logger = logger
        self.summary_add_dict = {}
        self.max_to_keep = max_to_keep
        self.serialize_object_and_save()
        if record_date == '':
            self.__copy_source_code(code_file)
        # saver shold be initialized after op are constructed.
        # self.saver = tf.train.Saver()

    def task_gen(self, task_pattern_list):
        return '-'.join(task_pattern_list)

    def save_pickle(self, obj, name):
        s = pickle.dumps(obj)
        os.makedirs(self.log_root + 'pickle/', exist_ok=True)
        f = open(self.log_root + 'pickle/' + name, 'wb')
        f.write(s)
        f.close()

    def load_pickle(self, name):
        if os.path.exists(self.log_root + 'pickle/' + name):
            return pickle.loads(open(self.log_root + 'pickle/' + name, 'rb').read())
        else:
            return None

    def print_var_list(self, var_list):
        for v in var_list:
            logger.info(v)

    def initial_with_prefix(self, sess, prefix):
        import tensorflow as tf
        assert sess == tf.get_default_session()
        var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, prefix + '/')
        self.print_var_list(var_list)
        sess.run(tf.initialize_variables(var_list))

    def delete_related_log(self):
        logger.info("log file: {}".format(self.log_file))
        logger.info("code_file: {}".format(self.code_file))
        logger.info("tester_file: {}".format(self.pkl_file))
        logger.info("checkpoint_dir: {}".format(self.checkpoint_dir))
        logger.info("results_dir: {}".format(self.results_dir))


    def print_log_file(self):
        logger.info("log file: {}".format(self.log_file))
        logger.info("pkl_file: {}".format(self.pkl_file))
        logger.info("checkpoint_dir: {}".format(self.checkpoint_dir))
        logger.info("results_dir: {}".format(self.results_dir))

    @classmethod
    def load_tester(cls, record_date, prefix_dir, log_root):
        logger.info("load tester")
        res_dir, res_file = cls.log_file_finder(record_date, prefix_dir=prefix_dir, file_root=log_root + 'result_tester/', log_type='files')
        import pickle
        return pickle.load(open(res_dir+'/'+res_file, 'rb'))
    @classmethod
    def load_checkpoint_from_date(cls, record_date, prefix_dir, log_root, checkpoint_root='checkpoint'):
        return cls.log_file_finder(record_date, prefix_dir=prefix_dir, file_root=log_root + checkpoint_root + '/')


    def add_summary(self, summary, name='', simple_val=False, freq=20):
        if name not in self.summary_add_dict:
            self.summary_add_dict[name] = []
        if freq > 0:
            summary_ts = int(self.time_step_holder.get_time() / freq)
        else:
            summary_ts = 0
        if freq <= 0 or summary_ts not in self.summary_add_dict[name]:
            from tensorflow.core.framework import summary_pb2
            summ = summary_pb2.Summary()
            summ.ParseFromString(summary)
            if simple_val:
                list_field = summ.ListFields()
                def recursion_util(inp_field):
                    if hasattr(inp_field, "__getitem__"):
                        for inp in inp_field:
                            recursion_util(inp)
                    elif hasattr(inp_field, 'simple_value'):
                        logger.record_tabular(name + '/' + inp_field.tag, inp_field.simple_value)
                    else:
                        pass

                recursion_util(list_field)
                logger.dump_tabular()
            else:
                self.writer.add_summary(summary, self.time_step_holder.get_time())
                self.writer.flush()
            self.summary_add_dict[name].append(summary_ts)

    @classmethod
    def log_file_finder(cls, record_date, prefix_dir='train', file_root='../checkpoint/', log_type='dir'):
        record_date = datetime.datetime.strptime(record_date, '%Y/%m/%d/%H-%M-%S-%f')
        prefix = file_root + prefix_dir
        directory = str(record_date.strftime("%Y/%m/%d"))
        directory = prefix + '/' + directory
        file_found = ''
        for root, dirs, files in os.walk(directory):
            if log_type == 'dir':
                search_list = dirs
            elif log_type =='files':
                search_list =files
            else:
                raise NotImplementedError
            for search_item in search_list:
                if search_item.startswith(str(record_date.strftime("%H-%M-%S-%f"))):
                    split_dir = search_item.split(' ')
                    # self.__ipaddr = split_dir[1]
                    info = " ".join(split_dir[2:])
                    logger.info("load data: \n ts {}, \n ip {}, \n info {}".format(split_dir[0], split_dir[1], info))
                    file_found = search_item
                    break
        if file_found == '':
            logger.info("[WARN] not log file found!")
        return directory, file_found

    @property
    def ipaddr(self):
        if self.__ipaddr is None:
            try:
                import socket
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect(("1.1.1.1", 80))
                self.__ipaddr = s.getsockname()[0]
                s.close()
            except Exception as e:
                self.__ipaddr = 'noip'
        return self.__ipaddr

    def __copy_source_code(self, code_file):
        import shutil
        from common.private_config import COPY_DIR_NAME
        # shutil.copytree("../../build/lib/", code_file)
        for dir_name in COPY_DIR_NAME:
            shutil.copytree("../../" + dir_name, code_file +'/' + dir_name)


    def __create_file_directory(self, prefix, ext='', is_file=True):
        directory = str(self.record_date.strftime("%Y/%m/%d"))
        directory = prefix + '/' + directory
        if is_file:
            os.makedirs(directory, exist_ok=True)
            file_name = '{dir}/{timestep} {ip} {info}{ext}'.format(dir=directory,
                                                                 timestep=str(self.record_date.strftime("%H-%M-%S-%f")),
                                                                 ip=str(self.ipaddr),
                                                                 info=self.info,
                                                                 ext=ext)
        else:
            directory = '{dir}/{timestep} {ip} {info}{ext}/'.format(dir=directory,
                                                                 timestep=str(self.record_date.strftime("%H-%M-%S-%f")),
                                                                 ip=str(self.ipaddr),
                                                                 info=self.info,
                                                                 ext=ext)
            os.makedirs(directory, exist_ok=True)
            file_name = ''
        return directory, file_name

    def add_record_param(self, keys):
        for k in keys:
            self.hyper_param_record.append(str(k) + '=' + str(self.hyper_param[k]).replace('[', '{').replace(']', '}'))

    def clear_record_param(self):
        self.hyper_param_record = []

    def time_used_wrap(self, name, func, *args, **kwargs):
        start_time = time.time()
        output = func(*args, **kwargs)
        end_time = time.time()
        self.logger.info("[test] func {0} time used {1:.2f}".format(name, end_time - start_time))
        return output

    def feed_hyper_params_to_tb(self):
        summary = self.dict_to_table_text_summary(self.hyper_param, 'hyperparameters')
        self.add_summary(summary, 'hyperparameters')


    def dict_to_table_text_summary(self, input_dict, name):
        import tensorflow as tf
        with tf.Session() as sess:
            to_tensor = [tf.convert_to_tensor([k, str(v)]) for k, v in input_dict.items()]
            return sess.run(tf.summary.text(name, tf.stack(to_tensor)))

    def print_large_memory_variable(self):
        import sys
        large_mermory_dict = {}
        def sizeof_fmt(num, suffix='B'):
            for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
                if abs(num) < 1024.0:
                    return "%3.1f %s%s" % (num, unit, suffix), unit
                num /= 1024.0
            return "%.1f %s%s" % (num, 'Yi', suffix), 'Yi'

        for name, size in sorted(((name, sys.getsizeof(value)) for name, value in locals().items()),
                                 key=lambda x: -x[1])[:10]:
            size_str, fmt_type = sizeof_fmt(size)
            if fmt_type in ['', 'Ki', 'Mi']:
                continue
            logger.info("{:>30}: {:>8}".format(name, size_str))
            large_mermory_dict[str(name)] = size_str
        if large_mermory_dict != {}:
            summary = self.dict_to_table_text_summary(large_mermory_dict, 'large_memory')
            self.add_summary(summary, 'large_memory')

    def new_saver(self, var_prefix):
        if self.do_save_checkpoint:
            import tensorflow as tf
            var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, var_prefix)
            logger.info("save variable :")
            self.print_var_list(var_list)
            self.saver = tf.train.Saver(var_list=var_list, max_to_keep=self.max_to_keep, filename=self.checkpoint_dir, save_relative_paths=True)

    def load_checkpoint(self, target_prefix_name, current_name, sess, checkpoint_dir=None):
        import tensorflow as tf
        if checkpoint_dir is None:
            checkpoint_dir = self.checkpoint_dir
        logger.info("load checkpoint dir: {}".format(checkpoint_dir))
        ckpt_path = tf.train.latest_checkpoint(checkpoint_dir)
        logger.info("load checkpoint path {}".format(ckpt_path))
        if target_prefix_name is None:
            target_prefix_name = current_name + '/'
        var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, current_name)
        var_dict = {}
        for v in var_list:
            key_name = v.name[v.name.find(current_name) + len(current_name) + 1:]
            # key_name = '/'.join(v.name.split('/')[1:])
            key_name = key_name.split(':')[0]
            var_dict[target_prefix_name + key_name] = v
        logger.info("graph mapping:")
        for k, v in var_dict.items():
            logger.info("{} -> {}".format(k, v.name))
        sess.run(tf.initialize_variables(var_list))
        saver = tf.train.Saver(var_list=var_dict)
        # U.initialize()
        logger.info("ckpt_path [load vae handle] {}".format(ckpt_path))
        saver.restore(sess, ckpt_path)
        print("----------load {} simulator model- successfully---------".format(sess))
        return int(ckpt_path.split('-')[-1])


    def run_with_metadata(self, sess, ops, feed, name):
        if self.run_metadata_mode:
            ts = self.time_step_holder.get_time()
            tag = name + '-' + str(ts % 100)
            if tag in self.metadata_list:
                # logger.warn("[WARN] repeat use runmeta data {}".format(tag))
                rets = sess.run(ops, feed_dict=feed, )
            else:
                import tensorflow as tf
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                rets = sess.run(ops, feed_dict=feed,
                                      options=run_options, run_metadata=run_metadata)
                self.writer.add_run_metadata(run_metadata, tag)
                self.metadata_list.append(tag)
        else:
            # logger.warn("[WARN] if you want to run with metadata, set self.run_metadata_mode = True firstly")
            rets = sess.run(ops, feed_dict=feed,)
        return rets

    def save_checkpoint(self, iters_so_far):
        import tensorflow as tf
        if self.do_save_checkpoint:
            self.saver.save(tf.get_default_session(), self.checkpoint_dir + 'checkpoint', global_step=iters_so_far)

    def auto_parse_info(self):
        return '&'.join(self.hyper_param_record)

    def period_record(self):
        current_time = time.time()
        if (current_time - self.last_record_time) >= self.record_period:
            logger.info("do save. store pkl tester file :%s " % self.pkl_file)
            self.serialize_object_and_save()
            self.last_record_time = current_time

    def end_check(self, satisfied_length, end_point):
        """
        Check if average return value of recent satisfied_length number larger than end_point

        Parameters
        ----------
        satisfied_length : int

        end_point : int

        """
        if end_point == None:
            return False
        else:
            if 'return' not in self.__custom_recorder:
                return False
            length = len(self.__custom_recorder['return'][
                         self.__custom_recorder['return']['name'][1]])
            to_cal_return = self.__custom_recorder['return'][
                self.__custom_recorder['return']['name'][1]][length-satisfied_length:]
            avg_ret = sum(to_cal_return) / (len(to_cal_return) + 1)
            print(
                "-----------------------------------recent return is %s -------------------------" % avg_ret)
            if avg_ret >= end_point:
                return True
            else:
                return False

    def set_hyper_param(self, **argkw):
        """
        This method is to record all of hyper parameters to test object.

        Place pass your parameters as follow format:
            self.set_hyper_param(param_a=a,param_b=b)

        Note: It is invalid to pass a local object to this function.

        Parameters
        ----------
        argkw : key-value
            for example: self.set_hyper_param(param_a=a,param_b=b)

        """
        self.hyper_param = argkw

    def add_custom_record(self, key, y, x=None,  x_name=None, y_name=None):
        """
        This model is to add record to specific 'key'.
        After that, you can load plk file and call 'print' function to print x-y curve.

        Parameters
        ----------
        key : string
            identify your curve

        x: float or int
            x value to be added.

        y: float or int
            y value to be added.

        x_name: string
            name of x axis, will be displayed when call print function

        y_name: string
            name of y axis, will be displayed when call print function
        """
        if x is None:
            x = self.time_step_holder.get_time()
            assert x_name is None
            x_name = 'time-step'
            logger.record_tabular(key, y)
            if not self.add_record_to_pkl:
                return
        if y_name is None:
            y_name = key
        if key not in self.__custom_recorder:
            self.__custom_recorder[key] = {}
            self.__custom_recorder[key][x_name] = [x]
            self.__custom_recorder[key][y_name] = [y]
            self.__custom_recorder[key]['name'] = [x_name, y_name]
        else:
            self.__custom_recorder[key][x_name].append(x)
            self.__custom_recorder[key][y_name].append(y)
        self.period_record()

    def dumpkvs(self):
        self.logger.record_tabular('time-step', self.time_step_holder.get_time())
        self.logger.dumpkvs()

    def add_graph(self, sess):
        assert self.writer is not None
        self.writer.add_graph(sess.graph)
        self.writer.flush()

    def add_custom_data(self, key, data, dtype=None):
        if key not in self.__custom_data:
            if isinstance(dtype, list):
                self.__custom_data[key] = [data]
            else:
                self.__custom_data[key] = data
        else:
            if isinstance(dtype, list):
                self.__custom_data[key].append(data)
            else:
                self.__custom_data[key] = data

    def get_custom_data(self, key):
        if key not in self.__custom_data:
            return None
        else:
            return self.__custom_data[key]

    def get_custom_recorder(self, key):
        if key not in self.__custom_recorder:
            return None
        else:
            return self.__custom_recorder[key]

    def simple_scatter(self, name, datas, texts, pretty=False, xlabel='', ylabel='',
                       cover=False, *args, **kwargs):
        import os.path as osp
        if pretty:
            save_path = osp.join(tester.results_dir, name + '.pdf')
        else:
            save_path = osp.join(tester.results_dir, name + '.png')
        if not osp.exists(save_path) or cover:
            from matplotlib import pyplot as plt
            from matplotlib.ticker import ScalarFormatter
            plt.cla()
            import matplotlib.colors as mcolors
            colors = list(mcolors.TABLEAU_COLORS.keys())  # 颜色变化
            index = 0
            for data, text in zip(datas, texts):
                color = colors[index % len(colors)]
                plt.scatter(data[:, 0], data[:, 1], color=color, marker='x', alpha=0.2)
                #plt.annotate(s=str(text), xy=data.mean(axis=0), color=color)
                plt.annotate(text=str(text), xy=data.mean(axis=0), color=color)
                index += 1
            texts = []
            texts.append(plt.xlabel(xlabel, fontsize=15))
            texts.append(plt.ylabel(ylabel, fontsize=15))
            plt.xticks(fontsize=13)
            plt.yticks(fontsize=13)
            plt.grid()
            ax = plt.gca()  # 获取当前图像的坐标轴信息
            # xfmt = ScalarFormatter(useMathText=True)
            # xfmt.set_powerlimits((-2, 2))  # Or whatever your limits are . . .
            # plt.gca().yaxis.set_major_formatter(xfmt)
            # plt.gcf().subplots_adjust(bottom=0.12, left=0.12)
            # plt.title(name, fontsize=7)
            save_dir = '/'.join(save_path.split('/')[:-1])
            os.makedirs(save_dir, exist_ok=True)
            #
            plt.savefig(save_path, bbox_extra_artists=tuple(texts), bbox_inches='tight')

    def simple_hist(self, name, data, labels=None, pretty=False, xlabel='', ylabel='',
                    colors=None, styles=None, cover=False, *args, **kwargs):
        import os.path as osp
        if pretty:
            save_path = osp.join(tester.results_dir, name + '.pdf')
        else:
            save_path = osp.join(tester.results_dir, name + '.png')
        if not osp.exists(save_path) or cover:
            from matplotlib import pyplot as plt
            from matplotlib.ticker import ScalarFormatter
            plt.cla()
            if pretty:
                # ['r', 'b'], ['x--', '+-']
                if labels is not None:
                    for d, l in zip(data, labels):
                        plt.hist(d, label=l, histtype='step', *args, **kwargs)
                else:
                    plt.hist(data, histtype='step', *args, **kwargs)
            else:
                plt.hist(data, label=labels,  histtype='step', *args, **kwargs)
            # plt.tight_layout()
            if labels is not None:
                plt.legend(prop={'size': 13})

            plt.xlabel(xlabel, fontsize=15)
            plt.ylabel(ylabel, fontsize=15)
            plt.xticks(fontsize=13)
            plt.yticks(fontsize=13)
            plt.grid()
            ax = plt.gca()  # 获取当前图像的坐标轴信息
            # xfmt = ScalarFormatter(useMathText=True)
            # xfmt.set_powerlimits((-2, 2))  # Or whatever your limits are . . .
            # plt.gca().yaxis.set_major_formatter(xfmt)
            # plt.gcf().subplots_adjust(bottom=0.12, left=0.12)
            # plt.title(name, fontsize=7)
            save_dir = '/'.join(save_path.split('/')[:-1])
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(save_path)

    def simple_plot(self, name, data=None, x=None, y=None, labels=None, pretty=False, xlabel='', ylabel='',
                    colors=None, styles=None, cover=False, figsize=(8,6)):
        import os.path as osp
        if pretty:
            save_path = osp.join(tester.results_dir, name + '.pdf')
        else:
            save_path = osp.join(tester.results_dir, name + '.png')

        if not osp.exists(save_path) or cover:
            from matplotlib import pyplot as plt
            plt.cla()
            if pretty:
                plt.figure(figsize=(7,6))
            else:
                plt.figure(figsize=figsize)
            if labels is None:
                if data is not None:
                    for d in data:
                        plt.plot(d)
                elif x is not None:
                    for x_i, y_i in zip(x, y):
                        plt.plot(x_i, y_i)
                else:
                    raise NotImplementedError
            else:
                if pretty:
                    # ['r', 'b'], ['x--', '+-']
                    if data is not None:
                        for d, l, c, s in zip(data, labels, colors, styles):
                            plt.plot(d, s, label=l, color=c)
                    elif x is not None:
                        for x_i, y_i, l, c, s in zip(x, y, labels, colors, styles):
                            plt.plot(x_i, y_i, s, label=l, color=c)
                    else:
                        raise NotImplementedError
                else:
                    if data is not None:
                        for d, l in zip(data, labels):
                            plt.plot(d, label=l)
                    elif x is not None:
                        for x_i, y_i, l_i in zip(x, y, labels):
                            plt.plot(x_i, y_i,label=l_i)
                    else:
                        raise NotImplementedError
            if labels is not None:
            # plt.xlabel('time-step (per day)', fontsize=15)
            # plt.ylabel('normalized FOs', fontsize=15)
                plt.legend(prop={'size': 15})
            if pretty:
                from matplotlib.ticker import ScalarFormatter
                xfmt = ScalarFormatter(useMathText=True)
                xfmt.set_powerlimits((-4, 4))  # Or whatever your limits are . . .
                plt.gca().yaxis.set_major_formatter(xfmt)

                plt.gca().yaxis.get_offset_text().set_fontsize(16)
                plt.gca().xaxis.get_offset_text().set_fontsize(16)
            plt.xlabel(xlabel, fontsize=18)
            plt.ylabel(ylabel, fontsize=18)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            # plt.title(name, fontsize=7)
            plt.grid(True)
            save_dir = '/'.join(save_path.split('/')[:-1])
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')

    def set_axis(self, axis):
        self.axis = axis

    def __print_unit(self, key, style, gen, label):
        import matplotlib.pyplot as plt
        if key in self.__custom_recorder:
            x_name, y_name = self.__custom_recorder[key]['name']
            scale_y = []
            scale_x = []
            gen_y = gen(self.__custom_recorder[key][y_name])
            for x, y in zip(self.__custom_recorder[key][x_name], gen_y):
                scale_y.append(y)
                scale_x.append(x)

            plt.plot(scale_x, scale_y, style, label=label)
            plt.grid(True)
            plt.legend(loc='upper left', shadow=False, prop={'size': 15})
            # plt.xlabel(x_name)
            # plt.ylabel(y_name)
        else:
            logger.info('[Tester plot wrong] key %s not exist. ' % key)

    def multi_print(self, key_list, style_list, gen_list, label_list=None):
        """
        plot several curve store in key_list.

        Parameters
        ----------
        key_list : array of string
            the id list which has been set when call self.add_custom_record function

        type_list: array of string
        """
        from matplotlib import ticker
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-1, 1))
        self.axis.yaxis.set_major_formatter(formatter)
        self.axis.xaxis.set_major_formatter(formatter)
        if label_list is None:
            label_list = key_list
        for key, style, gen, label in zip(key_list, style_list, gen_list, label_list):
            self.__print_unit(key, style, gen, label)

    def serialize_object_and_save(self):
        """
        This method is to save test object to a pickle.
        This method will be call every time you call add_custom_record or other record function like self.check_and_test
        """
        # remove object which can is not serializable
        if self.save_object:
            sess = self.session
            self.session = None
            writer = self.writer
            self.writer = None
            saver =self.saver
            self.saver = None
            self.logger = None
            with open(self.pkl_file, 'wb') as f:
                pickle.dump(self, f)
            self.session = sess
            self.writer = writer
            self.logger = logger
            self.saver = saver

    def init_unserialize_obj(self, sess):
        for fmt in logger.Logger.CURRENT.output_formats:
            if isinstance(fmt, logger.TensorBoardOutputFormat):
                self.writer = fmt.writer
        self.logger = logger
        self.session = sess

    def set_figure(self, fig):
        self.fig = fig

    def show(self):
        self.fig.show()

    def savefig(self, *argv, **argkw):
        self.fig.savefig(*argv, **argkw)

    def print_args(self):
        sort_list = sorted(self.hyper_param.items(), key=lambda i: i[0])
        for key, value in sort_list:
            logger.info("key: %s, value: %s" % (key, value))

    def save_suctom_object(self, class_name, ob):
        with open(self.pkl_file + '-'+class_name, 'wb') as f:
            pickle.dump(ob, f)

tester = Tester()
