# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------
 
from __future__ import division

import argparse
import copy
import mmcv
import os
import time
import torch
import warnings
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from os import path as osp

from mmdet import __version__ as mmdet_version
from mmdet3d import __version__ as mmdet3d_version
#from mmdet3d.apis import train_model

from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import collect_env, get_root_logger
from mmdet.apis import set_random_seed
from mmseg import __version__ as mmseg_version

from mmcv.utils import TORCH_VERSION, digit_version

import cv2
cv2.setNumThreads(1)

import sys
sys.path.append('')


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')                # 读取配置文件
    parser.add_argument('--work-dir', help='the dir to save logs and models')   # 指定保存日志和模型的目录
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')             # 从指定检查点恢复训练
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')          # 布尔标志：训练期间不进行验证，使用：--no-validate（不需要值）
    group_gpus = parser.add_mutually_exclusive_group()                          # 互斥组（下面两个二选一使用）
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')                        # 指定GPU数量
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')                        # 指定具体GPU ID
    parser.add_argument('--seed', type=int, default=0, help='random seed')      # 设置随机种子以保证可复现性，默认值：0
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')         # 启用CUDA确定性算法（影响性能但可复现）
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file (deprecate), '
        'change to --cfg-options instead.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')                                                          # 新版配置覆盖参数，支持复杂数据结构
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')                                                    # 分布式训练启动器
    parser.add_argument('--local_rank', type=int, default=0)                    # 分布式训练中的本地进程排名,通常由启动器自动设置，用户很少需要手动指定
    parser.add_argument(
        '--autoscale-lr',
        action='store_true',
        help='automatically scale lr with the number of gpus')                  # 根据GPU数量自动缩放学习率,线性缩放规则：lr_new = lr_base * num_gpus
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both specified, '
            '--options is deprecated in favor of --cfg-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options')
        args.cfg_options = args.options

    return args

#算法入口
def main(): 
    args = parse_args()                                             # 1. 解析命令行参数

    cfg = Config.fromfile(args.config)                              # 2. 加载配置文件
    if args.cfg_options is not None:                                
        cfg.merge_from_dict(args.cfg_options)                       # 3. 应用命令行覆盖
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    # import modules from plguin/xx, registry will be updated       从插件目录导入模块，注册表将更新
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file     导入目录为配置文件所在目录
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)

            # from projects.mmdet3d_plugin.bevformer.apis import custom_train_model
            from projects.mmdet3d_plugin.VAD.apis.train import custom_train_model
    # set cudnn_benchmark  设置cudnn_benchmark，以加速卷积运算（但可能会增加显存消耗）
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename 
    # 工作目录的确定优先级：命令行参数 > 配置文件中的设置 > 默认（基于配置文件名）
    if args.work_dir is not None:                                   # 4. 设置工作目录
        # update configs according to CLI args if args.work_dir is not None
        # 如果命令行指定了工作目录，则更新配置
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    # if args.resume_from is not None:
    # 设置恢复训练的检查点路径（如果提供了有效的文件）
    if args.resume_from is not None and osp.isfile(args.resume_from):
        cfg.resume_from = args.resume_from
    # 设置GPU ID（如果指定了具体的GPU ID）
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        # 否则根据GPU数量设置，默认使用一个GPU（如果没有指定gpus参数，则gpus为None，range(1)即[0]）
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)
    # 针对PyTorch 1.8.1版本和AdamW优化器的bug进行修复
    if digit_version(TORCH_VERSION) == digit_version('1.8.1') and cfg.optimizer['type'] == 'AdamW':
        cfg.optimizer['type'] = 'AdamW2' # fix bug in Adamw
    if args.autoscale_lr:
        # apply the linear scaling rule (https://arxiv.org/abs/1706.02677)
        # 如果启用了自动学习率缩放，则根据GPU数量线性缩放学习率
        cfg.optimizer['lr'] = cfg.optimizer['lr'] * len(cfg.gpu_ids) / 8

    # init distributed env first, since logger depends on the dist info.
    # 首先初始化分布式环境，因为日志记录器依赖于分布式信息
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        # re-set gpu_ids with distributed training mode
        # 在分布式训练模式下重新设置GPU ID
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # create work_dir    创建工作目录
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config   保存配置文件到工作目录
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps    初始化日志记录器（在其他步骤之前）
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    # specify logger name, if we still use 'mmdet', the output info will be
    # filtered and won't be saved in the log_file
    # TODO: ugly workaround to judge whether we are training det or seg model
    if cfg.model.type in ['EncoderDecoder3D']:
        logger_name = 'mmseg'
    else:
        logger_name = 'mmdet'
    logger = get_root_logger(
        log_file=log_file, log_level=cfg.log_level, name=logger_name)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged   初始化元数据字典，用于记录一些重要信息，如环境信息和随机种子，这些信息将被记录到日志中
    meta = dict()
    # log env info  记录环境信息
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text

    # log some basic info   记录一些基本信息
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds  设置随机种子
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)                # 设置随机种子
    cfg.seed = args.seed
    meta['seed'] = args.seed
    meta['exp_name'] = osp.basename(args.config)

    model = build_model(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))                                               # 创建模型
    model.init_weights()

    logger.info(f'Model:\n{model}')
    # 构建数据集
    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:                        # 如果工作流程包括训练和验证
        val_dataset = copy.deepcopy(cfg.data.val)
        # in case we use a dataset wrapper
        if 'dataset' in cfg.data.train:               # 如果训练配置中使用了数据集包装器，则使用相同的预处理流程
            val_dataset.pipeline = cfg.data.train.dataset.pipeline
        else:
            val_dataset.pipeline = cfg.data.train.pipeline
        # set test_mode=False here in deep copied config    在深度拷贝的配置中设置test_mode=False，这不会影响后续的AP/AR计算
        # which do not affect AP/AR calculation later
        # refer to https://mmdetection3d.readthedocs.io/en/latest/tutorials/customize_runtime.html#customize-workflow  # noqa
        val_dataset.test_mode = False
        datasets.append(build_dataset(val_dataset))
    # 配置检查点保存
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data  在检查点中保存mmdet版本、配置文件和类别名称作为元数据
        cfg.checkpoint_config.meta = dict(
            mmdet_version=mmdet_version,
            mmseg_version=mmseg_version,
            mmdet3d_version=mmdet3d_version,
            config=cfg.pretty_text,
            CLASSES=datasets[0].CLASSES,
            PALETTE=datasets[0].PALETTE  # for segmentors
            if hasattr(datasets[0], 'PALETTE') else None)
    # add an attribute for visualization convenience 为可视化方便，将类别信息添加到模型中
    model.CLASSES = datasets[0].CLASSES
    # 使用自定义训练函数开始训练
    custom_train_model(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta)                                                                     # 开始训练


if __name__ == '__main__':
    main()
