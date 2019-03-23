import os
import glob
import socket
import logging
import argparse
import pdb
import tensorflow as tf
import neuralgym as ng

from pathlib import Path
from shutil import copyfile
from inpaint_model import InpaintCAModel


logger = logging.getLogger()

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_dir', default='/userhome/checkpoints/GntIpt', type=str,
                    help='The directory of tensorflow checkpoint.')
parser.add_argument('--config', default="configs/re_inpaint.yml", type=str,
                    help='config YAML file')

def multigpu_graph_def(model, data, config, gpu_id=0, loss_type='g'):
    with tf.device('/cpu:0'):
        images = data.data_pipeline(config.BATCH_SIZE)
    if gpu_id == 0 and loss_type == 'g':
        _, _, losses = model.build_graph_with_losses(
            images, config, summary=True, reuse=True)
    else:
        _, _, losses = model.build_graph_with_losses(
            images, config, reuse=True)
    if loss_type == 'g':
        return losses['g_loss']
    elif loss_type == 'd':
        return losses['d_loss']
    else:
        raise ValueError('loss type is not supported.')


if __name__ == "__main__":
    args = parser.parse_args()
    checkpoint_dir = Path(args.checkpoint_dir)
    config = ng.Config(args.config)
    if config.GPU_ID != -1:
        ng.set_gpus(config.GPU_ID)
    else:
        ng.get_gpus(config.NUM_GPUS)
    mask = None
    if config.External_mask:
        batch_size = config.BATCH_SIZE
        # load irregualr mask dataset
        with open(config.External_mask) as f:
            fnames = f.read().splitlines()
        filename_dataset = tf.data.Dataset.list_files(fnames)
        img_dataset = filename_dataset.map(lambda x: tf.image.decode_png(tf.read_file(x)))
        img_dataset = img_dataset.map(lambda x: tf.image.resize_images(x, config.IMG_SHAPES[0:2]))
        img_dataset = img_dataset.repeat()
        img_dataset = img_dataset.batch(batch_size)
        img_dataset = img_dataset.prefetch(buffer_size=batch_size*2)
        iter = img_dataset.make_one_shot_iterator()
        mask = iter.get_next()

    # training data
    with open(config.DATA_FLIST[config.DATASET][0]) as f:
        fnames = f.read().splitlines()
    data = ng.data.DataFromFNames(
        fnames, config.IMG_SHAPES, random_crop=config.RANDOM_CROP)
    images = data.data_pipeline(config.BATCH_SIZE)
    # pdb.set_trace()
    # main model
    model = InpaintCAModel()
    g_vars, d_vars, losses = model.build_graph_with_losses(
        images, mask, config=config)
    # validation images
    if config.VAL:
        with open(config.DATA_FLIST[config.DATASET][1]) as f:
            val_fnames = f.read().splitlines()
        # progress monitor by visualizing static images
        for i in range(config.STATIC_VIEW_SIZE):
            static_fnames = val_fnames[i:i+1]
            static_images = ng.data.DataFromFNames(
                static_fnames, config.IMG_SHAPES, nthreads=1,
                random_crop=config.RANDOM_CROP).data_pipeline(1)
            static_inpainted_images = model.build_static_infer_graph(
                static_images, config, name='static_view/%d' % i)
    # training settings
    lr = tf.get_variable(
        'lr', shape=[], trainable=False,
        initializer=tf.constant_initializer(1e-4))
    d_optimizer = tf.train.AdamOptimizer(lr, beta1=0.5, beta2=0.9)
    g_optimizer = d_optimizer
    # gradient processor
    if config.GRADIENT_CLIP:
        gradient_processor = lambda grad_var: (
            tf.clip_by_average_norm(grad_var[0], config.GRADIENT_CLIP_VALUE),
            grad_var[1])
    else:
        gradient_processor = None
    # log dir
    # log_prefix = 'model_logs/' + '_'.join([
    #     ng.date_uid(), socket.gethostname(), config.DATASET,
    #     'MASKED' if config.GAN_WITH_MASK else 'NORMAL',
    #     config.GAN,config.LOG_DIR])
    log_prefix = checkpoint_dir.joinpath(config.DATASET).joinpath(ng.date_uid())
    if not log_prefix.parent.exists():
        log_prefix.parent.mkdir()
    log_prefix.mkdir()

    # copy config file to log folder
    copyfile(args.config, log_prefix.joinpath(Path(args.config).name))
    log_prefix = str(log_prefix)

    # train discriminator with secondary trainer, should initialize before
    # primary trainer.
    discriminator_training_callback = ng.callbacks.SecondaryTrainer(
        pstep=1,
        optimizer=d_optimizer,
        var_list=d_vars,
        max_iters=5,
        graph_def=multigpu_graph_def,
        graph_def_kwargs={
            'model': model, 'data': data, 'config': config, 'loss_type': 'd'},
    )
    # train generator with primary trainer
    # use multi gpu
    # trainer = ng.train.MultiGPUTrainer(
    trainer = ng.train.Trainer(
        num_gpus = config.NUM_GPUS,
        optimizer=g_optimizer,
        var_list=g_vars,
        max_iters=config.MAX_ITERS,
        graph_def=multigpu_graph_def,
        grads_summary=config.GRADS_SUMMARY,
        gradient_processor=gradient_processor,
        graph_def_kwargs={
            'model': model, 'data': data, 'config': config, 'loss_type': 'g'},
        spe=config.TRAIN_SPE,
        log_dir=log_prefix,
    )
    # add all callbacks
    if not config.PRETRAIN_COARSE_NETWORK:
        trainer.add_callbacks(discriminator_training_callback)
    trainer.add_callbacks([
        ng.callbacks.WeightsViewer(),
        ng.callbacks.ModelRestorer(trainer.context['saver'], dump_prefix='model_logs/'+config.MODEL_RESTORE+'/snap', optimistic=True),
        ng.callbacks.ModelSaver(config.TRAIN_SPE, trainer.context['saver'], log_prefix+'/snap'),
        ng.callbacks.SummaryWriter((config.VAL_PSTEPS//1), trainer.context['summary_writer'], tf.summary.merge_all()),
    ])
    # launch training
    trainer.train()
