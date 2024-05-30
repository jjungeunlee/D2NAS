import os
import sys
import time
import glob
import numpy as np
import logging
import argparse
import torch
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchinfo
import copy

import train_utils
from model_search import Network
from genotypes import Genotype, PRIMITIVES
from task_configs import get_data, get_config, get_metric
from task_utils import calculate_stats

def main():
    parser = argparse.ArgumentParser("cifar")
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset name')
    parser.add_argument('--learning_rate_min', type=float, default=0.0, help='min learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
    parser.add_argument('--epochs', type=int, default=25, help='num of training epochs')
    parser.add_argument('--save', type=str, default='/tmp/checkpoints/', help='experiment path')
    parser.add_argument('--seed', type=int, default=2, help='random seed')
    parser.add_argument('--tmp_data_dir', type=str, default='/tmp/cache/', help='temp data dir')
    parser.add_argument('--note', type=str, default='try', help='note for this run')
    parser.add_argument('--common', action='store_true', default=False, help='run common hyperparam')
    parser.add_argument('--no_es', action='store_true', default=False, help='early stop mode off')

    args = parser.parse_args()

    args.save = '{}_search_{}_{}_{}'.format(args.save, args.note, time.strftime("%Y%m%d_%H%M%S"), args.dataset)
    train_utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.save, 'log.log'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    #  prepare dataset
    dims, sample_shape, num_classes, batch_size, epochs, loss, lr, arch_lr, weight_decay, _, _, \
    _, _, accum, clip, _, _, _, _, _, _, _, _, _, _, config_kwargs = get_config(args.dataset, False, args.common)

    batch_size *= accum

    logging.info("batch size: %d lr: %s, arch_lr: %s", batch_size, lr, arch_lr)
    logging.info("clip: %d", clip)
    logging.info("arch configs: %s", config_kwargs)

    if not torch.cuda.is_available():
        logging.info('No GPU device available')
        sys.exit(1)
    np.random.seed(args.seed)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled=True
    torch.cuda.manual_seed(args.seed)
    logging.info("args = %s", args)

    train_loader, val_loader, _, n_train, n_val, _, data_kwargs = get_data(args.dataset, batch_size, arch='', valid_split=False, for_cell_search=True)
    metric, _ = get_metric(args.dataset)

    cs_args = config_kwargs['cs_args']
    train_queue = train_loader
    valid_queue = val_loader

    # build Network
    try:
        criterion = loss.cuda()
    except:
        criterion = loss
    switches_normal = []
    switches_reduce = []
    step = 4 if dims == 2 else 3
    num_edges = sum(n for n in range(2, 2+step))
    if config_kwargs['pool_k'] == 0:
        PRIMITIVES.remove('avg_pool_3x3')
    for i in range(num_edges):
        switches_normal.append([True for j in range(len(PRIMITIVES))])
    switches_reduce = copy.deepcopy(switches_normal)

    # model params
    widths = cs_args['widths']
    layers = [cs_args['init_layers'] + n for n in cs_args['add_layers']]
    drop_rate = cs_args['dropout_rate']
    sc_limit = cs_args['sc_limit']

    # train params
    decoder = data_kwargs['decoder'] if data_kwargs is not None and 'decoder' in data_kwargs else None 
    transform = data_kwargs['transform'] if data_kwargs is not None and 'transform' in data_kwargs else None 
    if decoder is not None:
        decoder.cuda()
    num_to_keep = cs_args['num_to_keep']
    eps_no_archs = [10, 10, 10]
    check_acc = num_classes > 1

    logging.info("widths: %s, layers: %s, drop_rate: %s", widths, layers, drop_rate)
    logging.info("num train batch: %d num validation batch: %d", n_train, n_val)

    for sp in range(len(num_to_keep)):
        model = Network(sample_shape[1], widths[sp], num_classes, layers[sp], criterion, config_kwargs=config_kwargs, switches_normal=switches_normal, switches_reduce=switches_reduce, step=step, p=float(drop_rate[sp]), dims=dims)
        logging.info("\n%s", torchinfo.summary(model, (batch_size, *sample_shape[1:]), depth=6))
        model = nn.DataParallel(model)
        model = model.cuda()
        logging.info("param size = %fMB", train_utils.count_parameters_in_MB(model))
        optimizer = torch.optim.SGD(
                model.module.network_parameters(),
                lr=lr,
                momentum=args.momentum,
                weight_decay=weight_decay)
        optimizer_a = torch.optim.Adam(
                model.module.arch_parameters(),
                lr=arch_lr,
                betas=(0.5, 0.999),
                weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, float(args.epochs), eta_min=args.learning_rate_min)
        scheduler_a = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer_a, float(args.epochs - eps_no_archs[sp]), eta_min=args.learning_rate_min)
        sm_dim = -1
        epochs = args.epochs
        eps_no_arch = eps_no_archs[sp]
        scale_factor = 0.2

        es_eps = 10
        prev_sw_nm = [[] for i in range(es_eps)]
        prev_sw_rd = [[] for i in range(es_eps)]
        pattern_unchanged = False

        for ep in range(epochs):
            scheduler.step()
            logging.info('Epoch: %d lr: %e', ep, scheduler.get_lr()[0])
            epoch_start = time.time()
            # training
            if ep < eps_no_arch:
                model.module.p = float(drop_rate[sp]) * (epochs - ep - 1) / epochs
                model.module.update_p()
                train_acc, _ = train(train_queue, valid_queue, model, criterion, optimizer, optimizer_a, clip, decoder, transform, train_arch=False, check_acc=check_acc, report_freq=args.report_freq)

            else:
                scheduler_a.step()
                logging.info('Epoch: %d lr_a: %e', ep, scheduler_a.get_lr()[0])

                model.module.p = float(drop_rate[sp]) * np.exp(-(ep - eps_no_arch) * scale_factor) 
                model.module.update_p()
                train_acc, _ = train(train_queue, valid_queue, model, criterion, optimizer, optimizer_a, clip, decoder, transform, train_arch=True, check_acc=check_acc, report_freq=args.report_freq)

                normal_prob = F.softmax(model.module.alphas_normal, dim=sm_dim).data.cpu().numpy()
                reduce_prob = F.softmax(model.module.alphas_reduce, dim=sm_dim).data.cpu().numpy()
                prev_sw_nm[ep%es_eps] = keep_n_on(switches_normal, normal_prob, num_to_keep[sp])
                prev_sw_rd[ep%es_eps] = keep_n_on(switches_reduce, reduce_prob, num_to_keep[sp])

                if args.no_es:
                    pattern_unchanged = all(prev_sw_nm[i] == prev_sw_nm[i+1] and \
                                            prev_sw_rd[i] == prev_sw_rd[i+1] for i in range(es_eps-1))

            if check_acc:
                logging.info('Train_acc %f', train_acc)
            epoch_duration = time.time() - epoch_start
            logging.info('Epoch Time: %ds', epoch_duration)

            # validation
            if ep >= epochs - 5:
                valid_acc, _ = infer(valid_queue, model, criterion, metric, decoder, transform, check_acc=check_acc, report_freq=args.report_freq)
                if check_acc:
                    logging.info('Valid_acc %f', valid_acc)
                else:
                    logging.info('Valid_score %f', valid_acc)

                if pattern_unchanged:
                    logging.info('Early Stopped due to unchanged pattern during last %d epoches.' % es_eps)
                    logging.info('switches_normal = %s', prev_sw_nm[ep%es_eps])
                    logging.info('switches_reduce = %s', prev_sw_rd[ep%es_eps])
                    break


        train_utils.save(model, os.path.join(args.save, 'weights.pt'))
        logging.info('------Keep %d paths------', num_to_keep[sp])


        normal_prob = F.softmax(model.module.alphas_normal, dim=sm_dim).data.cpu().numpy()
        reduce_prob = F.softmax(model.module.alphas_reduce, dim=sm_dim).data.cpu().numpy()
        
        if sp == len(num_to_keep) - 1:
            switches_normal_tmp = copy.deepcopy(switches_normal)

            for i in range(num_edges):
                if switches_normal[i][0]:
                    normal_prob[i][0] = 0
                if switches_reduce[i][0]:
                    reduce_prob[i][0] = 0

        # drop operations with low architecture weights
        switches_normal, top_n_ops = keep_n_on(switches_normal, normal_prob, num_to_keep[sp], print_top_n=True)
        for i in range(num_edges):
            logging.info('normal[%d] = %s', i, top_n_ops[i])
        logging.info('switches_normal = %s', switches_normal)

        switches_reduce, top_n_ops = keep_n_on(switches_reduce, reduce_prob, num_to_keep[sp], print_top_n=True)
        for i in range(num_edges):
            logging.info('reduce[%d] = %s', i, top_n_ops[i])
        logging.info('switches_reduce = %s', switches_reduce)

        if sp == len(num_to_keep) - 1:
            switches_normal = keep_2_branches(switches_normal, normal_prob)
            switches_reduce = keep_2_branches(switches_reduce, reduce_prob)

            # restrict maximum 2 skipconnect (normal cell only)
            genotype = parse_network(switches_normal, switches_reduce, step)
            logging.info(genotype)
            sc_cnt = check_sc_number(genotype[0])
            while sc_cnt > sc_limit:
                logging.info('Number of skip-connect in normal cell: %d', sc_cnt)
                logging.info('Restricting skip-connect...')

                switches_normal = copy.deepcopy(switches_normal_tmp)
                del_sc_idxs = get_sc_idxs_to_del(switches_normal, normal_prob, max_sc=sc_cnt-1)
                for i in range(num_edges):
                    if del_sc_idxs[i] is not None:
                        normal_prob[i][del_sc_idxs[i]] = 0

                switches_normal, top_n_ops = keep_n_on(switches_normal, normal_prob, 1, print_top_n=True)
                for i in range(num_edges):
                    logging.info('normal[%d] = %s', i, top_n_ops[i])
                logging.info('switches_normal = %s', switches_normal)
                switches_normal = keep_2_branches(switches_normal, normal_prob)
                genotype = parse_network(switches_normal, switches_reduce, step)
                logging.info(genotype)
                sc_cnt = check_sc_number(genotype[0])

    with open(f'./latest_genotype_{args.dataset}.txt', 'w') as f:
        f.write(str(genotype))


def train(train_queue, valid_queue, model, criterion, optimizer, optimizer_a, clip, decoder=None, transform=None, train_arch=True, check_acc=True, report_freq=50):
    objs = train_utils.AvgrageMeter()
    top1 = train_utils.AvgrageMeter()
    top5 = train_utils.AvgrageMeter()
    
    for step, data in enumerate(train_queue):
        model.train()

        if transform is not None:
            input, target, z = data
            z = z.cuda()
        else:
            input, target = data 
        input = input.cuda()
        target = target.cuda(non_blocking=True)

        if train_arch:
            # In the original implementation of DARTS, it is input_search, target_search = next(iter(valid_queue), which slows down
            # the training when using PyTorch 0.4 and above. 
            if transform is not None:
                try:
                    input_a, target_a, z_a = next(valid_queue_iter)
                except:
                    valid_queue_iter = iter(valid_queue)
                    input_a, target_a, z_a = next(valid_queue_iter)
                z_a = z_a.cuda()
            else:
                try:
                    input_a, target_a = next(valid_queue_iter)
                except:
                    valid_queue_iter = iter(valid_queue)
                    input_a, target_a = next(valid_queue_iter)
            input_a = input_a.cuda()
            target_a = target_a.cuda(non_blocking=True)

            optimizer_a.zero_grad()
            logits_a = model(input_a)

            if decoder is not None:
                logits_a = decoder.decode(logits_a).view(input_a.shape[0], -1)
                target_a = decoder.decode(target_a).view(input_a.shape[0], -1)

            if transform is not None:
                logits_a = transform(logits_a, z_a)
                target_a = transform(target_a, z_a)

            loss_a = criterion(logits_a, target_a)
            loss_a.backward()
            nn.utils.clip_grad_norm_(model.module.arch_parameters(), clip)
            optimizer_a.step()

        optimizer.zero_grad()
        logits = model(input)

        if decoder is not None:
            logits = decoder.decode(logits).view(input.shape[0], -1)
            target = decoder.decode(target).view(input.shape[0], -1)

        if transform is not None:
            logits = transform(logits, z)
            target = transform(target, z)

        loss = criterion(logits, target)
        loss.backward()
        nn.utils.clip_grad_norm_(model.module.network_parameters(), clip)
        optimizer.step()

        n = input.size(0)
        if check_acc:
            prec1, prec5 = train_utils.accuracy(logits, target, topk=(1, 5 if logits.size(-1) > 5 else 2))
            top1.update(prec1.data.item(), n)
            top5.update(prec5.data.item(), n)
        objs.update(loss.data.item(), n)

        if step % report_freq == 0:
            if check_acc:
                logging.info('Train Step: %03d Loss: %e R1: %f R5(R2): %f', step, objs.avg, top1.avg, top5.avg)
            else:
                logging.info('Train Step: %03d Loss: %e', step, objs.avg)

    return top1.avg, objs.avg


def infer(valid_queue, model, criterion, metric, decoder=None, transform=None, is_fsd=False, check_acc=True, report_freq=50):
    objs = train_utils.AvgrageMeter()
    top1 = train_utils.AvgrageMeter()
    top5 = train_utils.AvgrageMeter()
    mtrc = train_utils.AvgrageMeter()
    model.eval()

    outs, ys = [], []

    with torch.no_grad():
        for step, data in enumerate(valid_queue):
            if transform is not None:
                input, target, z = data
                z = z.cuda()
            else:
                input, target = data

            n = input.size(0)
            input = input.cuda()
            target = target.cuda(non_blocking=True)
            if is_fsd:
                logits = model(input).mean(0).unsqueeze(0)
            else:
                logits = model(input)

            if decoder is not None:
                logits = decoder.decode(logits).view(input.shape[0], -1)
                target = decoder.decode(target).view(input.shape[0], -1)

            if transform is not None:
                logits = transform(logits, z)
                target = transform(target, z)

            loss = criterion(logits, target)

            if check_acc:
                prec1, prec5 = train_utils.accuracy(logits, target, topk=(1, 5 if logits.size(-1) > 5 else 2))
                top1.update(prec1.data.item(), n)
                top5.update(prec5.data.item(), n)
            objs.update(loss.data.item(), n)

            if is_fsd:
                outs.append(torch.sigmoid(logits).detach().cpu().numpy()[0])
                ys.append(target.detach().cpu().numpy()[0])
            else:
                score = metric(logits, target)
                mtrc.update(score.item(), n)

            if step % report_freq == 0:
                if is_fsd:
                    logging.info('Valid Step: %03d Loss: %e R1: %f R5(R2): %f', step, objs.avg, top1.avg, top5.avg)
                elif check_acc:
                    logging.info('Valid Step: %03d Loss: %e R1: %f R5(R2): %f metric: %f', step, objs.avg, top1.avg, top5.avg, mtrc.avg)
                else:
                    logging.info('Valid Step: %03d Loss: %e metric: %f', step, objs.avg, mtrc.avg)

    if is_fsd:
        outs = np.asarray(outs).astype('float32')
        ys = np.asarray(ys).astype('int32')
        stats = calculate_stats(outs, ys)
        score = np.mean([stat['AP'] for stat in stats])
        return score, objs.avg
    elif check_acc:
        return top1.avg, objs.avg
    else:
        return mtrc.avg, objs.avg


def parse_network(switches_normal, switches_reduce, step=4):
    def _parse_switches(switches, concat):
        start = 0
        gene = []
        for n in concat:
            end = start + n
            for i in range(start, end):
                for j in range(len(switches[i])):
                    if switches[i][j]:
                        gene.append((PRIMITIVES[j], i - start))
            start = end
        return gene

    concat = range(2, 2+step)

    gene_normal = _parse_switches(switches_normal, concat)
    gene_reduce = _parse_switches(switches_reduce, concat)
    
    
    genotype = Genotype(
        normal=gene_normal, normal_concat=concat, 
        reduce=gene_reduce, reduce_concat=concat
    )
    return genotype


def get_op_rank(sw, p):
    on_idxs = list(filter(lambda x: sw[x] == True, range(len(PRIMITIVES))))
    p_rank = sorted(range(len(p)), key=lambda x: p[x], reverse=True)
    return [on_idxs[p_idx] for p_idx in p_rank]


def keep_n_on(switches_in, probs, n, print_top_n=False):
    switches = copy.deepcopy(switches_in)
    top_n_ops = [[] for i in range(len(switches))]
    for i in range(len(switches)):
        rank = get_op_rank(switches[i], probs[i, :])
        for idx in rank[n:]:
            switches[i][idx] = False
        top_n_ops[i] = [PRIMITIVES[idx] for idx in rank[:n]]

    if print_top_n:
        return switches, top_n_ops
    else:
        return switches


def keep_2_branches(switches_in, probs):
    switches = copy.deepcopy(switches_in)
    final_prob = [max(probs[i]) for i in range(len(switches))]
    keep = [0, 1]
    start = 2
    for n in range(3, 8):
        end = start + n
        if end > len(switches):
            break
        tb = final_prob[start:end]
        edge = sorted(range(n), key=lambda x: tb[x])
        keep.append(edge[-1] + start)
        keep.append(edge[-2] + start)
        start = end
    for i in range(len(switches)):
        if not i in keep:
            for j in range(len(PRIMITIVES)):
                switches[i][j] = False
    return switches  


def check_sc_number(genotype_normal):
    return [genotype_normal[i][0] == 'skip_connect' for i in range(len(genotype_normal))].count(True)


def get_sc_idxs_to_del(switches, probs, max_sc=8):
    def _get_sc_idx(sw):
        on_idxs = list(filter(lambda x: sw[x] == True, range(len(PRIMITIVES))))
        try:
            idx = on_idxs.index(1) # 1 is skip-connection op index.
        except ValueError:
            idx = None
        return idx

    sc_prob = [0 for i in range(len(switches))]
    sc_idxs = [None for i in range(len(switches))]
    for i in range(len(switches)):
        sc_idxs[i] = _get_sc_idx(switches[i])
        if sc_idxs[i] is not None:
            sc_prob[i] = probs[i][sc_idxs[i]]

    sc_rank = sorted(range(len(sc_prob)), key=lambda x: sc_prob[x], reverse=True)
    for idx in sc_rank[:max_sc]:
        sc_idxs[idx] = None

    return sc_idxs


if __name__ == '__main__':
    start_time = time.time()
    main() 
    end_time = time.time()
    duration = end_time - start_time
    logging.info('Total searching time: %ds', duration)
