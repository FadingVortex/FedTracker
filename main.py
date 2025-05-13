# -*- coding: UTF-8 -*-
import copy
import os.path
import time
import numpy as np
import torch
import random
from torch.backends import cudnn
from torch.utils.data import DataLoader
import json

from fed.client import create_clients
from fed.server import FedAvg
from utils.datasets import get_full_dataset
from utils.models import get_model
from utils.test import test_img
from utils.train import get_optim, gem_train, set_bn_eval
from utils.utils import printf, load_args
from watermark.fingerprint import *
from watermark.watermark import *
from tqdm import tqdm


if __name__ == '__main__':
    args = load_args()  # 加载命令行参数
    log_path = os.path.join(args.save_dir, 'log.txt')  # 定义日志文件路径
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)  # 若保存路径不存在，则创建
    # 保存参数信息到文件中
    with open(os.path.join(args.save_dir, "args.txt"), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    # 设置使用的设备：GPU 或 CPU
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # 加载完整数据集（训练集和测试集）
    train_dataset, test_dataset = get_full_dataset(args.dataset, img_size=(args.image_size, args.image_size))
    # 创建客户端
    clients = create_clients(args, train_dataset)
    # 获取全局模型
    global_model = get_model(args)
    if args.pre_train:
        # 加载预训练模型参数
        global_model.load_state_dict(torch.load(args.pre_train_path))
    # 初始化每个客户端的模型为全局模型的副本
    for client in clients:
        client.set_model(copy.deepcopy(global_model))

    # 初始化水印和指纹机制
    if args.watermark or args.fingerprint:
        weight_size = get_embed_layers_length(global_model, args.embed_layer_names)  # 获取嵌入层参数总长度
        local_fingerprints = generate_fingerprints(args.num_clients, args.lfp_length)  # 生成每个客户端的指纹
        extracting_matrices = generate_extracting_matrices(weight_size, args.lfp_length, args.num_clients)  # 生成提取矩阵
        trigger_set = generate_waffle_pattern(args)  # 生成水印触发样本集
        watermark_set = DataLoader(trigger_set, batch_size=args.local_bs, shuffle=True)  # 构建触发集的 DataLoader

    # 设置随机种子以保证结果可复现
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        random.seed(args.seed)
        cudnn.deterministic = True

    # 初始化训练指标
    train_loss = []
    val_loss, val_acc = [], []
    acc_best = None
    client_acc_best = [0 for i in range(args.num_clients)]  # 每个客户端的最佳准确率
    num_clients_each_iter = max(min(args.num_clients, args.num_clients_each_iter), 1)  # 每轮参与的客户端数

    # 开始联邦学习训练过程
    for epoch in tqdm(range(args.start_epochs, args.epochs)):
        start_time = time.time()
        local_losses = []
        local_models = []
        local_nums = []

        # 学习率衰减
        for client in clients:
            client.local_lr *= args.lr_decay
        clients_idxs = np.random.choice(range(args.num_clients), num_clients_each_iter, replace=False)  # 随机选择客户端参与本轮训练

        # 客户端本地训练
        for idx in clients_idxs:
            current_client = clients[idx]
            local_model, num_samples, local_loss = current_client.train_one_iteration()
            local_models.append(copy.deepcopy(local_model))
            local_losses.append(local_loss)
            local_nums.append(num_samples)

        # 全局模型聚合
        global_model.load_global_model(FedAvg(local_models, local_nums), args.device, args.gem)

        # 打印当前轮平均损失
        avg_loss = sum(local_losses) / len(local_losses)
        printf('Round {:3d}, Average loss {:.3f}'.format(epoch, avg_loss), log_path)
        printf('Time: {}'.format(time.time() - start_time), log_path)
        train_loss.append(avg_loss)

        # 测试全局模型
        if (epoch + 1) % args.test_interval == 0:
            acc_test, acc_test_top5 = test_img(global_model, test_dataset, args)
            printf("Testing accuracy: Top1: {:.3f}, Top5: {:.3f}".format(acc_test, acc_test_top5), log_path)
            if acc_best is None or acc_test >= acc_best:
                acc_best = acc_test
                if args.save:
                    torch.save(global_model.state_dict(), args.save_dir + "model_best.pth")

        # 嵌入水印和指纹
        if args.watermark:
            # 嵌入全局水印
            # 优化器设置，使用 SGD
            watermark_optim = get_optim(global_model, args.local_optim, args.lambda1)
            watermark_acc, _ = test_img(global_model, trigger_set, args)
            watermark_loss_func = nn.CrossEntropyLoss()
            watermark_embed_iters = 0
            while watermark_acc <= 98 and watermark_embed_iters <= args.watermark_max_iters:
                global_model.train()
                global_model.to(args.device)
                if args.freeze_bn:
                    global_model.apply(set_bn_eval)  # 冻结 BN 层
                watermark_embed_iters += 1
                for batch_idx, (images, labels) in enumerate(watermark_set):
                    images, labels = images.to(args.device), labels.to(args.device)
                    global_model.zero_grad()
                    probs = global_model(images)
                    watermark_loss = watermark_loss_func(probs, labels)
                    watermark_loss.backward()
                    if args.gem:
                        global_model = gem_train(global_model)
                    watermark_optim.step()
                watermark_acc, _ = test_img(global_model, trigger_set, args)

            # 嵌入本地指纹
            if args.fingerprint:
                for client_idx in range(len(clients)):
                    client_fingerprint = local_fingerprints[client_idx]
                    client_model = copy.deepcopy(global_model)
                    embed_layers = get_embed_layers(client_model, args.embed_layer_names)
                    fss, extract_idx = extracting_fingerprints(embed_layers, local_fingerprints, extracting_matrices)
                    count = 0
                    while (extract_idx != client_idx or (
                            client_idx == extract_idx and fss < 0.85)) and count <= args.fingerprint_max_iters:
                        client_grad = calculate_local_grad(embed_layers, client_fingerprint,
                                                           extracting_matrices[client_idx])
                        client_grad = torch.mul(client_grad, -args.lambda2)
                        weight_count = 0
                        for embed_layer in embed_layers:
                            weight_length = embed_layer.weight.shape[0]
                            embed_layer.weight = torch.nn.Parameter(
                                torch.add(embed_layer.weight, client_grad[weight_count: weight_count + weight_length]))
                            weight_count += weight_length
                        count += 1
                        fss, extract_idx = extracting_fingerprints(embed_layers, local_fingerprints,
                                                                   extracting_matrices)
                    printf("(Client_idx:{}, Result_idx:{}, FSS:{})".format(client_idx, extract_idx, fss), log_path)
                    clients[client_idx].set_model(client_model)
            else:
                for client in clients:
                    client.set_model(copy.deepcopy(global_model))
        else:
            # 如果没有水印，仅发送全局模型给客户端
            for client in clients:
                client.set_model(copy.deepcopy(global_model))

        # 测试客户端模型
        if (epoch + 1) % args.test_interval == 0:
            if args.watermark:
                avg_watermark_acc = 0.0
                avg_fss = 0.0
                client_acc = []
                client_acc_top5 = []
                for client_idx in range(args.num_clients):
                    acc, acc_top5 = test_img(clients[client_idx].model, test_dataset, args)
                    watermark_acc, _ = test_img(clients[client_idx].model, trigger_set, args)
                    client_acc.append(acc)
                    client_acc_top5.append(acc_top5)
                    avg_watermark_acc += watermark_acc
                    if args.fingerprint:
                        embed_layers = get_embed_layers(clients[client_idx].model, args.embed_layer_names)
                        fss, extract_idx = extracting_fingerprints(embed_layers, local_fingerprints,
                                                                   extracting_matrices)
                        avg_fss += fss
                    if acc >= client_acc_best[client_idx]:
                        client_acc_best[client_idx] = acc
                        if args.save:
                            torch.save(clients[client_idx].get_model().state_dict(),
                                       args.save_dir + "model_" + str(client_idx) + ".pth")
                # 打印客户端整体测试结果统计
                avg_acc = np.mean(client_acc)
                max_acc = np.max(client_acc)
                min_acc = np.min(client_acc)
                median_acc = np.median(client_acc)
                low_acc = np.percentile(client_acc, 25)
                high_acc = np.percentile(client_acc, 75)
                avg_acc_top5 = np.mean(client_acc_top5)
                max_acc_top5 = np.max(client_acc_top5)
                min_acc_top5 = np.min(client_acc_top5)
                median_acc_top5 = np.median(client_acc_top5)
                low_acc_top5 = np.percentile(client_acc_top5, 25)
                high_acc_top5 = np.percentile(client_acc_top5, 75)
                avg_watermark_acc /= args.num_clients
                avg_fss /= args.num_clients
                printf("Clients Average Testing accuracy: {:.2f}".format(avg_acc), log_path)
                printf("Clients Quantile Testing accuracy, Top1: {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}".format(min_acc,
                                                                                                                low_acc,
                                                                                                                median_acc,
                                                                                                                high_acc,
                                                                                                                max_acc),
                       log_path)
                printf("Clients Quantile Testing accuracy, Top5: {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}".format(
                    min_acc_top5, low_acc_top5, median_acc_top5, high_acc_top5, max_acc_top5), log_path)
                printf("Watermark Average Testing accuracy:{:.2f}".format(avg_watermark_acc), log_path)
                if args.fingerprint:
                    printf("Average fss: {:.4f}".format(avg_fss), log_path)

    # 训练结束后打印最优准确率信息
    printf("Best Acc of Global Model:" + str(acc_best), log_path)
    if args.watermark:
        printf("Clients' Best Acc:", log_path)
        for client_idx in range(args.num_clients):
            printf(client_acc_best[client_idx], log_path)
        avg_acc = np.mean(client_acc_best)
        max_acc = np.max(client_acc_best)
        min_acc = np.min(client_acc_best)
        median_acc = np.median(client_acc_best)
        low_acc = np.percentile(client_acc_best, 25)
        high_acc = np.percentile(client_acc_best, 75)
        printf("Clients Average Testing accuracy: {:.2f}".format(avg_acc), log_path)
        printf("Clients Quantile Testing accuracy: {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}".format(min_acc, low_acc,
                                                                                                  median_acc, high_acc,
                                                                                                  max_acc), log_path)
    # 保存最后一轮的全局模型
    if args.save:
        torch.save(global_model.state_dict(),
                   args.save_dir + "model_last_epochs_" + str((args.epochs + args.start_epochs)) + ".pth")
