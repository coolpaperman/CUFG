import copy
import os
import time
from Remaindatax import create_finetuning_dataset
import torch
import utils
from imagenet import get_x_y_from_data_dict
from torch.utils.data import DataLoader


def l1_regularization(model):
    params_vec = []
    for param in model.parameters():
        params_vec.append(param.view(-1))
    return torch.linalg.norm(torch.cat(params_vec), ord=1)


def get_optimizer_and_scheduler(model, args):
    decreasing_lr = list(map(int, args.decreasing_lr.split(",")))
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=decreasing_lr, gamma=0.1
    )
    return optimizer, scheduler

def new_ungrad(grad_remain, grad_forget, epoch):
    # Method 1 GD
    # Unlearn_grad = [(a/ (a.norm() + 1e-8) + b/ (b.norm() + 1e-8))/((a/ (a.norm() + 1e-8) + b/ (b.norm() + 1e-8)).norm() + 1e-8) for a, b in zip(grad_remain, grad_forget)]
    # Method 1
    angle = []
    for grad_r, grad_f in zip(grad_remain, grad_forget):
        dot_product = torch.dot(grad_r.view(-1), grad_f.view(-1))
        norm_gradr = torch.norm(grad_r)
        norm_gradf = torch.norm(grad_f)
        cosine_similarity = dot_product / (norm_gradr * norm_gradf)
        angle.append(cosine_similarity)
    
    if any(cs > 0.45-(epoch//10)*0.05 for cs in angle):   
        Unlearn_grad = []
        for grad_r, grad_f in zip(grad_remain, grad_forget):
            grad_nr = grad_r/ (grad_r.norm() + 1e-8)
            grad_nf = grad_f/ (grad_f.norm() + 1e-8)
            u_grad = (grad_nr + (0.9**epoch)*grad_nf) / ((grad_nr + grad_nf).norm() + 1e-8) #(0.9**epoch)*
            Unlearn_grad.append(u_grad)
    else:
        Unlearn_grad = grad_remain

    
    # Method 2 PCGrad
    # Unlearn_grad = []
    # for grad_r, grad_f in zip(grad_remain, grad_forget):
    #     dot1 = (grad_r/ (grad_r.norm() + 1e-8) * grad_f/ (grad_f.norm() + 1e-8))#.sum(dim=1, keepdim=True)
    #     grad_rf = grad_r/ (grad_r.norm() + 1e-8) - torch.clamp_max(dot1, 0) * grad_f/ (grad_f.norm() + 1e-8)
    #     dot2 = (grad_f/ (grad_f.norm() + 1e-8) * grad_r/ (grad_r.norm() + 1e-8))#.sum(dim=1, keepdim=True)
    #     grad_fr = grad_f/ (grad_f.norm() + 1e-8) - torch.clamp_max(dot2, 0) * grad_r/ (grad_r.norm() + 1e-8)
    #     u_grad = (grad_rf + grad_fr) / ((grad_rf + grad_fr).norm() + 1e-8)
    #     Unlearn_grad.append(u_grad)
    return Unlearn_grad

def get_grad_forget_now(data_loaders, model, criterion, args):
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.unlearn_lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    
    accumulated_grads = [torch.zeros_like(param) for param in model.parameters()]
    total_batches = 0
    forget_loader = data_loaders
    model.eval()

    for i, (image, target) in enumerate(forget_loader):
        image = image.cuda()
        target = target.cuda()

        # compute output
        output_clean = model(image)
        loss = - criterion(output_clean, target)

        optimizer.zero_grad()
        loss.backward()

        with torch.no_grad():
            for idx, param in enumerate(model.parameters()):
                if param.grad is not None:
                    accumulated_grads[idx] += param.grad.clone()

        total_batches += 1  # 统计批次数量

    # 计算平均梯度
    grad_forget = [grad / total_batches for grad in accumulated_grads]

    return grad_forget

def untrain(data_loaders, sub_datasets, model, criterion, optimizer, epoch, args, mask=None, l1=False):
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    # remain_x = create_finetuning_dataset(model, data_loaders["retain"])
    # retain_x_loader = DataLoader(remain_x, batch_size=256, shuffle=True)
    train_loader = data_loaders["retain"] # retain_x_loader
    # switch to train mode
    model.train()

    start = time.time()
    if args.imagenet_arch:
        device = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
        for i, data in enumerate(train_loader):
            image, target = get_x_y_from_data_dict(data, device)
            if epoch < args.warmup:
                utils.warmup_lr(
                    epoch, i + 1, optimizer, one_epoch_step=len(train_loader), args=args
                )
            # compute output
            output_clean = model(image)

            loss = criterion(output_clean, target)
            if l1:
                loss = loss + args.alpha * l1_regularization(model)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)

            grad_remain = [p.grad.clone() for p in model.parameters() if p.grad is not None]
            grad_forget = get_grad_forget_now(data_loaders, model, criterion, args)
            Unlearn_grad = new_ungrad(grad_remain, grad_forget, epoch)
            for param, grad in zip(model.parameters(), Unlearn_grad):
                param.grad = grad

            if mask:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param.grad *= mask[name]
            optimizer.step()

            output = output_clean.float()
            loss = loss.float()
            # measure accuracy and record loss
            prec1 = utils.accuracy(output.data, target)[0]

            losses.update(loss.item(), image.size(0))
            top1.update(prec1.item(), image.size(0))

            if (i + 1) % args.print_freq == 0:
                end = time.time()
                print(
                    "Epoch: [{0}][{1}/{2}]\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Accuracy {top1.val:.3f} ({top1.avg:.3f})\t"
                    "Time {3:.2f}".format(
                        epoch, i, len(train_loader), end - start, loss=losses, top1=top1
                    )
                )
                start = time.time()

        print("train_accuracy {top1.avg:.3f}".format(top1=top1))
    else:
        for i, (image, target) in enumerate(train_loader):
        # if i<10:
            if epoch < args.warmup:
                utils.warmup_lr(
                    epoch, i + 1, optimizer, one_epoch_step=len(train_loader), args=args
                )

            image = image.cuda()
            target = target.cuda()

            # compute output
            output_clean = model(image)

            loss = criterion(output_clean, target)
            if l1:
                loss = loss + args.alpha * l1_regularization(model)
            optimizer.zero_grad()
            loss.backward()

            grad_remain = [param.grad.clone() for param in model.parameters() if param.grad is not None]
            #if  (epoch// 2) < len(sub_datasets):
            current_sub_for_idx = (epoch // 10)
            #else:
            #    current_sub_for_idx = 7
            # print(len(sub_datasets)-1-current_sub_for_idx)
            forget_loader = DataLoader(sub_datasets[len(sub_datasets)-1-current_sub_for_idx], batch_size=256, shuffle=True)
            grad_forget = get_grad_forget_now(forget_loader, model, criterion, args) # data_loaders["forget"]
            Unlearn_grad = new_ungrad(grad_remain, grad_forget, epoch)
            for param, grad in zip(model.parameters(), Unlearn_grad):
                param.grad = grad

            if mask:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param.grad *= mask[name]

            optimizer.step()

            output = output_clean.float()
            loss = loss.float()
            # measure accuracy and record loss
            prec1 = utils.accuracy(output.data, target)[0]

            losses.update(loss.item(), image.size(0))
            top1.update(prec1.item(), image.size(0))

            if (i + 1) % args.print_freq == 0:
                end = time.time()
                print(
                    "Epoch: [{0}][{1}/{2}]\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Accuracy {top1.val:.3f} ({top1.avg:.3f})\t"
                    "Time {3:.2f}".format(
                        epoch, i, len(train_loader), end - start, loss=losses, top1=top1
                    )
                )
                start = time.time()

    print("train_accuracy {top1.avg:.3f}".format(top1=top1))

    return top1.avg
