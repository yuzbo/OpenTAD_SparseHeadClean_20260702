import copy
import torch
import tqdm
from opentad.utils.misc import AverageMeter, reduce_loss


def _find_first_nonfinite_grad(model):
    for name, param in model.named_parameters():
        if param.grad is not None and not torch.isfinite(param.grad).all():
            return name
    return None


def _summarize_grad_health(model, topk=8):
    nonfinite = []
    large_finite = []
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        grad = param.grad.detach()
        finite = torch.isfinite(grad)
        if not finite.all():
            nonfinite.append(
                dict(
                    name=name,
                    shape=tuple(grad.shape),
                    finite_count=int(finite.sum().item()),
                    nonfinite_count=int(grad.numel() - finite.sum().item()),
                )
            )
            continue
        absmax = float(grad.abs().max().item()) if grad.numel() > 0 else 0.0
        large_finite.append((absmax, name, tuple(grad.shape)))

    large_finite.sort(reverse=True, key=lambda item: item[0])
    return {
        "grad_nonfinite_param_count": len(nonfinite),
        "grad_nonfinite_params": nonfinite[:topk],
        "grad_top_absmax_params": [
            dict(name=name, absmax=absmax, shape=shape) for absmax, name, shape in large_finite[:topk]
        ],
    }


def _format_debug_report(report):
    ordered_items = []
    for key in sorted(report.keys()):
        ordered_items.append(f"{key}={report[key]}")
    return " | ".join(ordered_items)


def _collect_runtime_debug(model, data_dict, bad_param_name):
    target = model.module if hasattr(model, "module") else model
    if hasattr(target, "collect_runtime_debug"):
        try:
            return target.collect_runtime_debug(data_dict, bad_param_name)
        except Exception as exc:
            return {"debug_collection_error": repr(exc), "bad_param_name": bad_param_name}
    return None


def _grad_clip_parameters(model):
    target = model.module if hasattr(model, "module") else model
    if hasattr(target, "grad_clip_parameters"):
        return target.grad_clip_parameters()
    return model.parameters()


def train_one_epoch(
    train_loader,
    model,
    optimizer,
    scheduler,
    curr_epoch,
    logger,
    model_ema=None,
    clip_grad_l2norm=-1,
    logging_interval=200,
    runtime_debug_interval=-1,
    scaler=None,
):
    """Training the model for one epoch"""

    logger.info("[Train]: Epoch {:d} started".format(curr_epoch))
    losses_tracker = {}
    num_iters = len(train_loader)
    use_amp = False if scaler is None else True

    target = model.module if hasattr(model, "module") else model
    if hasattr(target, "set_train_epoch"):
        target.set_train_epoch(curr_epoch)

    model.train()
    for iter_idx, data_dict in enumerate(train_loader):
        optimizer.zero_grad()

        # current learning rate
        curr_backbone_lr = None
        if hasattr(model.module, "backbone"):  # if backbone exists
            if model.module.backbone.freeze_backbone == False:  # not frozen
                curr_backbone_lr = scheduler.get_last_lr()[0]
        curr_det_lr = scheduler.get_last_lr()[-1]

        # forward pass
        with torch.cuda.amp.autocast(dtype=torch.float16, enabled=use_amp):
            losses = model(**data_dict, return_loss=True)

        if not torch.isfinite(losses["cost"]):
            logger.error(
                "[Train]: non-finite cost detected at epoch=%d iter=%d, skip optimizer step",
                curr_epoch,
                iter_idx,
            )
            optimizer.zero_grad(set_to_none=True)
            continue

        # compute the gradients
        if use_amp:
            scaler.scale(losses["cost"]).backward()
        else:
            losses["cost"].backward()

        # gradient clipping (to stabilize training if necessary)
        grads_unscaled = False
        if clip_grad_l2norm > 0.0:
            if use_amp:
                scaler.unscale_(optimizer)
                grads_unscaled = True
            grad_clip_parameters = _grad_clip_parameters(model)
            torch.nn.utils.clip_grad_norm_(grad_clip_parameters, clip_grad_l2norm)

        # update parameters
        if use_amp:
            if not grads_unscaled:
                scaler.unscale_(optimizer)
            bad_param_name = _find_first_nonfinite_grad(model)
            if bad_param_name is not None:
                logger.error(
                    "[Train]: non-finite gradients detected at epoch=%d iter=%d, param=%s, skip optimizer step",
                    curr_epoch,
                    iter_idx,
                    bad_param_name,
                )
                debug_report = _collect_runtime_debug(model, data_dict, bad_param_name)
                if debug_report is not None:
                    debug_report.update(_summarize_grad_health(model))
                    logger.error(
                        "[Train][Diag]: epoch=%d iter=%d %s",
                        curr_epoch,
                        iter_idx,
                        _format_debug_report(debug_report),
                    )
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                continue
            scaler.step(optimizer)
            scaler.update()
        else:
            bad_param_name = _find_first_nonfinite_grad(model)
            if bad_param_name is not None:
                logger.error(
                    "[Train]: non-finite gradients detected at epoch=%d iter=%d, param=%s, skip optimizer step",
                    curr_epoch,
                    iter_idx,
                    bad_param_name,
                )
                debug_report = _collect_runtime_debug(model, data_dict, bad_param_name)
                if debug_report is not None:
                    debug_report.update(_summarize_grad_health(model))
                    logger.error(
                        "[Train][Diag]: epoch=%d iter=%d %s",
                        curr_epoch,
                        iter_idx,
                        _format_debug_report(debug_report),
                    )
                optimizer.zero_grad(set_to_none=True)
                continue
            optimizer.step()

        # update scheduler
        scheduler.step()

        # update ema
        if model_ema is not None:
            model_ema.update(model)

        # track all losses
        losses = reduce_loss(losses)  # only for log
        for key, value in losses.items():
            if key not in losses_tracker:
                losses_tracker[key] = AverageMeter()
            losses_tracker[key].update(value.item())

        # printing each logging_interval
        if ((iter_idx != 0) and (iter_idx % logging_interval) == 0) or ((iter_idx + 1) == num_iters):
            # print to terminal
            block1 = "[Train]: [{:03d}][{:05d}/{:05d}]".format(curr_epoch, iter_idx, num_iters - 1)
            block2 = "Loss={:.4f}".format(losses_tracker["cost"].avg)
            block3 = ["{:s}={:.4f}".format(key, value.avg) for key, value in losses_tracker.items() if key != "cost"]
            block4 = "lr_det={:.1e}".format(curr_det_lr)
            if curr_backbone_lr is not None:
                block4 = "lr_backbone={:.1e}".format(curr_backbone_lr) + "  " + block4
            block5 = "mem={:.0f}MB".format(torch.cuda.max_memory_allocated() / 1024.0 / 1024.0)
            logger.info("  ".join([block1, block2, "  ".join(block3), block4, block5]))

        should_log_runtime_debug = (
            runtime_debug_interval > 0
            and ((iter_idx + 1) == num_iters)
            and (((curr_epoch + 1) % runtime_debug_interval) == 0)
        )
        if should_log_runtime_debug:
            is_rank0 = True
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                is_rank0 = torch.distributed.get_rank() == 0
            if is_rank0:
                debug_report = _collect_runtime_debug(model, data_dict, "periodic")
                if debug_report is not None:
                    logger.info(
                        "[Train][RuntimeDebug]: epoch=%d iter=%d %s",
                        curr_epoch,
                        iter_idx,
                        _format_debug_report(debug_report),
                    )


def val_one_epoch(
    val_loader,
    model,
    logger,
    rank,
    curr_epoch,
    model_ema=None,
    use_amp=False,
):
    """Validating the model for one epoch: compute the loss"""

    # load the ema dict for evaluation
    if model_ema != None:
        current_dict = copy.deepcopy(model.state_dict())
        model.load_state_dict(model_ema.module.state_dict())

    logger.info("[Val]: Epoch {:d} Loss".format(curr_epoch))
    losses_tracker = {}

    model.eval()
    for data_dict in tqdm.tqdm(val_loader, disable=(rank != 0)):
        with torch.cuda.amp.autocast(dtype=torch.float16, enabled=use_amp):
            with torch.no_grad():
                losses = model(**data_dict, return_loss=True)

        # track all losses
        losses = reduce_loss(losses)  # only for log
        for key, value in losses.items():
            if key not in losses_tracker:
                losses_tracker[key] = AverageMeter()
            losses_tracker[key].update(value.item())

    # print to terminal
    block1 = "[Val]: [{:03d}]".format(curr_epoch)
    block2 = "Loss={:.4f}".format(losses_tracker["cost"].avg)
    block3 = ["{:s}={:.4f}".format(key, value.avg) for key, value in losses_tracker.items() if key != "cost"]
    logger.info("  ".join([block1, block2, "  ".join(block3)]))

    # load back the normal model dict
    if model_ema != None:
        model.load_state_dict(current_dict)
    return losses_tracker["cost"].avg
