"""CT-BS-Mamba training script.

The reported configurations are set by scripts/run_paper_configs.sh.
"""
import math, time, warnings, pickle, argparse, datetime, os, random, sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm


def log(msg: str):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

_HERE   = os.path.dirname(os.path.abspath(__file__))
_UTILS = os.path.join(_HERE, 'thp_utils')
sys.path.insert(0, _UTILS)
import Constants
import Utils
import Burst
from Dataset import get_dataloader

sys.path.insert(0, _HERE)
from models_ct_bsmamba import CTBSMamba, mixture_nll, mixture_mean

torch.set_printoptions(precision=2, sci_mode=False)
warnings.filterwarnings(action='ignore')

N_BIN   = 40
N_VOCAB = 1601   # MB-token vocabulary size N_BIN**2 + 1 (0 = PAD)


def quantization(value: float) -> int:
    value = value + 1
    num_bins  = N_BIN
    bin_width = 2.0 / num_bins
    q = int((value - 0) / bin_width)
    q = min(q, num_bins - 1)
    q = max(q, 0)
    return int(round(q, 0))


def prepare_mb(
    iet: torch.Tensor,
    event_type: torch.Tensor,
    mb_mode: str = 'rowmajor',
) -> tuple:
    """Sliding-window memory/burstiness statistics and the MB tokens for mb_mode.

    Returns (mb, mem_coef, burstiness), each (B, L). For 'rowmajor', mb is the MB token
    B_q * N_BIN + M_q + 1; for 'bs_only', mb is a constant non-zero token used only for the
    PAD mask. PAD positions are 0.
    """
    mem_coef   = Burst.memory_coefficient_batch(iet)
    burstiness = Burst.burstiness_batch(iet)

    device = iet.device
    mem_coef   = mem_coef.to(device=device, dtype=torch.float32)
    burstiness = burstiness.to(device=device, dtype=torch.float32)

    # Note: B_q is the quantised memory coefficient and M_q the quantised burstiness
    # (names swapped relative to convention, used consistently throughout).
    mb_cpu = torch.stack([mem_coef, burstiness], dim=-1).cpu().float()
    mb_cpu.apply_(quantization)   # each value -> 0..N_BIN-1

    B_q = mb_cpu[..., 0].long()
    M_q = mb_cpu[..., 1].long()

    non_pad = event_type.ne(Constants.PAD).to(device)

    def _pad(t):
        return (t.to(device) * non_pad).to(torch.int64)

    if mb_mode == 'rowmajor':
        mb = _pad((B_q * N_BIN + M_q + 1).to(torch.int64))
    else:  # bs_only
        mb = _pad(torch.ones_like(B_q))
    return mb, mem_coef, burstiness


# CTBS_* environment variables read by this script and the model; their values are written
# to the settings file of every run.
ENV_VARS = ('CTBS_VAL_PKL', 'CTBS_SELECT', 'CTBS_SCALE_TIME', 'CTBS_SCALE_MB', 'CTBS_SCALE_MEM',
            'CTBS_SCALE_BURST', 'CTBS_CT_POLES', 'CTBS_CT_WMAX', 'CTBS_EMA_BETA', 'CTBS_EMA_K')


def loss_scales(loss_weights: dict = None) -> dict:
    """Loss weights in use: the -scale_* arguments, overridden by CTBS_SCALE_* when set.

    'time' divides the summed squared point error, so the point-loss weight is eta = 1 / time;
    CTBS_SCALE_TIME=1e12 (eta ~ 0) is used for the point-loss sensitivity (Supplementary Table S2)
    and the NLL-only ablation.
    """
    lw = loss_weights or {}
    env = os.environ.get
    return {'event': lw.get('event', 1.0),
            'mb':    float(env('CTBS_SCALE_MB',    lw.get('mb', 4.0))),
            'mem':   float(env('CTBS_SCALE_MEM',   lw.get('mem', 10.0))),
            'burst': float(env('CTBS_SCALE_BURST', lw.get('burst', 10.0))),
            'time':  float(env('CTBS_SCALE_TIME',  1000.0))}


def _pinball(pred, true, q):
    """Pinball (quantile) loss for a single quantile q."""
    diff = true - pred
    return torch.sum(torch.where(diff >= 0, q * diff, (q - 1.0) * diff))


def compute_time_loss(t_pred: torch.Tensor, event_time: torch.Tensor,
                      mode: str = 'mse') -> tuple:
    """Point-head time loss. Returns (train_loss, raw_se); raw_se is the sum of squared IET
    errors used for RMSE.

    t_pred[:, i] predicts the IET between events i and i+1.
    """
    true  = (event_time[:, 1:] - event_time[:, :-1]).clamp(min=0)
    pred  = t_pred[:, :-1]
    pred  = torch.where(pred < 0, torch.full_like(pred, 999.0), pred)
    raw_se = torch.sum((pred - true) ** 2)

    if mode == 'mse':
        return raw_se, raw_se

    elif mode == 'huber':
        delta    = 1.0
        abs_diff = (pred - true).abs()
        loss     = torch.where(abs_diff <= delta,
                               0.5 * abs_diff ** 2,
                               delta * (abs_diff - 0.5 * delta))
        return torch.sum(loss), raw_se

    elif mode == 'q90':
        return _pinball(pred, true, 0.9), raw_se

    else:
        raise ValueError(f"Unknown time_loss mode: {mode}")


def run_batch(
    model:          CTBSMamba,
    event_time:     torch.Tensor,   # (B, L)
    iet:            torch.Tensor,   # (B, L)
    event_type:     torch.Tensor,   # (B, L) int, 0 = PAD, >= 1 = event
    pred_loss_func: nn.Module,
    time_loss_mode: str = 'mse',
    no_aux_loss:    bool = False,
    loss_weights:   dict = None,
) -> tuple:
    """Forward pass and losses for one batch; returns (total_loss, metrics dict)."""
    # binarise event_type (PAD=0, event=1)
    event_type_bin = event_type.clone()
    event_type_bin[event_type_bin >= 1] = 1

    mb, mem_coef, burstiness = prepare_mb(iet, event_type_bin, mb_mode=model.mb_mode)
    enc, head_out, mb_pred = model(mb, event_time, iet, burstiness, mem_coef)

    w = loss_scales(loss_weights)
    mb_loss, _ = Utils.type_loss(mb_pred, mb, pred_loss_func)
    num_event = event_time.ne(Constants.PAD).sum().item()

    if model.iet_head == 'mixture':
        # log-normal mixture head: IET NLL replaces the Hawkes LL term
        K     = model.n_mix
        raw   = head_out                                       # (B, L, 3K)
        true  = (event_time[:, 1:] - event_time[:, :-1]).clamp(min=0)   # (B, L-1)
        pmask = event_time[:, 1:].ne(Constants.PAD).float()    # (B, L-1)
        nll_elt = mixture_nll(raw[:, :-1], true, K)            # (B, L-1)
        iet_nll = torch.sum(nll_elt * pmask)
        # mixture mean as point estimate for RMSE and the auxiliary B/M terms
        point   = mixture_mean(raw, K)                         # (B, L)
        p       = point[:, :-1]
        se      = torch.sum(((p - true) * pmask) ** 2)
        t_det   = point.detach()
        pred_mem_coef   = Burst.memory_coefficient_batch(t_det).to(device=iet.device, dtype=torch.float32)
        pred_burstiness = Burst.burstiness_batch(t_det).to(device=iet.device, dtype=torch.float32)
        se_mem   = torch.sum((pred_mem_coef   - mem_coef)   ** 2)
        se_burst = torch.sum((pred_burstiness - burstiness) ** 2)
        total_loss = iet_nll + w['mb'] * mb_loss + se / w['time']
        if not no_aux_loss:
            total_loss = total_loss + w['mem'] * se_mem + w['burst'] * se_burst
        event_ll_report = -iet_nll
    else:
        # point (softplus) head with the Hawkes LL term
        t_pred = head_out.squeeze(-1)                          # (B, L)
        event_ll, non_event_ll = Utils.log_likelihood(model, enc, event_time, event_type_bin)
        event_loss = -torch.sum(event_ll - non_event_ll)
        time_loss, se = compute_time_loss(t_pred, event_time, mode=time_loss_mode)
        t_det          = t_pred.detach()
        pred_mem_coef  = Burst.memory_coefficient_batch(t_det).to(device=iet.device, dtype=torch.float32)
        pred_burstiness= Burst.burstiness_batch(t_det).to(device=iet.device, dtype=torch.float32)
        se_mem   = torch.sum((pred_mem_coef   - mem_coef)   ** 2)
        se_burst = torch.sum((pred_burstiness - burstiness) ** 2)
        total_loss = w['event'] * event_loss + w['mb'] * mb_loss + time_loss / w['time']
        if not no_aux_loss:
            total_loss = total_loss + w['mem'] * se_mem + w['burst'] * se_burst
        event_ll_report = -event_loss

    metrics = {
        'event_ll':  event_ll_report.item(),
        'se':         se.item(),          # raw squared error for RMSE
        'se_mem':     se_mem.item(),
        'se_burst':   se_burst.item(),
        'mb_loss':    mb_loss.item(),
        'num_event':  num_event,
        'num_pred':   num_event - event_time.shape[0],
    }
    return total_loss, metrics


def train_epoch(model, training_data, optimizer, pred_loss_func, opt):
    model.train()
    totals = dict(event_ll=0, se=0, se_mem=0, se_burst=0, mb=0,
                  num_event=0, num_pred=0)
    for idx, batch in enumerate(tqdm(training_data,
                                     mininterval=10,
                                     desc='Training',
                                     leave=False)):
        event_time, iet, event_type = (x.to(opt.device) for x in batch)
        optimizer.zero_grad()
        loss, metrics = run_batch(model, event_time, iet, event_type, pred_loss_func,
                                  time_loss_mode=opt.time_loss,
                                  no_aux_loss=opt.no_aux_loss,
                                  loss_weights=opt.loss_weights)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        for k in ('event_ll', 'se', 'se_mem', 'se_burst', 'mb_loss',
                  'num_event', 'num_pred'):
            totals[k if k != 'mb_loss' else 'mb'] += metrics.get(k, 0)

        if idx == 0:
            # weighted loss terms of the first batch, with the weights in use
            w = loss_scales(opt.loss_weights)
            s_ll = 1.0 if model.iet_head == 'mixture' else w['event']
            aux = ('' if opt.no_aux_loss else
                   f'  mem: {metrics["se_mem"] * w["mem"]:.1f}  burst: {metrics["se_burst"] * w["burst"]:.1f}')
            log(f'  Loss terms (first batch): NLL: {-metrics["event_ll"] * s_ll:.1f}  '
                f'squared error: {metrics["se"] / w["time"]:.4g}  '
                f'MB token: {metrics["mb_loss"] * w["mb"]:.1f}{aux}  '
                f'(weights: NLL {s_ll:g}, squared error 1/{w["time"]:g}, MB token {w["mb"]:g}'
                + ('' if opt.no_aux_loss else f', mem {w["mem"]:g}, burst {w["burst"]:g}') + ')')

    n = max(totals['num_pred'], 1)
    return (totals['event_ll'] / max(totals['num_event'], 1),
            math.sqrt(totals['se'] / n),
            math.sqrt(totals['se_mem'] / n),
            math.sqrt(totals['se_burst'] / n),
            totals['mb'] / max(totals['num_event'], 1))


def eval_epoch(model, validation_data, pred_loss_func, opt):
    model.eval()
    totals = dict(event_ll=0, se=0, se_mem=0, se_burst=0, mb=0,
                  num_event=0, num_pred=0)
    iet_true_all, iet_pred_all = {}, {}
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(validation_data,
                                         mininterval=10,
                                         desc='Validation',
                                         leave=False)):
            event_time, iet, event_type = (x.to(opt.device) for x in batch)
            loss, metrics = run_batch(model, event_time, iet, event_type, pred_loss_func,
                                      time_loss_mode=opt.time_loss,
                                      no_aux_loss=opt.no_aux_loss,
                                      loss_weights=opt.loss_weights)
            for k in ('event_ll', 'se', 'se_mem', 'se_burst', 'mb_loss',
                      'num_event', 'num_pred'):
                totals[k if k != 'mb_loss' else 'mb'] += metrics.get(k, 0)

            # second forward to collect IET predictions; t_pred[:, i] predicts iet[:, i+1]
            event_type_bin = event_type.clone()
            event_type_bin[event_type_bin >= 1] = 1
            mb_e, mem_coef_e, burst_e = prepare_mb(iet, event_type_bin, mb_mode=model.mb_mode)
            _, head_e, _ = model(mb_e, event_time, iet, burst_e, mem_coef_e)
            if model.iet_head == 'mixture':
                t_pred = mixture_mean(head_e, model.n_mix)      # (B, L)
            else:
                t_pred = head_e.squeeze(-1)
            iet_true_all[idx] = iet[:, 1:].cpu()
            iet_pred_all[idx] = t_pred[:, :-1].detach().cpu()

    n = max(totals['num_pred'], 1)

    # RMSE on the top-10% / top-25% largest true IETs; bias10 = mean pred / mean true
    tail_10_rmse, tail_25_rmse, tail_10_bias = 0.0, 0.0, 1.0
    try:
        all_true = np.concatenate([v.numpy().ravel() for v in iet_true_all.values()])
        all_pred = np.concatenate([v.numpy().ravel() for v in iet_pred_all.values()])
        mask_nonpad = all_true > 1e-6
        all_true, all_pred = all_true[mask_nonpad], all_pred[mask_nonpad]
        if len(all_true) > 10:
            q90 = np.percentile(all_true, 90)
            q75 = np.percentile(all_true, 75)
            m10 = all_true >= q90
            m25 = all_true >= q75
            if m10.sum() > 0:
                tail_10_rmse = float(np.sqrt(np.mean((all_pred[m10] - all_true[m10])**2)))
                tail_10_bias = float(np.mean(all_pred[m10]) / (np.mean(all_true[m10]) + 1e-9))
            if m25.sum() > 0:
                tail_25_rmse = float(np.sqrt(np.mean((all_pred[m25] - all_true[m25])**2)))
    except Exception:
        pass

    return (totals['event_ll'] / max(totals['num_event'], 1),
            math.sqrt(totals['se'] / n),
            math.sqrt(totals['se_mem'] / n),
            math.sqrt(totals['se_burst'] / n),
            totals['mb'] / max(totals['num_event'], 1),
            tail_10_rmse, tail_25_rmse, tail_10_bias,
            iet_true_all, iet_pred_all)


def train(model, training_data, validation_data,
          optimizer, scheduler, pred_loss_func, time_stamp, opt):
    valid_event_losses, valid_rmse = [], []
    valid_rmse_mem, valid_rmse_burst, valid_mbs = [], [], []
    best_iet_true, best_iet_pred = {}, {}
    best_state = None
    min_epoch = 0
    select_ll = os.environ.get('CTBS_SELECT') == 'll'

    # optional early stopping on validation LL
    patience = getattr(opt, 'patience', 0)
    best_ll = float('-inf')
    epochs_no_improve = 0

    for epoch_i in range(opt.epoch):
        epoch = epoch_i + 1
        log(f'Epoch {epoch}')

        t0 = time.time()
        tr_ll, tr_rmse, tr_mem, tr_burst, tr_mb = train_epoch(
            model, training_data, optimizer, pred_loss_func, opt)
        log('Train  ll:{ll:7.3f}  RMSE:{rmse:7.3f}  '
            'mem:{mem:6.3f}  burst:{burst:6.3f}  mb:{mb:6.3f}  '
            '{t:.1f}m'.format(ll=tr_ll, rmse=tr_rmse, mem=tr_mem,
                              burst=tr_burst, mb=tr_mb,
                              t=(time.time()-t0)/60))

        t0 = time.time()
        vl_ll, vl_rmse, vl_mem, vl_burst, vl_mb, \
            vl_tail10, vl_tail25, vl_bias10, iet_t, iet_p = eval_epoch(
            model, validation_data, pred_loss_func, opt)
        log('Valid  ll:{ll:7.3f}  RMSE:{rmse:7.3f}  '
            'mem:{mem:6.3f}  burst:{burst:6.3f}  mb:{mb:6.3f}  '
            'tail10:{t10:.3f}  bias10:{b10:.2f}  '
            '{t:.1f}m'.format(ll=vl_ll, rmse=vl_rmse, mem=vl_mem,
                              burst=vl_burst, mb=vl_mb,
                              t10=vl_tail10, b10=vl_bias10,
                              t=(time.time()-t0)/60))

        valid_event_losses.append(vl_ll)
        valid_rmse.append(vl_rmse)
        valid_rmse_mem.append(vl_mem)
        valid_rmse_burst.append(vl_burst)
        valid_mbs.append(vl_mb)

        log('Best so far  maxLL:{ll:7.3f}  minRMSE:{rmse:7.3f}  '
            'minMem:{mem:6.3f}  minBurst:{burst:6.3f}  minMB:{mb:6.3f}'.format(
                ll=max(valid_event_losses), rmse=min(valid_rmse),
                mem=min(valid_rmse_mem),    burst=min(valid_rmse_burst),
                mb=min(valid_mbs)))

        with open(os.path.join(opt.result_dir, time_stamp + '-log.txt'), 'a') as f:
            f.write('{e},{ll:.5f},{rmse:.5f},{mem:.5f},{burst:.5f},{mb:.5f},'
                    '{tll:.5f},{trmse:.5f},{tmem:.5f},{tburst:.5f},{tmb:.5f},'
                    '{t10:.5f},{t25:.5f},{b10:.5f}\n'.format(
                        e=epoch, ll=vl_ll, rmse=vl_rmse, mem=vl_mem,
                        burst=vl_burst, mb=vl_mb,
                        tll=tr_ll, trmse=tr_rmse, tmem=tr_mem,
                        tburst=tr_burst, tmb=tr_mb,
                        t10=vl_tail10, t25=vl_tail25, b10=vl_bias10))

        # Kept epoch: lowest validation RMSE; highest validation LL when CTBS_SELECT=ll.
        min_epoch = int(np.argmax(valid_event_losses)) if select_ll else int(np.argmin(valid_rmse))
        if min_epoch == epoch_i:
            best_iet_true = {k: v.cpu() for k, v in iet_t.items()}
            best_iet_pred = {k: v.cpu() for k, v in iet_p.items()}
            if getattr(opt, 'save_ckpt', False):
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        log(f'Valid RMSE by epoch: {[round(x, 2) for x in valid_rmse]}'
            f'  tail10: {vl_tail10:.3f}  bias10: {vl_bias10:.3f}')
        scheduler.step()

        if vl_ll > best_ll + 1e-4:
            best_ll = vl_ll
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if patience > 0 and epochs_no_improve >= patience:
            log(f'Early stop at epoch {epoch} (no validation-LL improvement for '
                f'{patience} epochs; best maxLL={max(valid_event_losses):.4f})')
            break

    # IET predictions of the kept epoch and of the final epoch
    prefix = os.path.join(opt.result_dir, time_stamp)
    _save_pkl(best_iet_true, prefix + '-iet_true.pkl')
    _save_pkl(best_iet_pred, prefix + '-iet_pred.pkl')
    _save_pkl({k: v.cpu() for k, v in iet_t.items()}, prefix + '-final_iet_true.pkl')
    _save_pkl({k: v.cpu() for k, v in iet_p.items()}, prefix + '-final_iet_pred.pkl')
    if getattr(opt, 'save_ckpt', False) and best_state is not None:
        torch.save(best_state, prefix + '-best_model.pt')
        log(f'Saved checkpoint: {os.path.basename(prefix)}-best_model.pt (epoch {min_epoch + 1})')
    log(f'Kept epoch (validation {"LL" if select_ll else "RMSE"}): {min_epoch + 1}')


def _save_pkl(obj, path: str):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def prepare_dataloader(opt):
    def _load(name, key):
        with open(name, 'rb') as f:
            data = pickle.load(f, encoding='latin-1')
        return data[key], int(data['dim_process'])

    log(f'Loading data from {opt.data}')
    train_data, _  = _load(opt.data + 'train.pkl', 'train')
    # CTBS_VAL_PKL=dev.pkl runs epoch-level evaluation and best-epoch selection on the dev
    # split; test metrics are then computed from the saved checkpoint. Unset: test split.
    _val_pkl = os.environ.get('CTBS_VAL_PKL')
    if _val_pkl:
        test_data, _ = _load(opt.data + _val_pkl, _val_pkl.rsplit('.', 1)[0])
        log(f'Validation stream: {_val_pkl}')
    else:
        test_data,  _  = _load(opt.data + 'test.pkl',  'test')
    trainloader = get_dataloader(train_data, opt.batch_size, shuffle=True)
    testloader  = get_dataloader(test_data,  opt.batch_size, shuffle=False)
    return trainloader, testloader, N_VOCAB


def dataset_tag_of(data_dir: str) -> str:
    """Dataset name for result paths; datasets/<ds>/prezeroclip/ is tagged <ds>."""
    path = os.path.normpath(data_dir)
    tag = os.path.basename(path)
    if tag == 'prezeroclip':
        tag = os.path.basename(os.path.dirname(path))
    return tag


def write_settings(path: str, setting: dict):
    with open(path, 'w') as f:
        for k, v in setting.items():
            print(f'{k}: {v}', file=f)
        for k in ENV_VARS:
            print(f'{k}: {os.environ.get(k, "(unset)")}', file=f)


def main():
    parser = argparse.ArgumentParser(description='CT-BS-Mamba training')
    parser.add_argument('-data',       type=str,   default='datasets/syn_exp_01/')
    parser.add_argument('-epoch',      type=int,   default=150)
    parser.add_argument('-batch_size', type=int,   default=16)
    parser.add_argument('-lr',         type=float, default=1e-4)
    parser.add_argument('-smooth',     type=float, default=0.1)
    parser.add_argument('-result_dir', type=str,   default='runs')
    parser.add_argument('-time_loss',  type=str,   default='mse',
                        choices=['mse', 'huber', 'q90'],
                        help='Time loss of the point head: mse (default), huber, q90 (pinball loss at 0.9)')
    parser.add_argument('-d_model',    type=int,   default=128)
    parser.add_argument('-d_rnn',      type=int,   default=64)
    parser.add_argument('-d_state',    type=int,   default=16)
    parser.add_argument('-d_conv',     type=int,   default=4)
    parser.add_argument('-expand',     type=int,   default=2)
    parser.add_argument('-n_layers',   type=int,   default=4)
    parser.add_argument('-d_bm',       type=int,   default=8)
    parser.add_argument('-dropout',    type=float, default=0.1)
    parser.add_argument('--use_ct',    action='store_true', default=False)
    parser.add_argument('--use_bs',    action='store_true', default=False)
    parser.add_argument('--no_rnn',    action='store_true', default=False)
    parser.add_argument('-ct_mode',  type=str, default='log_ratio',
                        choices=['log_ratio', 'log_abs'],
                        help='CT term: log_ratio (log interval relative to the EMA references, default) '
                             'or log_abs (absolute log interval)')
    parser.add_argument('--no_aux_loss', action='store_true', default=False,
                        help='Remove se_mem and se_burst auxiliary losses')
    parser.add_argument('--save_ckpt', action='store_true', default=False,
                        help='Save the state_dict of the kept epoch (lowest validation RMSE; '
                             'highest validation LL when CTBS_SELECT=ll) to <prefix>-best_model.pt')
    parser.add_argument('-mb_mode', type=str, default='rowmajor',
                        choices=['rowmajor', 'bs_only'],
                        help='rowmajor: MB-token embedding (default); bs_only: no token embedding')
    parser.add_argument('-scale_event', type=float, default=1.0,
                        help='Weight on the Hawkes LL term (point head only, default 1.0)')
    parser.add_argument('-scale_mb',    type=float, default=4.0,
                        help='Weight on the MB-token classification term (default 4.0; CTBS_SCALE_MB overrides)')
    parser.add_argument('-scale_mem',   type=float, default=10.0,
                        help='Weight on the auxiliary memory term (default 10.0; CTBS_SCALE_MEM overrides)')
    parser.add_argument('-scale_burst', type=float, default=10.0,
                        help='Weight on the auxiliary burstiness term (default 10.0; CTBS_SCALE_BURST overrides)')
    parser.add_argument('-patience',    type=int,   default=0,
                        help='Early-stop after N epochs without validation-LL improvement (0=off)')
    parser.add_argument('-iet_head', type=str, default='point',
                        choices=['point', 'mixture'],
                        help='IET head: point (softplus, default) or mixture (log-normal mixture density)')
    parser.add_argument('-n_mix', type=int, default=8,
                        help='Number of log-normal mixture components (iet_head=mixture)')
    parser.add_argument('-seed', type=int, default=42,
                        help='Random seed (multi-seed experiments)')

    opt = parser.parse_args()
    opt.device = torch.device('cuda')
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    np.random.seed(opt.seed)
    random.seed(opt.seed)
    opt.loss_weights = {'event': opt.scale_event, 'mb': opt.scale_mb,
                        'mem': opt.scale_mem, 'burst': opt.scale_burst}

    dataset_tag = dataset_tag_of(opt.data)
    cond_tag = (('ct' if opt.use_ct else '') + ('_bs' if opt.use_bs else '')) or 'plain'
    if opt.time_loss != 'mse':
        cond_tag = f"{cond_tag}-{opt.time_loss}"
    if opt.use_ct and opt.ct_mode != 'log_ratio':
        cond_tag = f"{cond_tag}-ct{opt.ct_mode}"
    if opt.use_bs and opt.d_bm != 8:
        cond_tag = f"{cond_tag}-dbm{opt.d_bm}"
    if opt.no_aux_loss:
        cond_tag = f"{cond_tag}-noaux"
    if opt.d_state != 16:
        cond_tag = f"{cond_tag}-ds{opt.d_state}"
    if opt.n_layers != 4:
        cond_tag = f"{cond_tag}-nl{opt.n_layers}"
    if opt.d_model != 128:
        cond_tag = f"{cond_tag}-dm{opt.d_model}"
    if opt.no_rnn:
        cond_tag = f"{cond_tag}-nornn"
    if opt.mb_mode != 'rowmajor':
        cond_tag = f"{cond_tag}-mb{opt.mb_mode}"
    if opt.iet_head == 'mixture':
        cond_tag = f"{cond_tag}-mix{opt.n_mix}"
    if opt.seed != 42:
        cond_tag = f"{cond_tag}-seed{opt.seed}"

    # results go to base_dir/{dataset}/{cond}/
    base_dir = opt.result_dir
    opt.result_dir = os.path.join(base_dir, dataset_tag, cond_tag)
    os.makedirs(opt.result_dir, exist_ok=True)

    run_date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    time_stamp = f"{run_date}-{dataset_tag}-{cond_tag}"

    setting = vars(opt)
    write_settings(os.path.join(opt.result_dir, time_stamp + '-settings.txt'), setting)
    with open(os.path.join(opt.result_dir, time_stamp + '-log.txt'), 'w') as f:
        f.write('Epoch,LL,RMSE,RMSE_MEM,RMSE_BURST,MB,'
                'Tr_LL,Tr_RMSE,Tr_RMSE_MEM,Tr_RMSE_BURST,Tr_MB,'
                'Tail10_RMSE,Tail25_RMSE,Tail10_Bias\n')

    log(f'dataset={dataset_tag}  cond={cond_tag}  result_dir={opt.result_dir}')
    log(f'Settings: {setting}')
    log('Environment: ' + '  '.join(f'{k}={os.environ[k]}' for k in ENV_VARS if k in os.environ))

    trainloader, testloader, num_types = prepare_dataloader(opt)
    model = CTBSMamba(
        num_types=num_types,
        d_model=opt.d_model,  d_rnn=opt.d_rnn,
        d_state=opt.d_state,  d_conv=opt.d_conv,
        expand=opt.expand,    n_layers=opt.n_layers,
        d_bm=opt.d_bm,
        use_ct=opt.use_ct,    use_bs=opt.use_bs,
        ct_mode=opt.ct_mode,
        dropout=opt.dropout,
        mb_mode=opt.mb_mode,
        iet_head=opt.iet_head,
        n_mix=opt.n_mix,
        no_rnn=opt.no_rnn,
    ).to(opt.device)

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=opt.lr, betas=(0.9, 0.999), eps=1e-5,
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.9)

    if opt.smooth > 0:
        pred_loss_func = Utils.LabelSmoothingLoss(
            opt.smooth, num_types, ignore_index=-1).to(opt.device)
    else:
        pred_loss_func = nn.CrossEntropyLoss(
            ignore_index=-1, reduction='none').to(opt.device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f'Parameters: {num_params:,}')
    log(f'use_ct={opt.use_ct}  use_bs={opt.use_bs}')

    train(model, trainloader, testloader,
          optimizer, scheduler, pred_loss_func,
          time_stamp, opt)


if __name__ == '__main__':
    warnings.filterwarnings(action='ignore')
    main()
