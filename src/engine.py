# src/engine.py
# Training engine: losses, train/val loops, CutMix, MC Dropout, Grad-CAM, evaluation.

import os, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import nullcontext
from torch.utils.data import DataLoader
from tqdm import tqdm
import cv2
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, classification_report


# ─── Reproducibility ──────────────────────────────────────────────────────────
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


# ─── Loss ─────────────────────────────────────────────────────────────────────
class ACDNetLoss(nn.Module):
    def __init__(self, lam_det=1.0, lam_seg=0.5, lam_sev=1.0, lam_temp=0.1, pos_weight=3.0, sev_class_weights=None):
        super().__init__()
        self.lam_det, self.lam_seg = lam_det, lam_seg
        self.lam_sev, self.lam_temp = lam_sev, lam_temp
        self.bce_det = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
        self.bce_seg = nn.BCEWithLogitsLoss()
        
        # 🔥 FIX #8: Update severity class weights for 3-class problem (merged grades 0-1 with 1)
        # New distribution: grade 0-1 (merged, ~30%), grade 2 (~50%), grade 3 (~20%)
        # Use inverse frequency weighting to emphasize minority classes
        if sev_class_weights is None:
            # 3-class weights: [merged_0-1, grade_2, grade_3]
            sev_class_weights = torch.tensor([1.0, 0.7, 1.2], dtype=torch.float32)
        
        self.ce_sev = nn.CrossEntropyLoss(weight=sev_class_weights, label_smoothing=0.1)

    def detection_loss(self, logit, label):
        v = label >= 0
        if v.sum() == 0: return torch.tensor(0., device=logit.device, requires_grad=True)
        # Keep BCE pos_weight on the same device as current logits.
        if self.bce_det.pos_weight is not None and self.bce_det.pos_weight.device != logit.device:
            self.bce_det.pos_weight = self.bce_det.pos_weight.to(logit.device)
        return self.bce_det(logit[v].squeeze(1), label[v].float())

    def segmentation_loss(self, mask_logit, mask_gt, bbox_pred, bbox_gt=None):
        has_mask = mask_gt.flatten(1).sum(1) > 0
        ml = self.bce_seg(mask_logit[has_mask], mask_gt[has_mask].float()) if has_mask.sum() > 0 else torch.tensor(0., device=mask_logit.device)
        bl = torch.tensor(0., device=bbox_pred.device)
        if bbox_gt is not None:
            vb = bbox_gt.sum(1) > 0
            if vb.sum() > 0: bl = F.mse_loss(bbox_pred[vb], bbox_gt[vb].float())
        return ml + 0.5 * bl

    def severity_loss(self, logit, grade):
        v = grade >= 0
        if v.sum() == 0: return torch.tensor(0., device=logit.device, requires_grad=True)
        # 🔥 Move class weights to same device as logit
        if self.ce_sev.weight is not None:
            self.ce_sev.weight = self.ce_sev.weight.to(logit.device)
        return self.ce_sev(logit[v], grade[v])

    def temporal_loss(self, sev_seq):
        if sev_seq.shape[0] < 2: return torch.tensor(0., device=sev_seq.device, requires_grad=True)
        probs = F.softmax(sev_seq, dim=-1)
        return (probs[1:] - probs[:-1]).abs().mean()

    def forward(self, outputs, targets, video_seq=None):
        l_det  = self.detection_loss(outputs["detection_logit"], targets["polyp_label"])
        l_seg  = self.segmentation_loss(outputs["mask_logit"], targets["mask"], outputs["bbox"], targets.get("bbox"))
        l_sev  = self.severity_loss(outputs["severity_logit"], targets["uc_grade"])
        l_temp = self.temporal_loss(video_seq) if video_seq is not None else torch.tensor(0., device=l_det.device)
        total  = self.lam_det*l_det + self.lam_seg*l_seg + self.lam_sev*l_sev + self.lam_temp*l_temp
        return {"total": total, "detection": l_det, "segmentation": l_seg, "severity": l_sev, "temporal": l_temp}


# ─── CutMix ───────────────────────────────────────────────────────────────────
def cutmix(images, labels_det, alpha=0.4):
    lam = np.random.beta(alpha, alpha)
    B, _, H, W = images.shape
    idx = torch.randperm(B, device=images.device)
    cr  = np.sqrt(1 - lam)
    ch, cw = int(H*cr), int(W*cr)
    cx, cy = random.randint(0,W), random.randint(0,H)
    x1,x2 = max(cx-cw//2,0), min(cx+cw//2,W)
    y1,y2 = max(cy-ch//2,0), min(cy+ch//2,H)
    mixed = images.clone()
    mixed[:,:,y1:y2,x1:x2] = images[idx,:,y1:y2,x1:x2]
    lam_adj = 1 - (x2-x1)*(y2-y1)/(H*W)
    return mixed, labels_det, labels_det[idx], lam_adj


# ─── Train one epoch ──────────────────────────────────────────────────────────
def train_one_epoch(model, image_loader, video_loader, criterion, optimizer, device,
                    use_cutmix=True, use_amp=False, scaler=None):
    model.train()
    M = {"total":0.,"detection":0.,"segmentation":0.,"severity":0.,"temporal":0.}
    vid_iter = iter(video_loader) if video_loader else None
    nb = 0
    amp_enabled = bool(use_amp and device.type == "cuda")
    for batch in tqdm(image_loader, desc="  Train", leave=False):
        imgs     = batch["image"].to(device)
        poly_lbl = batch["polyp_label"].to(device)
        uc_grade = batch["uc_grade"].to(device)
        mask     = batch["mask"].to(device)
        if use_cutmix and random.random() < 0.5:
            imgs, la, lb, lam = cutmix(imgs, poly_lbl)
        else:
            la = lb = poly_lbl; lam = 1.0
        optimizer.zero_grad()
        amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled)
        with amp_ctx:
            out = model(imgs)
            tgt = {"polyp_label": la, "uc_grade": uc_grade, "mask": mask, "bbox": None}
            L   = criterion(out, tgt)
            if lam < 1.0:
                tgt2 = dict(tgt); tgt2["polyp_label"] = lb
                L2   = criterion(out, tgt2)
                total = (lam*L["detection"] + (1-lam)*L2["detection"] +
                         criterion.lam_seg*L["segmentation"] + criterion.lam_sev*L["severity"])
            else:
                total = L["total"]
        temp = torch.tensor(0., device=device)
        if vid_iter is not None:
            vb = None
            try:
                vb = next(vid_iter)
            except StopIteration:
                try:
                    vid_iter = iter(video_loader)
                    vb = next(vid_iter)
                except Exception as e:
                    print(f"[WARN] Could not restart video iterator this step: {e}")
                    vb = None
            except Exception as e:
                print(f"[WARN] Failed to fetch video batch this step: {e}")
                vb = None

            if vb is not None:
                try:
                    frames = vb["frames"].to(device)
                    # Validate shape before using
                    if frames.dim() == 4:
                        # Shape is [B, C, H, W], treat as single-frame sequence.
                        frames = frames.unsqueeze(1)

                    if frames.dim() != 5:
                        raise ValueError(f"Invalid frames shape: {frames.shape}")

                    B, T, C, H, W = frames.shape
                    with amp_ctx:
                        # Keep grad enabled: temporal loss is a trainable objective.
                        fo = model(frames.view(B * T, C, H, W))
                    # Model outputs are flattened as [B*T, classes]. Rebuild [T, B, classes]
                    # so temporal_loss compares adjacent time steps.
                    sev_seq = fo["severity_logit"].view(B, T, -1).permute(1, 0, 2)
                    temp = criterion.temporal_loss(sev_seq)
                    total = total + criterion.lam_temp * temp
                except Exception as e:
                    # Gracefully skip corrupted video batches
                    print(f"[WARN] Skipping corrupted video batch: {e}")
                    temp = torch.tensor(0., device=device)

        if amp_enabled and scaler is not None:
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        for k in ["detection","segmentation","severity"]:
            M[k] += L[k].item()
        M["total"] += total.item(); M["temporal"] += temp.item(); nb += 1
    return {k: v/max(nb,1) for k,v in M.items()}


# ─── Validate ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def validate(model, loader, criterion, device, use_amp=False):
    model.eval()
    M = {"total":0.,"detection":0.,"segmentation":0.,"severity":0.}
    dc,dt,sc,st = 0,0,0,0
    nb = 0
    amp_enabled = bool(use_amp and device.type == "cuda")
    for batch in tqdm(loader, desc="  Val", leave=False):
        imgs     = batch["image"].to(device)
        poly_lbl = batch["polyp_label"].to(device)
        uc_grade = batch["uc_grade"].to(device)
        mask     = batch["mask"].to(device)
        amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled)
        with amp_ctx:
            out  = model(imgs)
            tgt  = {"polyp_label":poly_lbl,"uc_grade":uc_grade,"mask":mask,"bbox":None}
            L    = criterion(out, tgt)
        for k in M: M[k] += L[k].item() if k in L else 0
        vd = poly_lbl >= 0
        if vd.sum() > 0:
            p = (torch.sigmoid(out["detection_logit"].squeeze(1)) > 0.5).long()
            dc += (p[vd]==poly_lbl[vd]).sum().item(); dt += vd.sum().item()
        vs = uc_grade >= 0
        if vs.sum() > 0:
            p = out["severity_logit"][vs].argmax(1)
            sc += (p==uc_grade[vs]).sum().item(); st += vs.sum().item()
        nb += 1
    avg = {k:v/max(nb,1) for k,v in M.items()}
    avg["det_acc"] = dc/max(dt,1); avg["sev_acc"] = sc/max(st,1)
    return avg


# ─── MC Dropout inference ─────────────────────────────────────────────────────
def enable_mc_dropout(model):
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout): m.train()


@torch.no_grad()
def mc_dropout_forward(model, image_tensor, n_passes=10):
    enable_mc_dropout(model)
    det_p, sev_p, msk_p = [], [], []
    for _ in range(n_passes):
        out = model(image_tensor)
        det_p.append(torch.sigmoid(out["detection_logit"]).squeeze().item())
        sev_p.append(F.softmax(out["severity_logit"],dim=-1).squeeze().cpu().numpy())
        msk_p.append(torch.sigmoid(out["mask_logit"]).squeeze().cpu().numpy())
    da, sa, ma = np.array(det_p), np.array(sev_p), np.array(msk_p)
    dm, ds = float(da.mean()), float(da.std())
    sm, ss = sa.mean(0), sa.std(0)
    mm     = ma.mean(0)
    return {"det_prob_mean": dm, "det_prob_std": ds, "det_label": int(dm>0.5),
            "sev_prob_mean": sm, "sev_prob_std": ss, "sev_label": int(sm.argmax()),
            "mask_mean": mm, "mask_binary": (mm>0.5).astype(np.uint8),
            "uncertainty_score": float(max(ds, ss.max()))}


# ─── Grad-CAM ────────────────────────────────────────────────────────────────
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model; self.grads = None; self.acts = None; self._hooks = []
        self._hooks.append(target_layer.register_forward_hook(
            lambda m,i,o: setattr(self,'acts',o.detach())))
        self._hooks.append(target_layer.register_full_backward_hook(
            lambda m,gi,go: setattr(self,'grads',go[0].detach())))

    def remove(self):
        for h in self._hooks: h.remove()

    def generate(self, img_tensor, target_class=None, head="detection"):
        self.model.eval(); img_tensor = img_tensor.requires_grad_(True)
        out = self.model(img_tensor)
        score = out["detection_logit"].squeeze() if head=="detection" else out["severity_logit"][0, target_class if target_class is not None else out["severity_logit"].argmax(1).item()]
        self.model.zero_grad(); score.backward()
        w   = self.grads.mean(dim=[2,3], keepdim=True)
        cam = F.relu((w * self.acts).sum(1)).squeeze().cpu().numpy()
        return cam / cam.max() if cam.max() > 0 else cam

    def overlay(self, img_np, cam, alpha=0.4):
        H, W = img_np.shape[:2]
        h    = cv2.applyColorMap((cv2.resize(cam,(W,H))*255).astype(np.uint8), cv2.COLORMAP_JET)
        return (alpha * cv2.cvtColor(h, cv2.COLOR_BGR2RGB) + (1-alpha)*img_np).astype(np.uint8)


def predict_single(model, image_tensor, image_np=None, n_mc_passes=10, device=None):
    if device is None: device = next(model.parameters()).device
    image_tensor = image_tensor.to(device)
    mc = mc_dropout_forward(model, image_tensor, n_mc_passes)
    gc = GradCAM(model, model.film)
    model.train()
    for m in model.modules():
        if isinstance(m,(nn.BatchNorm2d,nn.BatchNorm1d)): m.eval()
    it = image_tensor.clone().detach().requires_grad_(True)
    cam_det = gc.generate(it, head="detection")
    cam_sev = gc.generate(it, target_class=mc["sev_label"], head="severity")
    gc.remove()
    ov_det = gc.overlay(image_np, cam_det) if image_np is not None else None
    ov_sev = gc.overlay(image_np, cam_sev) if image_np is not None else None
    return {**mc, "cam_detection": cam_det, "cam_severity": cam_sev,
            "overlay_det": ov_det, "overlay_sev": ov_sev,
            "flag_for_review": mc["uncertainty_score"] > 0.2}


# ─── Test set evaluation ──────────────────────────────────────────────────────
def evaluate_test_set(model, loader, device, n_mc_passes=10):
    enable_mc_dropout(model)
    dp,dpr,dt_,sp,st_,ious,uncs = [],[],[],[],[],[],[]
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            imgs     = batch["image"].to(device)
            poly_lbl = batch["polyp_label"]
            uc_grade = batch["uc_grade"]
            gt_mask  = batch["mask"].detach().cpu().numpy()
            dp_p,sp_p,mp_p = [],[],[]
            for _ in range(n_mc_passes):
                out = model(imgs)
                dp_p.append(torch.sigmoid(out["detection_logit"]).cpu())
                sp_p.append(F.softmax(out["severity_logit"],dim=-1).cpu())
                mp_p.append(torch.sigmoid(out["mask_logit"]).cpu())
            dm = torch.stack(dp_p).mean(0).squeeze(1).numpy()
            sm = torch.stack(sp_p).mean(0).numpy()
            mm = torch.stack(mp_p).mean(0).numpy()
            ds = torch.stack(dp_p).std(0).squeeze(1).numpy()
            ss = torch.stack(sp_p).std(0).numpy()
            for i in range(imgs.size(0)):
                if poly_lbl[i].item()>=0:
                    dpr.append(float(dm[i])); dp.append(int(dm[i]>0.5)); dt_.append(int(poly_lbl[i]))
                if uc_grade[i].item()>=0:
                    sp.append(int(sm[i].argmax())); st_.append(int(uc_grade[i]))
                pb = (mm[i,0]>0.5).astype(np.uint8); gb = (gt_mask[i,0]>0.5).astype(np.uint8)
                if gb.sum()>0:
                    inter = (pb&gb).sum(); union = (pb|gb).sum()
                    ious.append(float(inter/max(union,1)))
                uncs.append(float(max(ds[i], ss[i].max())))
    print("\n" + "="*55)
    print("DETECTION")
    if dt_:
        acc = np.mean(np.array(dp)==np.array(dt_))
        f1  = f1_score(dt_,dp,average="binary",zero_division=0)
        try: auc = roc_auc_score(dt_,dpr)
        except: auc = float("nan")
        print(f"  Accuracy:{acc:.4f}  F1:{f1:.4f}  AUC:{auc:.4f}")
    print("SEVERITY")
    if st_:
        print(f"  Accuracy:{np.mean(np.array(sp)==np.array(st_)):.4f}  "
              f"F1:{f1_score(st_,sp,average='macro',zero_division=0):.4f}")
        sev_labels = [0, 1, 2]
        sev_names = ["G0-1/1", "G2", "G3"]
        print(classification_report(st_, sp, labels=sev_labels,
                                    target_names=sev_names, zero_division=0))
        print(confusion_matrix(st_, sp, labels=sev_labels))
    print("SEGMENTATION")
    if ious: print(f"  Mean IoU:{np.mean(ious):.4f}  n={len(ious)}")
    print(f"UNCERTAINTY  mean:{np.mean(uncs):.4f}  "
          f"flagged:{sum(u>0.2 for u in uncs)}/{len(uncs)}")
    print("="*55)
