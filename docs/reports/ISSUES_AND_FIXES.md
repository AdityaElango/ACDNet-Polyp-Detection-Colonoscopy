# ACDNet Pipeline — Issues & Fixes Guide

## Overview
This document catalogs all identified issues in the ACDNet training pipeline, their root causes, impact, and recommended fixes. Each issue is mapped to the relevant cells in `ACDNet_Pipeline.ipynb`.

---

## 🔴 CRITICAL ISSUES (Must Fix Before Deployment)

### Issue #1: Non-Stratified Dataset Split (Class Imbalance in Test Set)

**Severity:** 🔴 CRITICAL  
**Cell:** Cell 3 (`build_image_splits()`)  
**File:** `src/dataset.py`

#### Problem
- Current split: Random 70/15/15 without stratification by UC grade
- Result: Test set severely imbalanced
  - G2 (Normal colons): 59.4% (486 images)
  - G1 (Mild UC): 23.4% (191 images)
  - G3 (Moderate UC): 15.6% (127 images)
  - G0-1 (Severe UC): 1.6% (12 images)
- Expected (stratified): ~25% per class
- Impact: Model "learns" to always predict G2 (majority class), inflating apparent accuracy

#### How to Fix
1. **In `src/dataset.py` — Modify `build_image_splits()` function:**
   ```python
   # Before (WRONG):
   train_val, test = train_test_split(samples, test_size=0.15, random_state=seed)
   train, val = train_test_split(train_val, test_size=0.176, random_state=seed)
   
   # After (CORRECT):
   uc_grades = [s['uc_grade'] for s in samples]  # Extract grades for stratification
   train_val, test = train_test_split(
       samples, test_size=0.15, random_state=seed, stratify=uc_grades
   )
   train, val = train_test_split(
       train_val, test_size=0.176, random_state=seed, 
       stratify=[s['uc_grade'] for s in train_val]
   )
   ```

2. **Re-run Cell 3** to rebuild splits with stratification

3. **Then re-run Cells 6, 7, 8** to retrain and evaluate with balanced test set

#### Expected Outcome
- Test set: ~25% per grade (balanced)
- Per-class accuracy improvement: +3–8%
- More reliable accuracy metrics (severity grades equally weighted)

---

### Issue #2: Suboptimal Loss Weighting (Detection Overweighted)

**Severity:** 🔴 CRITICAL  
**Cell:** Cell 6 (Training configuration)  
**File:** `notebooks/ACDNet_Pipeline.ipynb` — Cell 6

#### Problem
- Current weights: `lam_det=1.0, lam_seg=0.5, lam_sev=1.0`
- Detection saturates at 100% early → only detection gradients flow
- Segmentation weighted at 0.5 → receives weak supervision
- Result:
  - Detection: 100% (overfitted/inflated)
  - Segmentation: Only 57.8% IoU (severely undertrained)
  - Severity: 71% (okay, but could be better)

#### How to Fix
1. **In Cell 6 — Change loss weights:**
   ```python
   # Before (WRONG):
   criterion = ACDNetLoss(
       lam_det=1.0, lam_seg=0.5, lam_sev=1.0,
       lam_temp=0.1, pos_weight=3.0
   )
   
   # After (CORRECT):
   # Inverse weighting by current performance
   criterion = ACDNetLoss(
       lam_det=0.5,    # Reduce (already at 100%)
       lam_seg=1.5,    # Increase (weak at 57.8%)
       lam_sev=1.0,    # Keep (solid at 71%)
       lam_temp=0.1, 
       pos_weight=3.0
   )
   ```

2. **Re-run Cell 6** to retrain with new weights

#### Expected Outcome
- Segmentation IoU: +10–15% improvement (57.8% → 68%+)
- Detection: -1–2% drop (100% → 98–99%, more realistic)
- Severity: Slight improvement or stable

---

### Issue #3: Wrong Checkpoint Saved (Epoch 20 < Epoch 12)

**Severity:** 🔴 CRITICAL  
**Cell:** Cell 6 (best checkpoint selection), Cell 7 (loading)  
**File:** `notebooks/ACDNet_Pipeline.ipynb` — Cell 6, `checkpoints/acdnet_best.pth`

#### Problem
- Checkpoint saved at Epoch 20 (current loaded state)
- Epoch 20 metrics: val_det=100%, val_sev=68.5%, combined=88.00%
- Epoch 12 metrics: val_det=100%, val_sev=71.65%, combined=88.66% ← Better!
- Training shows loss increasing Epoch 12→20 (overfitting signal)
- Severity accuracy drops 71.65% → 68.5% (overfitting on severity)
- Currently using suboptimal checkpoint

#### How to Fix

**Option A: Use early stopping (prevent Epoch 20 save)**
1. In Cell 6, add early stopping logic:
   ```python
   # Add after scheduler.step() in training loop:
   if epoch > 12 and combined < best_combined - 0.001:
       print(f"Early stopping at epoch {epoch} (no improvement in 8 epochs)")
       break
   ```

**Option B: If you still have training logs, manually reload Epoch 12**
1. Check if you saved epoch-wise checkpoints (not just best)
2. Load checkpoint from Epoch 12 manually in Cell 7:
   ```python
   # Modify Cell 7 checkpoint loading:
   # ACDNET_CKPT should point to epoch_12_checkpoint.pth instead
   ```

**Option C: Re-run training with both fixes**
1. Apply Issue #1 fix (stratified split)
2. Apply Issue #2 fix (loss weighting)
3. Add early stopping condition above
4. Re-run Cell 6 (will naturally stop at Epoch 12)

#### Expected Outcome
- Use checkpoint with 88.66% combined accuracy (vs 88.00%)
- Better generalization (less overfitting)
- Improvement: +0.6% on test set

---

### Issue #4: Weak Segmentation Supervision (74% Zero Masks)

**Severity:** 🔴 CRITICAL  
**Cell:** Cell 3 (data preparation)  
**File:** `src/dataset.py` — `collect_polyp_samples()`

#### Problem
- Training set: 3,792 polyp samples total
- Real segmentation masks: Only 1,000 (26.4%)
- Zero masks (no supervision): 2,792 (73.6%)
- Impact: Segmentation head receives weak supervision on 74% of batches
- Result: Only 57.8% IoU (poor segmentation quality)

#### How to Fix
Generate synthetic masks from existing data:

1. **In `src/dataset.py` — Create synthetic mask generator:**
   ```python
   def generate_synthetic_mask_from_bbox(image, bbox, mask_size=224):
       """
       Generate a synthetic mask from bounding box.
       Args: image (H,W,3), bbox (dict with xmin,ymin,xmax,ymax)
       Returns: mask (224,224) with 1s in bbox region, 0s elsewhere
       """
       mask = np.zeros((mask_size, mask_size), dtype=np.float32)
       x1 = int(bbox['xmin'] * mask_size / image.shape[1])
       y1 = int(bbox['ymin'] * mask_size / image.shape[0])
       x2 = int(bbox['xmax'] * mask_size / image.shape[1])
       y2 = int(bbox['ymax'] * mask_size / image.shape[0])
       mask[y1:y2, x1:x2] = 1.0
       return mask
   ```

2. **In `collect_polyp_samples()` — Use synthetic masks when real masks unavailable:**
   ```python
   # When creating polyp samples:
   if mask_path is None and bbox is not None:
       sample['mask_source'] = 'synthetic'
       sample['bbox'] = bbox  # Store for mask generation in Dataset
   elif mask_path is None:
       sample['mask_source'] = 'zero'  # Fallback to zero mask
   else:
       sample['mask_source'] = 'real'
   ```

3. **In `HyperKvasirDataset.__getitem__()` — Generate mask on-the-fly:**
   ```python
   if sample['mask_source'] == 'synthetic':
       mask = generate_synthetic_mask_from_bbox(image, sample['bbox'])
   elif sample['mask_source'] == 'zero':
       mask = np.zeros((224, 224), dtype=np.float32)
   else:  # real
       mask = load_mask_from_file(sample['mask_path'])
   ```

4. **Re-run Cell 3** to apply changes
5. **Re-run Cell 6** to retrain with better supervision

#### Expected Outcome
- Segmentation IoU: +5–10% improvement
- Combined with Issue #2 fix (weight increase): +15–25% total improvement

---

## 🟡 IMPORTANT ISSUES (Should Fix)

### Issue #5: 100% Detection Unrealistic for Medical Data

**Severity:** 🚩 HIGH (Realistic expectation issue)  
**Cell:** Cell 8 (test evaluation), Cell 9 (inference)  
**File:** `notebooks/ACDNet_Pipeline.ipynb` — Cells 8, 9

#### Problem
- Test set detection accuracy: 100% (both detection & F1)
- This is unrealistic for medical imaging
- Typical state-of-the-art polyp detection: 95–98% (real-world variance)
- Possible causes:
  - HyperKvasir dataset may be too clean/standardized
  - Limited equipment/imaging protocol variation
  - Model may have high dataset-specific tuning

#### How to Fix
1. **Validate on external dataset:**
   - Use colonoscopy images from different hospital/equipment
   - Expected drop: 100% → 95–98%
   - Validates real-world generalization

2. **In Cell 8 — Add external validation section:**
   ```python
   # Add after test set evaluation:
   print("\n=== EXTERNAL DATASET VALIDATION ===")
   if external_data_path.exists():
       external_loader = get_external_dataloaders(external_data_path)
       external_metrics = evaluate_test_set(model, external_loader, DEVICE)
       print(f"HyperKvasir detection: 100%")
       print(f"External dataset detection: {external_metrics['det_acc']*100:.1f}%")
   else:
       print("No external data available. Recommendation: acquire validation dataset")
   ```

3. **Collect real-world performance baseline before deployment**

#### Expected Outcome
- Realistic performance estimate: 95–98% detection
- Identifies if model is dataset-specific vs generalizable
- Critical for deployment credibility

---

### Issue #6: Loss Increasing After Epoch 12 (Overfitting Signal)

**Severity:** 🟡 IMPORTANT  
**Cell:** Cell 6 (training loop)  
**File:** `notebooks/ACDNet_Pipeline.ipynb` — Cell 6

#### Problem
- Training loss increases Epoch 12→20 (1.214 → 1.237)
- Severity validation accuracy drops (71.65% → 68.5%)
- Indicates model is overfitting after Epoch 12
- Yet training continues to Epoch 20 (wasteful, degrading performance)

#### How to Fix
Implement early stopping (see Issue #3 Option A for code):
1. Monitor `combined = 0.6 * val_det_acc + 0.4 * val_sev_acc`
2. Stop if no improvement for N epochs (e.g., N=5)
3. In Cell 6:
   ```python
   patience = 5
   epochs_no_improve = 0
   
   for epoch in range(1, MAIN_EPOCHS + 1):
       # ... training code ...
       if combined > best_combined:
           best_combined = combined
           epochs_no_improve = 0
           # ... save checkpoint ...
       else:
           epochs_no_improve += 1
           if epochs_no_improve >= patience:
               print(f"Early stopping at epoch {epoch}")
               break
   ```

#### Expected Outcome
- Training stops at Epoch 12 naturally
- Saves computational time (8 fewer epochs)
- Uses better checkpoint (Issue #3 fixed)

---

### Issue #7: Severe Class Imbalance in Severity (G2 Dominates)

**Severity:** 🟡 IMPORTANT  
**Cell:** Cell 3 (data exploration), Cell 6 (class weighting)  
**File:** `src/dataset.py`

#### Problem
- UC grade distribution (training set):
  - G2 (normal): ~50%
  - G1 (mild): ~25%
  - G3 (moderate): ~20%
  - G0-1 (severe): ~5%
- Model learns to bias toward G2
- Rare classes (G0-1, G3) receive less gradient signal
- Per-class test accuracies vary: G2=79%, G1=70%, G3=50%, G0-1=0%

#### How to Fix
Implement weighted loss for severity classification:

1. **In `src/engine.py` — Add class weighting to severity loss:**
   ```python
   # In ACDNetLoss.forward():
   severity_weights = torch.tensor([0.8, 1.2, 1.1, 2.0]).to(loss_sev.device)
   # Weights for [G0-1, G1, G2, G3]
   # Rare classes (G0-1, G3) weighted higher
   
   sev_loss = F.cross_entropy(sev_logits, sev_labels, weight=severity_weights)
   ```

2. **Adjust weights based on training distribution:**
   - Compute inverse frequency weights
   - G0-1 (5%) → weight 2.0
   - G1 (25%) → weight 1.2
   - G2 (50%) → weight 0.8
   - G3 (20%) → weight 1.1

3. **Re-run Cell 6** with weighted severity loss

#### Expected Outcome
- Per-class accuracy more balanced
- G0-1 accuracy: 0% → 40–50%
- G3 accuracy: 50% → 65–70%
- Overall accuracy: slight improvement or stable

---

## 🟠 OPTIONAL IMPROVEMENTS

### Issue #8: Small Validation Set Size

**Severity:** 🟠 OPTIONAL  
**Cell:** Cell 3 (data split)  
**File:** `src/dataset.py` — `build_image_splits()`

#### Problem
- Validation set: 15% of 3,792 = 569 samples
- UC severity validation: 254 samples (if 15% of 1,694 with labels)
- Small set → high variance in validation metrics
- Example: 1 sample misclassified can change accuracy by 0.4%

#### How to Fix (If you have more data)
1. Adjust split to 70% train / 20% val / 10% test (if dataset grows)
2. Use stratified split (already recommended in Issue #1)
3. Current split is reasonable for 3,792 samples

#### Expected Outcome
- More stable validation metrics
- Less noisy loss curves
- (Low priority since current split is acceptable)

---

### Issue #9: Limited Video Temporal Supervision

**Severity:** 🟠 OPTIONAL  
**Cell:** Cell 6 (video loader)  
**File:** `notebooks/ACDNet_Pipeline.ipynb` — Cell 6, `src/dataset.py`

#### Problem
- Video temporal loss (`lam_temp=0.1`) currently disabled
- Training uses only image-level supervision
- Temporal consistency not enforced between frames
- Could improve robustness to video artifacts

#### How to Fix
1. **Fix video bytecode cache issue:**
   ```python
   # In Cell 6, replace video_loader generation:
   import importlib
   import src.dataset as dataset_module
   importlib.reload(dataset_module)
   
   video_loader = dataset_module.get_video_loader(...)
   ```

2. **Or keep disabled** (current approach is valid):
   - λ=0.1 is low weight (optional)
   - Image supervision is sufficient
   - Temporal loss adds complexity without guaranteed benefit

#### Expected Outcome
- Slight robustness improvement (1–2%)
- More complex training pipeline
- (Low priority, training already stable without it)

---

### Issue #10: No Synthetic Mask Generation for Unlabeled Polyps

**Severity:** 🟠 OPTIONAL  
**Cell:** Cell 3c (BBox inspection)  
**File:** `src/dataset.py`

#### Problem
- 2,792 polyp samples without real masks (74%)
- No synthetic bounding box data loaded
- Currently using zero masks (no segmentation supervision)
- Could improve mask quality with synthetic data

#### How to Fix
See **Issue #4** complete implementation (generates synthetic masks from bboxes or image centers)

---

## 🟢 IMPLEMENTATION PRIORITY MATRIX

| Priority | Issues | Timeline | Expected Gain | Difficulty |
|----------|--------|----------|---------------|------------|
| 🔴 CRITICAL | #1, #2, #3, #4 | 1–2 weeks | +8–25% | Medium |
| 🟡 IMPORTANT | #5, #6, #7 | 1–2 weeks | +1–5% | Low |
| 🟠 OPTIONAL | #8, #9, #10 | <1 week | +0–2% | Low |

### Recommended Execution Order
1. **Fix #1** (stratified split) — Cell 3
2. **Fix #2** (loss weights) — Cell 6
3. **Fix #3** (early stopping) — Cell 6
4. **Fix #4** (synthetic masks) — Cell 3, then Cell 6 retrain
5. **Fix #6** (Epoch 12 checkpoint) — Automatic if #3 done
6. **Fix #5** (external validation) — After retraining
7. **Fixes #7, #8, #9, #10** — Optional, lower priority

---

## 📋 IMPLEMENTATION CHECKLIST

### Phase 1: Critical Fixes (Week 1)
- [ ] Update `build_image_splits()` with `stratify=uc_grade`
- [ ] Change loss weights in Cell 6: det=0.5, seg=1.5, sev=1.0
- [ ] Add early stopping logic to Cell 6
- [ ] Add weighted loss for severity classes
- [ ] Generate synthetic masks or ensure real masks loaded
- [ ] Re-run Cell 3 (data prep)
- [ ] Re-run Cell 6 (training) — should complete at Epoch 12
- [ ] Re-run Cell 7 (load checkpoint)
- [ ] Run Cell 8 (test evaluation)

### Phase 2: Important Fixes (Week 2)
- [ ] Acquire external validation dataset
- [ ] Run Cell 8 with external data
- [ ] Fine-tune loss weights based on new results
- [ ] Test on different medical equipment (if available)

### Phase 3: Optional Improvements (Week 3+)
- [ ] Fix video loader if needed
- [ ] Implement per-class weighted metrics
- [ ] Add confidence calibration
- [ ] Deploy Gradio UI (Cell 10)

---

## 📊 EXPECTED RESULTS AFTER ALL FIXES

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Val combined accuracy | 88.66% | 90–92% | +1.3–3.3% |
| Test detection | 100% | 95–98% | -2–5% (realistic) |
| Test severity | 71.09% | 74–77% | +3–6% |
| Test segmentation IoU | 57.83% | 70–75% | +12–17% |
| G0-1 accuracy | 0% | 40–50% | +40–50% |
| G3 accuracy | 50% | 65–75% | +15–25% |
| Training time | 20 epochs | ~12 epochs | -40% faster |

---

## 🔧 FILES TO MODIFY

1. **`src/dataset.py`**
   - `build_image_splits()` — add stratification
   - `collect_polyp_samples()` — add synthetic mask flags
   - `HyperKvasirDataset.__getitem__()` — generate synthetic masks

2. **`src/engine.py`**
   - `ACDNetLoss.forward()` — add severity class weights
   - `train_one_epoch()` — add early stopping return value

3. **`notebooks/ACDNet_Pipeline.ipynb`**
   - Cell 3 — rebuild splits (re-run)
   - Cell 6 — update loss weights, add early stopping
   - Cell 7 — checkpoint will auto-update
   - Cell 8 — add external validation (optional)

---

## 📝 NOTES

- All fixes are **non-breaking** — won't corrupt existing code
- Apply fixes in priority order for maximum benefit per effort
- Each fix can be tested independently
- Retraining with fixes expected to complete in ~2–3 hours (vs 4+ hours before)
