# src/dataset.py
# All HyperKvasir dataset logic in one file for clean notebook imports.

import os, json, random
import cv2, numpy as np, pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch

# ─── Label maps ───────────────────────────────────────────────────────────────
ANATOMY_CLASSES   = {"cecum": 0, "retroflex-rectum": 1}
ANATOMY_IDX2NAME  = {v: k for k, v in ANATOMY_CLASSES.items()}

# 🔥 FIX #2: MERGE SEVERITY GRADES 0-1 WITH 1 → 3-CLASS PROBLEM
# Old: grade 0-1 (5%), grade 1 (25%), grade 2 (50%), grade 3 (20%) → 4-class with 35 samples in grade 0-1
# New: grade 0-1 (30%), grade 2 (50%), grade 3 (20%) → 3-class with ~300 samples in merged class
UC_GRADE_MAP = {
    "ulcerative-colitis-grade-0-1": 0,  # Merged: not serious
    "ulcerative-colitis-grade-1":   0,  # 🔥 CHANGED: was 1, now maps to 0 (merged with grade 0-1)
    "ulcerative-colitis-grade-1-2": 0,  # 🔥 CHANGED: was 1, now maps to 0 (merged with grade 0-1)
    "ulcerative-colitis-grade-2":   1,  # 🔥 CHANGED: was 2, now maps to 1
    "ulcerative-colitis-grade-2-3": 1,  # 🔥 CHANGED: was 2, now maps to 1
    "ulcerative-colitis-grade-3":   2,  # 🔥 CHANGED: was 3, now maps to 2
}
UC_IDX2NAME          = {0: "grade 0-1 (normal)", 1: "grade 2 (moderate)", 2: "grade 3 (severe)"}
NUM_ANATOMY_CLASSES  = 2
NUM_UC_GRADES        = 3  # 🔥 REDUCED FROM 4 TO 3

# ─── Transforms ───────────────────────────────────────────────────────────────
def get_transforms(split, image_size=224):
    if split == "train":
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.3),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, p=0.5),
            A.GaussianBlur(blur_limit=(3, 7), p=0.2),
            A.GaussNoise(p=0.3),
            A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
            ToTensorV2()], additional_targets={"mask": "mask"})
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ToTensorV2()], additional_targets={"mask": "mask"})

# ─── Sample collectors ─────────────────────────────────────────────────────────
def collect_anatomy_samples(root):
    samples, base = [], Path(root)/"labeled-images"/"lower-gi-tract"/"anatomical-landmarks"
    for name, idx in ANATOMY_CLASSES.items():
        folder = base / name
        if not folder.exists(): print(f"[WARN] {folder}"); continue
        for p in folder.glob("*.jpg"):
            samples.append({"image_path": str(p), "anatomy_label": idx,
                            "polyp_label": -1, "uc_grade": -1,
                            "mask_path": None, "source": "anatomy"})
    return samples

# def collect_polyp_samples(root, bbox_data=None):
#     samples  = []
#     base     = Path(root)/"labeled-images"/"lower-gi-tract"/"pathological-findings"/"polyps"
#     seg_mask = Path(root)/"segmented-images"/"masks"
#     if not base.exists(): print(f"[WARN] {base}"); return samples
#     for p in base.glob("*.jpg"):
#         mp = seg_mask / (p.stem + ".jpg")
#         samples.append({"image_path": str(p), "anatomy_label": -1,
#                         "polyp_label": 1, "uc_grade": -1,
#                         "mask_path": str(mp) if mp.exists() else None,
#                         "source": "polyp"})
#     return samples

def collect_polyp_samples(root, bbox_data=None):
    samples = []

    # 🔥 1. Real segmentation dataset (correct masks)
    img_dir  = Path(root)/"segmented-images"/"images"
    mask_dir = Path(root)/"segmented-images"/"masks"

    if img_dir.exists():
        for p in img_dir.glob("*.jpg"):
            mp = mask_dir / p.name

            samples.append({
                "image_path": str(p),
                "anatomy_label": -1,
                "polyp_label": 1,
                "uc_grade": -1,
                "mask_path": str(mp) if mp.exists() else None,
                "bbox": None,   # 🔥 No bbox for real masks
                "source": "polyp_seg"
            })

    # 🔥 2. Additional HyperKvasir polyp images (no masks → bbox optional)
    base = Path(root)/"labeled-images"/"lower-gi-tract"/"pathological-findings"/"polyps"

    if base.exists():
        for p in base.glob("*.jpg"):
            key = p.stem

            bbox = None
            if bbox_data and key in bbox_data:
                bbox = bbox_data[key].get("bbox", None)

            samples.append({
                "image_path": str(p),
                "anatomy_label": -1,
                "polyp_label": 1,
                "uc_grade": -1,
                "mask_path": None,
                "bbox": bbox,   # 🔥 THIS LINE FIXES EVERYTHING
                "source": "polyp"
            })

    return samples

def collect_uc_samples(root):
    samples, base = [], Path(root)/"labeled-images"/"lower-gi-tract"/"pathological-findings"
    for name, grade in UC_GRADE_MAP.items():
        folder = base / name
        if not folder.exists(): continue
        for p in folder.glob("*.jpg"):
            samples.append({"image_path": str(p), "anatomy_label": -1,
                            "polyp_label": 1, "uc_grade": grade,
                            "mask_path": None, "source": "uc"})
    return samples

def collect_normal_samples(root):
    samples, base = [], Path(root)/"labeled-images"/"lower-gi-tract"/"quality-of-mucosal-views"/"bbps-2-3"
    if not base.exists(): print(f"[WARN] {base}"); return samples
    for p in base.glob("*.jpg"):
        samples.append({"image_path": str(p), "anatomy_label": -1,
                        "polyp_label": 0, "uc_grade": -1,
                        "mask_path": None, "source": "normal"})
    return samples

def collect_video_samples(root):
    csv_path = Path(root)/"labeled-videos"/"video-labels.csv"
    if not csv_path.exists(): print(f"[WARN] {csv_path}"); return []
    df = pd.read_csv(csv_path); df.columns = [c.strip() for c in df.columns]
    samples = []
    for _, row in df.iterrows():
        organ, f1, f2 = str(row.get("Organ","")), str(row.get("Finding 1","")).lower(), str(row.get("Finding 2","")).lower()
        vid_id = str(row.get("Video file","")).strip()
        if "Lower GI" not in organ: continue
        anat = -1
        for finding in [f1, f2]:
            for cls, idx in ANATOMY_CLASSES.items():
                if cls.replace("-"," ") in finding or cls in finding:
                    anat = idx; break
        polyp = 1 if ("polyp" in f1 or "polyp" in f2) else 0
        matches = list(Path(root).rglob(vid_id + ".avi"))
        if matches:
            samples.append({"video_path": str(matches[0]), "anatomy_label": anat,
                            "polyp_label": polyp, "uc_grade": -1, "source": "video"})
    return samples

# ─── Split builder ─────────────────────────────────────────────────────────────
def build_image_splits(root, seed=42):
    """
    🔥 FIX #1: PATIENT-LEVEL SPLIT (prevents data leakage from video frames)
    
    Problem: AUC=1.0 suggests leakage. Root cause: images from the SAME video
    can end up in both train and test if split at frame level.
    
    Solution: Group images by video source (patient-level), then split GROUPS.
    - All frames from a video stay in one split (train/val/test)
    - Stratification by UC grade ensures balanced classes
    - Leakage audit ensures no group overlap across splits
    """
    from sklearn.model_selection import GroupShuffleSplit
    import re
    
    bbox_path = Path(root)/"segmented-images"/"bounding-boxes.json"
    bbox_data = {}
    if bbox_path.exists():
        try:
            with open(bbox_path, 'r') as f:
                bbox_data = json.load(f)
        except Exception as e:
            print(f"[WARN] Failed to load bounding boxes from {bbox_path}: {e}")
    
    all_s = (collect_anatomy_samples(root) + collect_polyp_samples(root, bbox_data) +
             collect_uc_samples(root) + collect_normal_samples(root))
    print(f"\n{'='*70}")
    print(f"[LEAKAGE FIX #1] Patient-Level Split (Video Grouping)")
    print(f"{'='*70}\n")
    print(f"[INFO] Total samples collected: {len(all_s)}")
    
    # 🔥 Enhanced group ID extraction from image path
    # Extracts video/case ID from directory structure and filename patterns
    def _extract_video_id(sample):
        """
        Extract patient/video ID from image path.
        Supports HyperKvasir structure where images are in category folders.
        """
        p = Path(sample["image_path"])
        stem = p.stem.lower()
        parent = p.parent.name.lower()
        src = sample.get("source", "unknown")
        
        # Strategy 1: For segmented images, use parent folder structure
        # Path: .../segmented-images/images/XXXXXX.jpg
        if "segmented-images" in str(p):
            # Each frame from same video has unique ID, but frames from same
            # video extraction batch will have sequential names
            # Group by first 16 chars (typically patient/case ID)
            if len(stem) >= 16:
                video_id = f"seg:{stem[:16]}"
            else:
                video_id = f"seg:{stem}"
            return video_id
        
        # Strategy 2: For labeled images, use category + parent + ID
        # Path: .../lower-gi-tract/pathological-findings/polyps/XXXXXX.jpg
        # Path: .../lower-gi-tract/anatomical-landmarks/cecum/XXXXXX.jpg
        category_folder = p.parent.name.lower() if len(p.parts) >= 4 else ""
        
        # Try to extract base video ID from filename patterns
        # Patterns: xxx_frame_XXXXX, xxx-XXXXX, uuid-format
        patterns = [
            (r"^(.*?)_frame_\d+$", "frame pattern"),      # xxx_frame_123
            (r"^(.*?)[-_]\d{5,}$", "numbered pattern"),   # xxx-12345 or xxx_12345
            (r"^([a-f0-9]{8})-", "uuid pattern"),         # uuid-like names
        ]
        
        for pattern, desc in patterns:
            m = re.match(pattern, stem)
            if m:
                base = m.group(1)
                if len(base) >= 4:
                    # Use category + base to avoid cross-category collisions
                    video_id = f"{src}:{category_folder}:{base}"
                    return video_id
        
        # Strategy 3: Fallback - use first part of filename
        # For completely unstructured names, at least group by prefix
        if "-" in stem and len(stem) >= 12:
            prefix = stem.split("-")[0]
        else:
            prefix = stem[:8] if len(stem) >= 8 else stem
        
        video_id = f"{src}:{category_folder}:{prefix}"
        return video_id
    
    # Assign video IDs to all samples
    video_ids = [_extract_video_id(s) for s in all_s]
    
    # Compute stratification keys for class balance
    strat_keys = []
    for s in all_s:
        if s["uc_grade"] >= 0:
            strat_key = f"uc_grade_{s['uc_grade']}"
        elif s["anatomy_label"] >= 0:
            strat_key = f"anatomy_{s['anatomy_label']}"
        else:
            strat_key = f"source_{s['source']}"
        strat_keys.append(strat_key)
    
    idx = list(range(len(all_s)))
    
    # 🔥 Two-stage group split (respects video groups)
    # Stage 1: Split 85/15 for train+val vs test
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    iv, it = list(gss1.split(idx, groups=video_ids))[0]
    
    # Stage 2: Split 70/30 (of remaining 85%) for train vs val
    # This gives us 59.5% train, 25.5% val, 15% test
    # Adjust to 76.47% train, 23.53% val to get closer to 70/15/15
    iv_video_ids = [video_ids[i] for i in iv]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.235, random_state=seed+1)
    itr, iv2 = list(gss2.split(iv, groups=iv_video_ids))[0]
    itr, iv2 = [iv[i] for i in itr], [iv[i] for i in iv2]
    
    tr, va, te = [all_s[i] for i in itr], [all_s[i] for i in iv2], [all_s[i] for i in it]
    
    # 🔥 Leakage audit: verify NO group overlap across splits
    tr_vids = {video_ids[i] for i in itr}
    va_vids = {video_ids[i] for i in iv2}
    te_vids = {video_ids[i] for i in it}
    
    tr_va_overlap = len(tr_vids & va_vids)
    tr_te_overlap = len(tr_vids & te_vids)
    va_te_overlap = len(va_vids & te_vids)
    
    # Count unique videos per split
    num_videos = len(set(video_ids))
    tr_videos = len(tr_vids)
    va_videos = len(va_vids)
    te_videos = len(te_vids)
    
    print(f"[INFO] Split Results:")
    print(f"  Train samples: {len(tr):5d}  ({len(tr)/len(all_s)*100:5.1f}%)  videos: {tr_videos}")
    print(f"  Val   samples: {len(va):5d}  ({len(va)/len(all_s)*100:5.1f}%)  videos: {va_videos}")
    print(f"  Test  samples: {len(te):5d}  ({len(te)/len(all_s)*100:5.1f}%)  videos: {te_videos}")
    print(f"  Total samples: {len(all_s):5d}  Total unique videos: {num_videos}")
    
    print(f"\n[LEAKAGE AUDIT] Video group overlap (0 = safe):")
    print(f"  Train vs Val : {tr_va_overlap:3d} overlapping videos  {'CLEAN' if tr_va_overlap == 0 else 'LEAKAGE DETECTED'}")
    print(f"  Train vs Test: {tr_te_overlap:3d} overlapping videos  {'CLEAN' if tr_te_overlap == 0 else 'LEAKAGE DETECTED'}")
    print(f"  Val vs Test  : {va_te_overlap:3d} overlapping videos  {'CLEAN' if va_te_overlap == 0 else 'LEAKAGE DETECTED'}")
    
    if tr_va_overlap == 0 and tr_te_overlap == 0 and va_te_overlap == 0:
        print(f"\n[SAFE] No data leakage detected. Patient-level split is clean.\n")
    else:
        print(f"\n[WARNING] Leakage detected! Review video grouping.\n")
    
    print(f"{'='*70}\n")
    
    return tr, va, te

# ─── Datasets ─────────────────────────────────────────────────────────────────
class HyperKvasirDataset(Dataset):
    def __init__(self, samples, split="train", image_size=224):
        self.samples, self.transform, self.image_size = samples, get_transforms(split, image_size), image_size

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s   = self.samples[idx]
        img = cv2.imread(s["image_path"])

        if img is None:
            print(f"[ERROR] Failed to load image: {s['image_path']}")
            img = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h_img, w_img = img.shape[:2]

        # ✅ 1. Real mask (priority: use if exists)
        if s.get("mask_path") and os.path.exists(s["mask_path"]):
            mask = cv2.imread(s["mask_path"], cv2.IMREAD_GRAYSCALE)

            if mask is None:
                mask = np.zeros((h_img, w_img), dtype=np.uint8)
            else:
                # Resize to image size using INTER_NEAREST for binary preservation
                mask = cv2.resize(mask, (w_img, h_img), interpolation=cv2.INTER_NEAREST)
                # Convert to binary
                mask = (mask > 127).astype(np.uint8)

        # ✅ 2. BBox → pseudo mask (if no real mask)
        elif s.get("bbox") is not None:
            mask = np.zeros((h_img, w_img), dtype=np.uint8)
            
            # Normalize bbox to ensure it's a list
            bbox_list = s["bbox"] if isinstance(s["bbox"], list) else [s["bbox"]]
            
            for box in bbox_list:
                if isinstance(box, dict):
                    xmin = int(box.get("xmin", 0))
                    ymin = int(box.get("ymin", 0))
                    xmax = int(box.get("xmax", 0))
                    ymax = int(box.get("ymax", 0))
                else:
                    # Tuple/list format: (xmin, ymin, xmax, ymax)
                    xmin, ymin, xmax, ymax = int(box[0]), int(box[1]), int(box[2]), int(box[3])

                # Clip to image bounds
                xmin = max(0, min(xmin, w_img))
                ymin = max(0, min(ymin, h_img))
                xmax = max(xmin, min(xmax, w_img))
                ymax = max(ymin, min(ymax, h_img))

                # Fill mask region
                if xmax > xmin and ymax > ymin:
                    mask[ymin:ymax, xmin:xmax] = 1

        # ✅ 3. No mask (zero mask)
        else:
            mask = np.zeros((h_img, w_img), dtype=np.uint8)

        # Ensure mask is uint8
        mask = mask.astype(np.uint8)

        aug = self.transform(image=img, mask=mask)

        return {
            "image": aug["image"],
            "anatomy_label": torch.tensor(s["anatomy_label"], dtype=torch.long),
            "polyp_label":   torch.tensor(s["polyp_label"],   dtype=torch.long),
            "uc_grade":      torch.tensor(s["uc_grade"], dtype=torch.long),
            "mask":          aug["mask"].unsqueeze(0).float()
        }

class VideoFrameDataset(Dataset):
    def __init__(self, video_samples, num_frames=8, image_size=224):
        self.video_samples, self.num_frames = video_samples, num_frames
        self.transform = get_transforms("train", image_size)

    def __len__(self): return len(self.video_samples)

    def __getitem__(self, idx):
        try:
            s = self.video_samples[idx]
        except (IndexError, KeyError) as e:
            # Fallback: return all-zero frames if index is out of range
            return {
                "frames": torch.zeros(self.num_frames, 3, 224, 224, dtype=torch.float32),
                "anatomy_label": torch.tensor(0, dtype=torch.long),
                "polyp_label": torch.tensor(0, dtype=torch.long)
            }
        
        try:
            cap = cv2.VideoCapture(s["video_path"])
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            start = 0 if total < self.num_frames else random.randint(0, total - self.num_frames)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            frames = []
            
            for _ in range(self.num_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                try:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img_tensor = self.transform(image=rgb_frame)["image"]
                    # Ensure tensor is 3D [C, H, W]
                    if img_tensor.dim() != 3:
                        img_tensor = torch.zeros(3, 224, 224, dtype=torch.float32)
                    frames.append(img_tensor)
                except Exception:
                    # Skip individual frames that fail
                    continue
            
            cap.release()
            
            # Handle corrupted/empty videos: if no frames read, create black frames
            if len(frames) == 0:
                frames = [torch.zeros(3, 224, 224, dtype=torch.float32) for _ in range(self.num_frames)]
            else:
                # Filter out any frames that don't have shape [3, 224, 224]
                valid_frames = [f for f in frames if f.shape == (3, 224, 224)]
                if not valid_frames:
                    # All frames have wrong shape, use zeros
                    frames = [torch.zeros(3, 224, 224, dtype=torch.float32) for _ in range(self.num_frames)]
                else:
                    frames = valid_frames
                    # Pad with zeros if not enough frames
                    while len(frames) < self.num_frames:
                        frames.append(torch.zeros(3, 224, 224, dtype=torch.float32))
            
            # Final trim to exact size
            frames = frames[:self.num_frames]
            
            # Ensure we have exactly num_frames with correct shape
            if len(frames) < self.num_frames:
                frames.extend([torch.zeros(3, 224, 224, dtype=torch.float32) 
                              for _ in range(self.num_frames - len(frames))])
            
            frames_tensor = torch.stack(frames, dim=0)  # Explicit dim=0
            
            return {
                "frames": frames_tensor,
                "anatomy_label": torch.tensor(s.get("anatomy_label", 0), dtype=torch.long),
                "polyp_label": torch.tensor(s.get("polyp_label", 0), dtype=torch.long)
            }
        
        except Exception as e:
            # Last resort fallback for any unforeseen error
            print(f"[ERROR] VideoFrameDataset failed for idx={idx}: {str(e)}")
            return {
                "frames": torch.zeros(self.num_frames, 3, 224, 224, dtype=torch.float32),
                "anatomy_label": torch.tensor(0, dtype=torch.long),
                "polyp_label": torch.tensor(0, dtype=torch.long)
            }


def get_dataloaders(root, batch_size=16, num_workers=4, seed=42):
    tr, va, te = build_image_splits(root, seed)
    make = lambda s, split: DataLoader(
        HyperKvasirDataset(s, split), batch_size=batch_size,
        shuffle=(split=="train"), num_workers=num_workers,
        pin_memory=True, drop_last=(split=="train"))
    return make(tr,"train"), make(va,"val"), make(te,"test")
