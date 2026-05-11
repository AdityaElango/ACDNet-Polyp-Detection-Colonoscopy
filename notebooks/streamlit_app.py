import json
import sys
import tempfile
from io import BytesIO
from pathlib import Path

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import torch
import torch.nn.functional as F

st.set_page_config(
    page_title="ACDNet — Colonoscopy Analysis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_ROOT = Path.cwd()
if PROJECT_ROOT.name.lower() == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent
SRC_PATH = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from src.dataset import ANATOMY_IDX2NAME, UC_IDX2NAME  # noqa: E402
from src.engine import enable_mc_dropout, predict_single  # noqa: E402
from src.models import build_acdnet  # noqa: E402

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
ACDNET_CKPT = CHECKPOINT_DIR / "acdnet_best.pth"
ANATOMY_CKPT = CHECKPOINT_DIR / "anatomy_cnn_best.pth"
DEMO_SVG = PROJECT_ROOT / "notebooks" / "assets" / "demo_colonoscopy.svg"


def find_data_root():
    candidates = [
        PROJECT_ROOT / "Dataset",
        PROJECT_ROOT / "hyper_kvasir",
        PROJECT_ROOT.parent / "Dataset",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return PROJECT_ROOT / "Dataset"


@st.cache_resource
def load_model(device_name: str):
    if not ACDNET_CKPT.exists() or not ANATOMY_CKPT.exists():
        st.warning("Model checkpoints not found. Run notebook Cell 4 and Cell 6 to create them.")

    use_cuda = device_name == "cuda" and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = build_acdnet(
        anatomy_checkpoint=str(ANATOMY_CKPT),
        num_uc_grades=len(UC_IDX2NAME),
        embedding_dim=64,
        dropout_p=0.3,
        pretrained_backbone=True,
    ).to(device)

    try:
        state = torch.load(ACDNET_CKPT, map_location=device)
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state, strict=False)
    except Exception as exc:
        st.warning(f"Could not fully load ACDNet checkpoint: {exc}")

    model.eval()
    return model


@st.cache_resource
def get_preprocess():
    return A.Compose(
        [
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )


def video_to_frames_bytes(video_bytes, max_frames=20):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_file.write(video_bytes)
    temp_file.flush()
    temp_file.close()

    cap = cv2.VideoCapture(temp_file.name)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []

    idxs = np.linspace(0, max(0, total - 1), num=min(max_frames, total), dtype=int)
    frames = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok and frame is not None:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def blend_overlay(image_np, overlay_np, alpha=0.55):
    if overlay_np is None:
        return image_np
    base = image_np.astype(np.float32)
    over = overlay_np.astype(np.float32)
    return np.clip((1 - alpha) * base + alpha * over, 0, 255).astype(np.uint8)


def compute_anatomy_probs(model, device, image_tensor):
    with torch.no_grad():
        _, embedding = model.anatomy_cnn(image_tensor.to(device))
        logits = model.anatomy_cnn.classifier(embedding)
        probs = torch.softmax(logits, dim=-1).squeeze(0).detach().cpu().numpy()

    names = [ANATOMY_IDX2NAME.get(i, f"Region {i}") for i in range(len(probs))]
    anatomy_probs = {name: float(prob) for name, prob in zip(names, probs)}
    anatomy_region = max(anatomy_probs, key=anatomy_probs.get) if anatomy_probs else "Unknown"
    return anatomy_probs, anatomy_region


def collect_mc_scores(model, image_tensor, n_passes, device):
    image_tensor = image_tensor.to(device)
    enable_mc_dropout(model)

    det_scores = []
    sev_scores = []
    mask_scores = []

    with torch.no_grad():
        for _ in range(n_passes):
            outputs = model(image_tensor)
            det_scores.append(float(torch.sigmoid(outputs["detection_logit"]).squeeze().item()))
            sev_scores.append(F.softmax(outputs["severity_logit"], dim=-1).squeeze(0).detach().cpu().numpy())
            mask_scores.append(torch.sigmoid(outputs["mask_logit"]).squeeze(0).detach().cpu().numpy())

    det_scores_arr = np.asarray(det_scores, dtype=np.float32)
    sev_scores_arr = np.asarray(sev_scores, dtype=np.float32)
    mask_scores_arr = np.asarray(mask_scores, dtype=np.float32)

    return {
        "mc_scores": det_scores_arr.tolist(),
        "mc_mean": float(det_scores_arr.mean()),
        "mc_var": float(det_scores_arr.var()),
        "sev_prob_mean": sev_scores_arr.mean(axis=0),
        "mask_mean": mask_scores_arr.mean(axis=0),
        "mask_binary": (mask_scores_arr.mean(axis=0) > 0.5).astype(np.uint8),
    }


def run_inference(model, device, image_np, n_mc_passes):
    preprocess = get_preprocess()
    tensor = preprocess(image=image_np)["image"].unsqueeze(0)

    result = predict_single(
        model,
        tensor,
        image_np=image_np,
        n_mc_passes=int(n_mc_passes),
        device=device,
    )
    mc_stats = collect_mc_scores(model, tensor, int(n_mc_passes), device)
    anatomy_probs, anatomy_region = compute_anatomy_probs(model, device, tensor)

    det_prob = float(mc_stats["mc_mean"])
    det_label = int(det_prob >= 0.5)
    det_conf = det_prob if det_label else 1.0 - det_prob

    sev_probs = np.asarray(mc_stats["sev_prob_mean"], dtype=np.float32)
    sev_label = int(sev_probs.argmax())
    high_risk_prob = float(sev_probs[-1]) if len(sev_probs) else 0.0

    return {
        **result,
        **mc_stats,
        "detected": bool(det_label),
        "det_conf": float(det_conf),
        "high_risk": bool(sev_label == len(sev_probs) - 1),
        "sev_score": high_risk_prob,
        "anatomy_region": anatomy_region,
        "anatomy_probs": anatomy_probs,
        "gradcam_img": result["overlay_det"],
        "seg_mask": result["mask_mean"],
        "review_flag": det_conf < 0.70 or mc_stats["mc_var"] >= 0.05,
    }


def get_demo_image(data_root):
    polyp_dir = data_root / "labeled-images" / "lower-gi-tract" / "pathological-findings" / "polyps"
    if polyp_dir.exists():
        imgs = list(polyp_dir.glob("*.jpg"))
        if imgs:
            return np.array(Image.open(imgs[0]).convert("RGB"))
    return None


def render_placeholder_results(show_uncertainty):
    st.subheader("Results")
    st.caption("Run inference to populate detection, severity, anatomy, Grad-CAM, segmentation, and uncertainty outputs.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Detection", "Awaiting inference")
        st.progress(0.0, text="Confidence: --")
    with col2:
        st.metric("Severity", "Awaiting inference")
        st.progress(0.0, text="Severity score: --")
    with col3:
        st.metric("Anatomy location", "Awaiting inference")
        st.progress(0.0, text="Region probability: --")

    col4, col5 = st.columns(2)
    with col4:
        st.subheader("Grad-CAM heatmap")
        st.info("Grad-CAM overlay will appear here after inference.")
    with col5:
        st.subheader("Segmentation mask")
        st.info("Segmentation mask will appear here after inference.")

    if show_uncertainty:
        st.subheader("MC Dropout uncertainty")
        st.info("MC Dropout bar chart will appear here after inference.")


def render_results_panel(results, mc_passes, show_gradcam, show_mask, show_uncertainty, heatmap_blend, anatomy_filter):
    st.subheader("Results")

    if not results:
        render_placeholder_results(show_uncertainty)
        return

    col1, col2, col3 = st.columns(3)

    with col1:
        severity_label = "🟢 Polyp" if results["detected"] else "⚪ Normal"
        st.metric("Detection", severity_label)
        st.progress(results["det_conf"], text=f"Confidence: {results['det_conf']:.0%}")

    with col2:
        sev_badge = "🔴 High Risk" if results["high_risk"] else "🟡 Low Risk"
        st.metric("Severity", sev_badge)
        st.progress(results["sev_score"], text=f"Severity score: {results['sev_score']:.2f}")
        color = "#ff5a5f" if results["high_risk"] else "#f5b942"
        sev_text = "High risk" if results["high_risk"] else "Low risk"
        st.markdown(
            f"<div style='margin-top:0.25rem; padding:0.45rem 0.7rem; border-radius:0.55rem; background:rgba(255,255,255,0.03); border-left:4px solid {color};'>"
            f"<b style='color:{color};'>Severity class:</b> {sev_text}</div>",
            unsafe_allow_html=True,
        )

    with col3:
        st.metric("Anatomy location", results["anatomy_region"])
        if anatomy_filter != "All":
            st.caption(f"Filtered region: {anatomy_filter}")
        for region, prob in results["anatomy_probs"].items():
            if anatomy_filter != "All" and region.lower() != anatomy_filter.lower():
                continue
            st.progress(prob, text=f"{region}: {prob:.0%}")

    col4, col5 = st.columns(2)
    with col4:
        st.subheader("Grad-CAM heatmap")
        if show_gradcam and results.get("gradcam_img") is not None:
            gradcam_img = blend_overlay(results["input_image"], results["gradcam_img"], alpha=heatmap_blend)
            st.image(gradcam_img, use_container_width=True)
        else:
            st.info("Grad-CAM overlay is disabled or unavailable.")

    with col5:
        st.subheader("Segmentation mask")
        if show_mask and results.get("seg_mask") is not None:
            st.image(results["seg_mask"], clamp=True, use_container_width=True, channels="GRAY")
        else:
            st.info("Segmentation mask is disabled or unavailable.")

    if show_uncertainty:
        st.subheader(f"MC Dropout — {mc_passes} inference passes")
        chart_df = pd.DataFrame(
            {
                "pass": [f"P{i + 1}" for i in range(len(results["mc_scores"]))],
                "score": results["mc_scores"],
            }
        ).set_index("pass")
        st.bar_chart(chart_df, height=220)
        st.caption(f"Mean: {results['mc_mean']:.3f}")

        c1, c2, c3 = st.columns(3)
        c1.metric("Mean prediction", f"{results['mc_mean']:.3f}")
        c2.metric("Variance", f"{results['mc_var']:.4f}")
        c3.metric("Uncertainty", "Low" if results["mc_var"] < 0.05 else "High")

    if results.get("is_video") and results.get("frame_timeline") and results.get("frame_predictions"):
        st.subheader("Frame timeline")
        frame_cols = st.columns(min(6, len(results["frame_timeline"])))
        for idx, (frame_img, frame_pred) in enumerate(
            zip(results["frame_timeline"][:6], results["frame_predictions"][:6])
        ):
            with frame_cols[idx]:
                color = "🔴" if frame_pred["high_risk"] else "🟢"
                st.image(frame_img, caption=f"f{idx + 1} {color}", use_container_width=True)

    report = {
        "detection": results["detected"],
        "severity": "high" if results["high_risk"] else "low",
        "confidence": results["det_conf"],
        "anatomy": results["anatomy_region"],
        "mc_mean": results["mc_mean"],
        "mc_variance": results["mc_var"],
    }
    st.download_button(
        "Download prediction report (JSON)",
        data=json.dumps(report, indent=2),
        file_name="acdnet_report.json",
        mime="application/json",
    )


def main():
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(180deg, #101722, #07101b);
            color: #e7eef6;
        }
        .main-banner {
            background: linear-gradient(135deg, rgba(255,255,255,0.08), rgba(255,255,255,0.03));
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 20px;
            padding: 18px 22px;
            margin-bottom: 16px;
        }
        .small-muted { color: #9db0c6; font-size: 0.95rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if "results" not in st.session_state:
        st.session_state["results"] = None
    if "input_image" not in st.session_state:
        st.session_state["input_image"] = None
    if "is_video" not in st.session_state:
        st.session_state["is_video"] = False
    if "frame_timeline" not in st.session_state:
        st.session_state["frame_timeline"] = []
    if "frame_predictions" not in st.session_state:
        st.session_state["frame_predictions"] = []

    st.markdown(
        """
        <div class='main-banner'>
          <h1 style='margin:0;'>ACDNet — Colonoscopy Analysis</h1>
          <div class='small-muted'>Upload a frame or video to inspect multi-task predictions, Grad-CAM, segmentation, and MC Dropout uncertainty.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    data_root = find_data_root()
    demo_image = get_demo_image(data_root)
    svg_demo_available = DEMO_SVG.exists()

    st.sidebar.markdown("### Controls")
    device_options = ["cuda", "cpu"]
    default_index = 0 if torch.cuda.is_available() else 1
    device_choice = st.sidebar.selectbox("Device", device_options, index=default_index)
    if device_choice == "cuda" and not torch.cuda.is_available():
        st.sidebar.warning("CUDA is not available on this machine. Falling back to CPU.")
        device_choice = "cpu"
    device = torch.device(device_choice)

    mc_passes = st.sidebar.slider("MC Dropout passes", 5, 20, 10)
    heatmap_blend = st.sidebar.slider("Heatmap blend", 0.0, 1.0, 0.55)
    conf_threshold = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.70)
    show_gradcam = st.sidebar.checkbox("Show Grad-CAM overlay", value=True)
    show_mask = st.sidebar.checkbox("Show segmentation mask", value=True)
    show_uncertainty = st.sidebar.checkbox("Show MC Dropout bars", value=False)

    st.sidebar.markdown("### Anatomy filter")
    anatomy_filter = st.sidebar.radio("Region", ["All", "Cecum", "Rectum"])

    st.sidebar.markdown("---")
    st.sidebar.write("Model status")
    if ACDNET_CKPT.exists() and ANATOMY_CKPT.exists():
        st.sidebar.success("Checkpoints found")
    else:
        st.sidebar.warning("Run the notebook training cells first")

    if st.sidebar.button("Load model") or ("model" not in st.session_state):
        with st.spinner("Loading model..."):
            st.session_state["model"] = load_model(device_choice)
        st.sidebar.success("Model loaded")

    if st.sidebar.button("Clear model"):
        st.session_state.pop("model", None)
        st.session_state["results"] = None
        st.sidebar.success("Model cache cleared")

    model = st.session_state.get("model")

    col_left, col_right = st.columns([1.2, 1.35], gap="large")

    with col_left:
        st.subheader("Input")
        demo_tab, upload_tab = st.tabs(["Demo", "Upload"])
        uploaded_file = None
        selected_image_np = None
        is_video = False
        frames = []

        with demo_tab:
            if svg_demo_available:
                st.image(str(DEMO_SVG), caption="Bundled demo illustration", use_container_width=True)
            if demo_image is not None and st.button("Use dataset demo image"):
                selected_image_np = demo_image
                st.session_state["input_image"] = demo_image
                st.session_state["is_video"] = False
                st.session_state["frame_timeline"] = []
                st.session_state["frame_predictions"] = []
                st.image(demo_image, caption="Demo image", use_container_width=True)

        with upload_tab:
            uploaded_file = st.file_uploader(
                "Upload an image (.jpg/.png) or video (.mp4)",
                type=["jpg", "jpeg", "png", "mp4"],
            )

        if uploaded_file is not None:
            raw_bytes = uploaded_file.getvalue()
            if uploaded_file.type == "video/mp4":
                is_video = True
                frames = video_to_frames_bytes(raw_bytes, max_frames=20)
                if not frames:
                    st.error("Could not extract frames from the uploaded video.")
                else:
                    selected_image_np = frames[len(frames) // 2]
                    st.session_state["input_image"] = selected_image_np
                    st.session_state["is_video"] = True
                    st.image(selected_image_np, caption="Representative frame", use_container_width=True)
            else:
                image = Image.open(BytesIO(raw_bytes)).convert("RGB")
                selected_image_np = np.array(image)
                st.session_state["input_image"] = selected_image_np
                st.session_state["is_video"] = False
                st.session_state["frame_timeline"] = []
                st.session_state["frame_predictions"] = []
                st.image(selected_image_np, caption="Uploaded image", use_container_width=True)

        can_run = model is not None and selected_image_np is not None
        run_clicked = st.button("Run inference", type="primary", disabled=not can_run)

        if run_clicked and can_run:
            with st.status("Running ACDNet inference...", expanded=True) as status:
                st.write("Preprocessing input image...")
                st.write("Extracting features (EfficientNet-B0)...")
                st.write("Applying CBAM attention...")
                st.write("FiLM conditioning with anatomy embedding...")
                st.write(f"Running MC Dropout ({mc_passes} passes)...")

                main_result = run_inference(model, device, selected_image_np, mc_passes)
                main_result["input_image"] = selected_image_np
                main_result["is_video"] = is_video

                if is_video and frames:
                    sampled_frames = frames[:6]
                    frame_predictions = []
                    frame_mc_passes = max(3, min(mc_passes, 8))
                    for frame_np in sampled_frames:
                        frame_result = run_inference(model, device, frame_np, frame_mc_passes)
                        frame_predictions.append(
                            {
                                "high_risk": frame_result["high_risk"],
                                "detected": frame_result["detected"],
                                "det_conf": frame_result["det_conf"],
                            }
                        )
                    main_result["frame_timeline"] = sampled_frames
                    main_result["frame_predictions"] = frame_predictions
                else:
                    main_result["frame_timeline"] = []
                    main_result["frame_predictions"] = []

                st.session_state["results"] = main_result
                st.session_state["frame_timeline"] = main_result["frame_timeline"]
                st.session_state["frame_predictions"] = main_result["frame_predictions"]

                status.update(label="Inference complete", state="complete")

    with col_right:
        render_results_panel(
            st.session_state.get("results"),
            mc_passes=mc_passes,
            show_gradcam=show_gradcam,
            show_mask=show_mask,
            show_uncertainty=show_uncertainty,
            heatmap_blend=heatmap_blend,
            anatomy_filter=anatomy_filter,
        )

        results = st.session_state.get("results")
        if results:
            if results["det_conf"] < conf_threshold:
                st.warning("Confidence is below the selected threshold. Review recommended.")
            if results["review_flag"]:
                st.info("Uncertainty is elevated. Consider manual review.")

    st.sidebar.markdown("---")
    st.sidebar.caption("Run the notebook training cells first, then return here to inspect predictions.")


if __name__ == "__main__":
    main()
