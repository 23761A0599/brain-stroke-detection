"""
explainers.py  -  genuine Grad-CAM, Grad-CAM++ and LIME for the EfficientNet-B0 classifier,
plus a faithfulness (deletion) test and simple renderers.

Why this replaces the old gradcam_images.py / lime_explainer.py logic
  * "Grad-CAM" used to show only the largest top-8% blob of the map (and zeroed a 6% border),
    so a "region" was always drawn even when the evidence was diffuse or lay on the border.
    Here the FULL continuous map is shown, and border/background attention is MEASURED, not hidden.
  * "LIME" used to be a re-coloured Grad-CAM contour. Here it is a real LIME:
    superpixels -> random on/off perturbations -> model predictions -> weighted ridge surrogate.
  * The "Grad-CAM++" / "pro" view used true Grad-CAM weights only; here Grad-CAM++ uses the
    real second/third-order weighting.
  * Labels such as "Suspected Hemorrhage" are no longer drawn unless the model predicted hemorrhage.

Torch is imported lazily so the numpy/OpenCV parts can be tested without it.
"""
import numpy as np
import cv2

HEM_IDX = 0                      # class index 0 = Hemorrhagic (ImageFolder alphabetical order)
IMG = 224


# ----------------------------------------------------------------------------------------
# 1. Grad-CAM / Grad-CAM++
# ----------------------------------------------------------------------------------------
def compute_cam(model, x, method="gradcam", target_class=None, layer=None, size=IMG):
    """
    x: (1,3,H,W) normalised tensor on the model's device.
    method: "gradcam" | "gradcam++"
    target_class: class to explain (default = predicted class)
    layer: module whose output is used (default model.features[-1], 7x7 for 224 input)
    Returns (cam [size,size] in [0,1], predicted_class_index, class_probabilities ndarray).
    Works even when the backbone parameters are frozen (gradient flows through the input).
    """
    import torch
    model.eval()
    layer = layer if layer is not None else model.features[-1]
    store = {}
    handle = layer.register_forward_hook(lambda m, i, o: store.__setitem__("a", o))
    try:
        x = x.clone().detach().requires_grad_(True)
        with torch.enable_grad():
            out = model(x)
            probs = torch.softmax(out, dim=1)[0].detach().cpu().numpy()
            pred = int(out.argmax(dim=1))
            cls = pred if target_class is None else int(target_class)
            act = store["a"]
            grads = torch.autograd.grad(out[0, cls], act)[0]
    finally:
        handle.remove()

    A = act.detach()[0]            # (C,h,w)
    G = grads.detach()[0]
    if method == "gradcam":
        w = G.mean(dim=(1, 2))
    elif method == "gradcam++":
        g2 = G ** 2
        g3 = g2 * G
        denom = 2 * g2 + A.sum(dim=(1, 2), keepdim=True) * g3
        denom = torch.where(denom != 0, denom, torch.ones_like(denom))
        alpha = g2 / denom
        w = (alpha * torch.relu(G)).sum(dim=(1, 2))
    else:
        raise ValueError(method)
    cam = torch.relu((w[:, None, None] * A).sum(dim=0)).cpu().numpy().astype(np.float32)
    cam = cv2.resize(cam, (size, size), interpolation=cv2.INTER_LINEAR)
    m = cam.max()
    if m > 0:
        cam = cam / m
    return cam, pred, probs


def make_predict_fn(model, device, batch=64):
    """uint8 RGB batch (N,H,W,3) -> softmax probabilities (N,2); same normalisation as training."""
    import torch
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    model.eval()

    def predict(imgs):
        outs = []
        with torch.no_grad():
            for i in range(0, len(imgs), batch):
                xb = torch.from_numpy(np.ascontiguousarray(imgs[i:i + batch])).to(device)
                xb = xb.permute(0, 3, 1, 2).float() / 255.0
                outs.append(torch.softmax(model((xb - mean) / std), dim=1).cpu().numpy())
        return np.concatenate(outs)
    return predict


# ----------------------------------------------------------------------------------------
# 2. LIME (image) - superpixels, random on/off perturbations, weighted ridge surrogate
# ----------------------------------------------------------------------------------------
def segment_image(img_rgb, n_segments=40, compactness=10.0):
    h, w = img_rgb.shape[:2]
    try:
        from skimage.segmentation import slic
        try:
            seg = slic(img_rgb, n_segments=n_segments, compactness=compactness,
                       start_label=0, channel_axis=-1)
        except TypeError:                                   # older scikit-image
            seg = slic(img_rgb, n_segments=n_segments, compactness=compactness, multichannel=True)
        _, seg = np.unique(seg, return_inverse=True)
        return seg.reshape(h, w)
    except ImportError:                                     # fallback: square grid
        g = max(2, int(round(np.sqrt(n_segments))))
        ys = np.minimum(np.arange(h) * g // h, g - 1)
        xs = np.minimum(np.arange(w) * g // w, g - 1)
        return ys[:, None] * g + xs[None, :]


def lime_image(predict_fn, img_rgb, target_class, n_samples=800, n_segments=40,
               kernel_width=0.25, seed=0, chunk=128):
    """
    Returns dict(segments, coef, intercept, r2, base_prob).
    coef[k] > 0  -> keeping superpixel k raises P(target_class) in the local linear surrogate.
    r2 is the weighted R^2 of the surrogate: report it, a low value means a weak explanation.
    """
    from sklearn.linear_model import Ridge
    rng = np.random.default_rng(seed)
    seg = segment_image(img_rgb, n_segments=n_segments)
    K = int(seg.max()) + 1
    Z = rng.integers(0, 2, size=(n_samples, K)).astype(np.float32)
    Z[0, :] = 1.0                                           # the unperturbed image
    fill = img_rgb.reshape(-1, 3).mean(axis=0).astype(np.uint8)

    probs = np.empty(n_samples, dtype=np.float32)
    for s in range(0, n_samples, chunk):
        zs = Z[s:s + chunk]
        batch = np.repeat(img_rgb[None], len(zs), axis=0)
        for j, z in enumerate(zs):
            batch[j][(z[seg] == 0)] = fill
        probs[s:s + len(zs)] = predict_fn(batch)[:, target_class]

    ones = np.ones(K, dtype=np.float32)
    cos = (Z @ ones) / (np.linalg.norm(Z, axis=1) * np.linalg.norm(ones) + 1e-12)
    dist = 1.0 - cos
    wts = np.sqrt(np.exp(-(dist ** 2) / kernel_width ** 2))
    reg = Ridge(alpha=1.0, fit_intercept=True).fit(Z, probs, sample_weight=wts)
    return {"segments": seg, "coef": reg.coef_, "intercept": float(reg.intercept_),
            "r2": float(reg.score(Z, probs, sample_weight=wts)), "base_prob": float(probs[0])}


# ----------------------------------------------------------------------------------------
# 3. Faithfulness: deletion test (lower area = the map really points at what the model uses)
# ----------------------------------------------------------------------------------------
def deletion_curve(predict_fn, img_rgb, importance, target_class, steps=11, max_frac=0.5):
    """Remove the most important pixels first (replace by the image mean) and track P(target)."""
    flat = img_rgb.reshape(-1, 3)
    order = np.argsort(-importance.ravel(), kind="stable")
    n = order.size
    fill = flat.mean(axis=0).astype(np.uint8)
    fracs = np.linspace(0.0, max_frac, steps)
    batch = []
    for f in fracs:
        im = flat.copy()
        im[order[:int(round(f * n))]] = fill
        batch.append(im.reshape(img_rgb.shape))
    p = predict_fn(np.stack(batch))[:, target_class]
    return fracs, p


def deletion_auc(fracs, p):
    p = np.asarray(p, dtype=float)
    return float(np.sum((p[1:] + p[:-1]) / 2.0 * np.diff(fracs)) / fracs[-1])   # trapezoid rule (numpy-version safe)


# ----------------------------------------------------------------------------------------
# 4. Sanity metric: how much of the map lies on the black background (outside the head)?
# ----------------------------------------------------------------------------------------
def background_attention_fraction(cam, img_rgb, bg_thresh=15):
    bg = img_rgb.mean(axis=2) < bg_thresh
    total = float(cam.sum())
    return float(cam[bg].sum() / total) if total > 0 else 0.0


# ----------------------------------------------------------------------------------------
# 5. Renderers (RGB uint8 in, RGB uint8 out)
# ----------------------------------------------------------------------------------------
def overlay_heatmap(base_rgb, cam, alpha=0.45):
    heat = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * np.clip(cam, 0, 1)), cv2.COLORMAP_JET),
                        cv2.COLOR_BGR2RGB)
    return np.uint8((1 - alpha) * base_rgb + alpha * heat)


def draw_regions(base_rgb, cam, thr=0.5, min_area=60, color=(255, 215, 0)):
    """Boxes around connected areas where cam >= thr * max. Nothing is drawn if no area qualifies."""
    binary = (cam >= thr).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out, boxes = base_rgb.copy(), []
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        boxes.append((x, y, w, h))
    return out, boxes


def overlay_lime(base_rgb, seg, coef, top_k=5):
    """Green tint + yellow outline on the top_k superpixels that raise P(target class)."""
    out = base_rgb.astype(np.float32)
    thr = 0.1 * float(coef.max()) if coef.max() > 0 else 0.0     # ignore near-zero (noise) segments
    top = [k for k in np.argsort(-coef)[:top_k] if coef[k] > thr]
    for k in top:
        m = seg == k
        out[m] = 0.55 * out[m] + 0.45 * np.array([0, 255, 0], dtype=np.float32)
    out = np.uint8(out)
    for k in top:
        cs, _ = cv2.findContours((seg == k).astype(np.uint8) * 255, cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cs, -1, (255, 255, 0), 1)
    return out


def lime_importance_map(seg, coef):
    return coef[seg]