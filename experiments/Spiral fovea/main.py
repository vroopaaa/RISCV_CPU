    import cv2
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn.functional as F

    # -------------------------------------------------------------------------
    # 1. Load image and compute local Shannon entropy map (Stage 1)
    # -------------------------------------------------------------------------
    img_path = "/home/roopi/Desktop/rsvp/experiments/Spiral fovea/01226.jpg"
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
    raise FileNotFoundError(f"Cannot read image at: {img_path}")

    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))

    # BT.601 luminance
    gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    gray_t = torch.tensor(gray, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    # Quantize into B=16 bins
    BINS = 16
    quant = (gray_t / 256.0 * BINS).long().clamp(0, BINS - 1)

    # Unfold sliding window (w=9, r=4) for local histogram probabilities
    W = 9
    patches_unfold = F.unfold(quant.float(), kernel_size=W, padding=W // 2)
    B_oh = F.one_hot(
        patches_unfold.long().squeeze(0).permute(1, 0), num_classes=BINS
    ).float()
    prob = B_oh.mean(dim=1) + 1e-7
    entropy = -(prob * torch.log2(prob)).sum(dim=-1).reshape(224, 224).numpy()

    # -------------------------------------------------------------------------
    # 2. Hotspot Anchors across S=4 strips with Deduplication (Stage 2)
    # -------------------------------------------------------------------------
    S = 4
    strip_h = 224 // S
    raw_anchors = []
    for s in range(S):
    sub_ent = entropy[s * strip_h : (s + 1) * strip_h, :]
    y_local, x = np.unravel_index(np.argmax(sub_ent), sub_ent.shape)
    y = s * strip_h + y_local
    raw_anchors.append((int(x), int(y)))

    # Deduplicate anchors (tau_dedup exclusion radius)
    TAU_DEDUP = 20
    anchors = []
    for ax, ay in raw_anchors:
    if not any(np.hypot(ax - rx, ay - ry) < TAU_DEDUP for rx, ry in anchors):
        anchors.append((ax, ay))

    # -------------------------------------------------------------------------
    # 3. Multi-Scale Spiral Ring Extraction (Stage 3)
    # Schedule: [(patch_size sigma, ring_radius rho)]
    # -------------------------------------------------------------------------
    schedule = [(24, 0), (28, 18), (36, 40), (48, 70)]
    alpha = 1.3
    selected_patches = []  # list of (px, py, sigma, ring_index)

    for ax, ay in anchors:
    for k, (sigma, rho) in enumerate(schedule):
        if k == 0:
        selected_patches.append((ax, ay, sigma, k))
        else:
        nk = max(1, int(np.floor(2 * np.pi * rho / (alpha * sigma))))
        for j in range(nk):
            theta = 2 * np.pi * j / nk
            px = int(round(ax + rho * np.cos(theta)))
            py = int(round(ay + rho * np.sin(theta)))
            # Out-of-bounds filter (paper τ_oob): drop if patch center leaves the canvas
            if 0 <= px < 224 and 0 <= py < 224:
            selected_patches.append((px, py, sigma, k))

    # -------------------------------------------------------------------------
    # 4. Plot Overlay (Color-coded by Ring Scale)
    # -------------------------------------------------------------------------
    ring_colors = {
        0: "#00FF66",
        1: "#FFD700",
        2: "#FF8C00",
        3: "#FF3366",
    }  # Center, Inner, Mid, Outer
    ring_labels = {
        0: "k=0 (Foveal)",
        1: "k=1 (Ring 1)",
        2: "k=2 (Ring 2)",
        3: "k=3 (Ring 3)",
    }

    fig, ax = plt.subplots(1, 2, figsize=(13, 6))

    # Panel 1: Shannon Entropy Map
    ax[0].imshow(entropy, cmap="magma")
    ax[0].set_title(
        "Stage 1 & 2: Local Shannon Entropy + Anchors", fontweight="bold"
    )
    for x, y in anchors:
    ax[0].plot(x, y, "cx", markersize=14, markeredgewidth=3)
    ax[0].axis("off")

    # Panel 2: Spiral Patches Overlay
    ax[1].imshow(img)
    ax[1].set_title(
        f"Stage 3: SpiralFovea Token Set (N = {len(selected_patches)} / 196)",
        fontweight="bold",
    )

    plotted_labels = set()
    for px, py, sigma, k in selected_patches:
    lbl = ring_labels[k] if k not in plotted_labels else None
    rect = patches.Rectangle(
        (px - sigma // 2, py - sigma // 2),
        sigma,
        sigma,
        linewidth=1.2,
        edgecolor=ring_colors[k],
        facecolor="none",
        label=lbl,
    )
    ax[1].add_patch(rect)
    plotted_labels.add(k)

    for x, y in anchors:
    ax[1].plot(x, y, "r+", markersize=10, markeredgewidth=2)

    ax[1].legend(loc="upper right", framealpha=0.85)
    ax[1].axis("off")

    plt.tight_layout()
    plt.show()