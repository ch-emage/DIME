# =========================================================
# config.py — Global parameters. Edit here instead of the UI.
# =========================================================

ROI_MARGIN      = 20
SHOW_MARGIN     = False

# Minimum bounding-box area (pixels) for a detection to count as an anomaly.
# Smaller boxes are filtered out as noise.
MIN_ANOMALY_AREA = 20

VIDEO_CONFIRM_FRAMES = 5
VIDEO_HOLD_FRAMES    = 5

LIVE_CONFIRM_FRAMES  = 3
LIVE_HOLD_FRAMES     = 3

# ── training ────────────────────────────────────────────
TRAINING_SCRIPT      = "/home/emage/DIME/DIME/dime_training/train_tuple.py"            # path to train.py (e.g. "/home/user/DIME/train.py")
FEATURE_WINDOW       = 1             # --feature-window passed to train_tuple.py
WINDOW_STEP          = 1             # --window-step passed to train_tuple.py

# ── anomaly clip recording ──────────────────────────────
CLIP_ENABLED         = True          # set False to disable clip saving
CLIP_SECONDS_BEFORE  = 3             # seconds of footage to keep before anomaly
CLIP_SECONDS_AFTER   = 3             # seconds of footage to keep after anomaly clears
CLIP_OUTPUT_DIR      = "anomaly_clips"

# ── full-session video recording ────────────────────────
FULL_VIDEO_ENABLED        = False         # set True to save the entire inference session
FULL_VIDEO_SAVE_RAW       = True          # save the unannotated feed
FULL_VIDEO_SAVE_PROCESSED = True          # save the annotated feed (with boxes/labels)
FULL_VIDEO_OUTPUT_DIR     = "full_videos"
