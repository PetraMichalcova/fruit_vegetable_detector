import cv2
import torch
import numpy as np

from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
)

from config import (
    CANDIDATE_COUNT,
    DEFAULT_FPS,
    INPUT_PATH,
    MIN_SHARPNESS,
    MODEL_NAME,
    MOTION_BLUR_KERNEL,
    MOTION_DIFF_THRESHOLD,
    MOTION_RESIZE,
    MOTION_THRESHOLD,
    OUTPUT_DIR,
    ROI,
    ROI_VIDEO_PATH,
    SELECTED_COUNT,
    STABLE_TIME,
    START_TIME,
    VIDEO_CODEC,
)

# =====================================================
# PRIPRAVENIE MODELU A JEHO POUŽITIE VO FUNKCII
# =====================================================

processor = AutoImageProcessor.from_pretrained(
    MODEL_NAME
)

model = AutoModelForImageClassification.from_pretrained(
    MODEL_NAME
)

model.eval()

def classify_frame(frame):
    # OpenCV -> RGB
    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    image = Image.fromarray(rgb)

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(
        outputs.logits,
        dim=-1
    )[0]

    return probabilities.cpu().numpy()

# =====================================================
# CESTY
# =====================================================

input_path = INPUT_PATH
output_dir = OUTPUT_DIR
roi_video_path = ROI_VIDEO_PATH

output_dir.mkdir(parents=True, exist_ok=True)
roi_video_path.parent.mkdir(parents=True, exist_ok=True)

# =====================================================
# ROI OVOCIA
# =====================================================

fruit_x1, fruit_y1, fruit_x2, fruit_y2 = ROI


# =====================================================
# MOTION DETECTION
# =====================================================

def has_motion(previous_roi, current_roi):

    previous_gray = cv2.cvtColor(
        previous_roi,
        cv2.COLOR_BGR2GRAY
    )

    current_gray = cv2.cvtColor(
        current_roi,
        cv2.COLOR_BGR2GRAY
    )

    # Menší obraz úplne stačí na motion detection
    previous_gray = cv2.resize(
        previous_gray,
        MOTION_RESIZE
    )

    current_gray = cv2.resize(
        current_gray,
        MOTION_RESIZE
    )

    # Jemné vyhladenie
    previous_gray = cv2.GaussianBlur(
        previous_gray,
        MOTION_BLUR_KERNEL,
        0
    )

    current_gray = cv2.GaussianBlur(
        current_gray,
        MOTION_BLUR_KERNEL,
        0
    )

    # Rozdiel medzi dvoma susednými framami
    diff = cv2.absdiff(
        previous_gray,
        current_gray
    )

    # Prah rozdielu pixelov
    _, motion_mask = cv2.threshold(
        diff,
        MOTION_DIFF_THRESHOLD,
        255,
        cv2.THRESH_BINARY
    )

    changed_pixels = cv2.countNonZero(
        motion_mask
    )

    total_pixels = motion_mask.size

    motion_ratio = (
        changed_pixels / total_pixels
    )

    motion_detected = (
        motion_ratio > MOTION_THRESHOLD
    )

    return motion_detected, motion_ratio


# =====================================================
# BLUR / OSTROST
# =====================================================

def is_blurry(roi):

    gray = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2GRAY
    )

    sharpness = cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()

    blurry = sharpness < MIN_SHARPNESS

    return blurry, sharpness


# =====================================================
# VIDEO
# =====================================================

cap = cv2.VideoCapture(
    str(input_path)
)

if not cap.isOpened():
    raise SystemExit(
        f"Video sa nepodarilo otvorit: {input_path}"
    )


# =====================================================
# FPS
# =====================================================

fps = cap.get(
    cv2.CAP_PROP_FPS
)

if not fps or fps <= 0:
    fps = DEFAULT_FPS

stable_frames_required = max(
    1,
    int(STABLE_TIME * fps)
)

print(f"FPS videa: {fps:.2f}")
print(f"Kontrola od: {START_TIME:.1f} s")
print(f"Stabilita: {STABLE_TIME:.2f} s")
print(f"Stable frames required: {stable_frames_required}")
print(f"Motion threshold: {MOTION_THRESHOLD:.4f}")
print(f"Candidate count: {CANDIDATE_COUNT}")


# =====================================================
# ROZMERY
# =====================================================

frame_width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

frame_height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)


# =====================================================
# KONTROLA ROI
# =====================================================

if (
    fruit_x1 < 0
    or fruit_y1 < 0
    or fruit_x2 > frame_width
    or fruit_y2 > frame_height
    or fruit_x1 >= fruit_x2
    or fruit_y1 >= fruit_y2
):
    raise SystemExit(
        "ROI ovocia je mimo rozmerov videa."
    )


# =====================================================
# ROI VIDEO
# =====================================================

roi_width = fruit_x2 - fruit_x1
roi_height = fruit_y2 - fruit_y1

roi_video_writer = cv2.VideoWriter(
    str(roi_video_path),
    cv2.VideoWriter_fourcc(*VIDEO_CODEC),
    fps,
    (roi_width, roi_height)
)

if not roi_video_writer.isOpened():

    cap.release()

    raise SystemExit(
        f"Video s ROI sa nepodarilo vytvorit: "
        f"{roi_video_path}"
    )


# =====================================================
# PREMENNE
# =====================================================

recording = False

frame_number = 0

# Predchádzajúci frame
previous_fruit_roi = None

# Koľko stabilných porovnaní máme za sebou
stable_frames = 0

# Kandidáti zo stabilnej sekvencie
candidates = []

finished = False


# =====================================================
# MAIN LOOP
# =====================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    current_time = (
        frame_number / fps
    )

    # ---------------------------------------------
    # ROI
    # ---------------------------------------------

    fruit_roi = frame[
        fruit_y1:fruit_y2,
        fruit_x1:fruit_x2
    ]

    roi_video_writer.write(
        fruit_roi
    )


    # ---------------------------------------------
    # START
    # ---------------------------------------------

    if (
        not recording
        and current_time >= START_TIME
    ):

        recording = True

        previous_fruit_roi = None
        stable_frames = 0
        candidates.clear()

        print()
        print("==============================")
        print("ZACINAM HLADAT STABILNU SCENU")
        print("==============================")
        print()


    # =================================================
    # QUALITY / STABILITY PIPELINE
    # =================================================

    if recording and not finished:

        # =============================================
        # PRVY FRAME
        # =============================================

        if previous_fruit_roi is None:

            previous_fruit_roi = (
                fruit_roi.copy()
            )

            print(
                f"Frame {frame_number} | "
                f"{current_time:.2f}s | "
                f"PRVY FRAME"
            )

            frame_number += 1
            continue


        # =============================================
        # 1. MOTION MEDZI SUSEDNYMI FRAMAMI
        # =============================================

        motion, motion_score = has_motion(
            previous_fruit_roi,
            fruit_roi
        )

        # Aktualny frame bude pri dalsom kole previous
        previous_fruit_roi = (
            fruit_roi.copy()
        )


        # =============================================
        # 2. AK JE POHYB -> RESET
        # =============================================

        if motion:

            stable_frames = 0

            # Ak sme už zbierali kandidátov,
            # stabilná sekvencia bola prerušená.
            if candidates:

                print(
                    f"Frame {frame_number} | "
                    f"{current_time:.2f}s | "
                    f"POHYB -> MAZEM "
                    f"{len(candidates)} kandidatov"
                )

                candidates.clear()

            print(
                f"Frame {frame_number} | "
                f"{current_time:.2f}s | "
                f"MOTION={motion_score:.4f} | "
                f"stable=0"
            )

            frame_number += 1
            continue


        # =============================================
        # 3. STABILNY FRAME
        # =============================================

        stable_frames += 1

        print(
            f"Frame {frame_number} | "
            f"{current_time:.2f}s | "
            f"motion={motion_score:.4f} | "
            f"stable={stable_frames}/"
            f"{stable_frames_required}"
        )


        # =============================================
        # 4. ESTE NEMAME DOSTATOCNU STABILITU
        # =============================================

        if stable_frames < stable_frames_required:

            frame_number += 1
            continue


        # =============================================
        # SCENA JE STABILNA
        # =============================================

        # Od tohto bodu už frame môže byť kandidát.


        # =============================================
        # 5. BLUR GATE
        # =============================================

        blurry, sharpness = is_blurry(
            fruit_roi
        )

        if blurry:

            print(
                f"Frame {frame_number} | "
                f"{current_time:.2f}s | "
                f"STABLE ale BLUR | "
                f"sharpness={sharpness:.1f}"
            )

            frame_number += 1
            continue


        # =============================================
        # 6. CANDIDATE
        # =============================================

        candidate = {
            "frame_number": frame_number,
            "time": current_time,
            "frame": fruit_roi.copy(),
            "motion": motion_score,
            "sharpness": sharpness,
        }

        candidates.append(
            candidate
        )

        print(
            f"Frame {frame_number} | "
            f"{current_time:.2f}s | "
            f"CANDIDATE "
            f"{len(candidates)}/{CANDIDATE_COUNT} | "
            f"motion={motion_score:.4f} | "
            f"sharpness={sharpness:.1f}"
        )


        # =============================================
        # 7. ESTE ZBIERAME
        # =============================================

        if len(candidates) < CANDIDATE_COUNT:

            frame_number += 1
            continue


        # =============================================
        # 8. MAME DOST KANDIDATOV
        # =============================================

        print()
        print("==============================")
        print("MAM DOST KANDIDATOV")
        print("==============================")


        # Zoradenie od najostrejšieho
        candidates.sort(
            key=lambda c: c["sharpness"],
            reverse=True
        )


        # Vyber top 3
        selected = candidates[:SELECTED_COUNT]

        all_probabilities = []

        for candidate in selected:
            probabilities = classify_frame(
                candidate["frame"]
            )

            all_probabilities.append(
                probabilities
            )

        avg_probabilities = np.mean(
            all_probabilities,
            axis=0
        )

        class_id = int(
            np.argmax(avg_probabilities)
        )

        confidence = float(
            avg_probabilities[class_id]
        )

        label = model.config.id2label[class_id]

        print()
        print("==============================")
        print("KLASIFIKACIA")
        print("==============================")
        print(f"Predikcia: {label}")
        print(f"Confidence: {confidence * 100:.2f} %")


        # =============================================
        # 9. ULOZENIE SELECTED
        # =============================================

        print()

        for index, candidate in enumerate(
            selected,
            start=1
        ):

            filename = output_dir / (
                f"selected_{index}_"
                f"frame_"
                f"{candidate['frame_number']:06d}.jpg"
            )

            success = cv2.imwrite(
                str(filename),
                candidate["frame"]
            )

            if success:

                print(
                    f"SELECTED #{index} | "
                    f"frame="
                    f"{candidate['frame_number']} | "
                    f"time="
                    f"{candidate['time']:.2f}s | "
                    f"motion="
                    f"{candidate['motion']:.4f} | "
                    f"sharpness="
                    f"{candidate['sharpness']:.1f} | "
                    f"{filename}"
                )


        finished = True


    frame_number += 1


# =====================================================
# KONIEC
# =====================================================

cap.release()

roi_video_writer.release()


print()
print("==============================")
print("HOTOVO")

print(
    f"Candidates: {len(candidates)}"
)

print(
    f"Selected frames: "
    f"{min(len(candidates), SELECTED_COUNT)}"
)

print(
    f"ROI video: {roi_video_path}"
)

print(
    f"Selected ulozene v: {output_dir}"
)

print("==============================")
