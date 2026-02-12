import cv2
import mediapipe as mp
import numpy as np
import pygame
import random
import time
import os
import math

# --- AUDIO SETUP ---
pygame.mixer.pre_init(44100, -16, 2, 512)
pygame.mixer.init()
script_dir = os.path.dirname(os.path.abspath(__file__))
sound_path = os.path.join(script_dir, "purple.mp3")

try:
    if os.path.exists(sound_path):
        purple_sound = pygame.mixer.Sound(sound_path)
    else:
        purple_sound = None
except:
    purple_sound = None

# --- AI MODELS SETUP ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)

mp_selfie = mp.solutions.selfie_segmentation
segmentation = mp_selfie.SelfieSegmentation(model_selection=1)

# NEW: Face Mesh for Eye Tracking
mp_face_mesh = mp.solutions.face_mesh
# refine_landmarks=True is crucial for accurate iris detection
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.8)

cap = cv2.VideoCapture(0)
# High Def Resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# --- GLOBAL VARIABLES ---
frame_counter = 0
purple_start_time = None
launch_mode = False
launch_scale = 1.0
particles = [] 

# --- HELPER FUNCTIONS ---
def create_particle(center, color_type):
    x, y = center
    vx = random.uniform(-3, 3)
    vy = random.uniform(-10, -2)
    life = random.randint(10, 25)
    return [x, y, vx, vy, life, color_type]

def draw_eye_glow(img, lx, ly, rx, ry):
    """Draws the intense Six Eyes glow"""
    overlay = img.copy()
    # Inner bright core (Cyan/White)
    cv2.circle(overlay, (lx, ly), 6, (255, 255, 200), -1)
    cv2.circle(overlay, (rx, ry), 6, (255, 255, 200), -1)
    
    # Outer strong glow (Bright Blue)
    cv2.circle(overlay, (lx, ly), 20, (255, 150, 0), -1)
    cv2.circle(overlay, (rx, ry), 20, (255, 150, 0), -1)
    
    # Apply heavy blur for light bleed effect
    blurred = cv2.GaussianBlur(overlay, (31, 31), 0)
    
    # Blend heavily so it looks like light source
    img = cv2.addWeighted(blurred, 1.0, img, 1.0, 0)
    
    # Pinpoint white center for sharp look
    cv2.circle(img, (lx, ly), 2, (255, 255, 255), -1)
    cv2.circle(img, (rx, ry), 2, (255, 255, 255), -1)
    return img

def draw_complex_orb(img, center, tech_type, frame_count):
    cx, cy = center
    overlay = img.copy()
    
    if tech_type == "Blue":
        core_color = (255, 255, 200); glow_color = (255, 100, 0); ring_color = (255, 200, 50)
    else: # Red
        core_color = (200, 200, 255); glow_color = (0, 0, 180); ring_color = (50, 50, 255)

    base_radius = 60 + int(10 * math.sin(frame_count * 0.15))
    
    # Particles Loop
    for p in particles:
        px, py, vx, vy, life, p_type = p
        if p_type == tech_type:
            p[0] += vx; p[1] += vy; p[4] -= 1
            if life > 0:
                size = random.randint(2, 6)
                cv2.circle(overlay, (int(px), int(py)), size, ring_color, -1)

    for _ in range(4): particles.append(create_particle(center, tech_type))
    particles[:] = [p for p in particles if p[4] > 0]

    # Gyro Rings
    axes1 = (base_radius + 20, base_radius - 10)
    angle1 = frame_count * 15
    cv2.ellipse(overlay, (cx, cy), axes1, angle1, 0, 360, ring_color, 3)
    axes2 = (base_radius - 10, base_radius + 20)
    angle2 = frame_count * 10 + 45
    cv2.ellipse(overlay, (cx, cy), axes2, angle2, 0, 360, ring_color, 3)

    # Core Glow
    cv2.circle(overlay, (cx, cy), base_radius + 30, glow_color, -1)
    cv2.circle(overlay, (cx, cy), base_radius - 15, core_color, -1)
    
    blurred = cv2.GaussianBlur(overlay, (41, 41), 0)
    img = cv2.addWeighted(blurred, 0.7, img, 1.0, 0)
    
    # Sharp Rings on top
    cv2.ellipse(img, (cx, cy), axes1, angle1, 0, 360, (255,255,255), 1)
    cv2.ellipse(img, (cx, cy), axes2, angle2, 0, 360, (255,255,255), 1)
    return img

# --- MAIN LOOP ---
while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    frame_counter += 1
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 1. PROCESS FACE MESH (For Eyes)
    face_results = face_mesh.process(img_rgb)
    left_eye_pos = None
    right_eye_pos = None

    if face_results.multi_face_landmarks:
        for face_lms in face_results.multi_face_landmarks:
            # Landmark 468 is Left Iris Center, 473 is Right Iris Center
            li = face_lms.landmark[468]
            ri = face_lms.landmark[473]
            left_eye_pos = (int(li.x * w), int(li.y * h))
            right_eye_pos = (int(ri.x * w), int(ri.y * h))

    # 2. BACKGROUND REMOVAL (VOID)
    seg_results = segmentation.process(img_rgb)
    mask = seg_results.segmentation_mask
    mask = cv2.GaussianBlur(mask, (7, 7), 0)
    condition = np.stack((mask,) * 3, axis=-1) > 0.5
    bg_image = np.zeros(frame.shape, dtype=np.uint8)
    img = np.where(condition, frame, bg_image)
    
    # 3. HAND TRACKING & TECH DRAWING
    results = hands.process(img_rgb)
    hand_data = []
    if results.multi_hand_landmarks:
        for i, hand_lms in enumerate(results.multi_hand_landmarks):
            label = results.multi_handedness[i].classification[0].label
            pos = (int(hand_lms.landmark[8].x * w), int(hand_lms.landmark[8].y * h))
            hand_data.append((pos, label))

        # Draw Individual Techs
        for pos, label in hand_data:
            tech_type = "Blue" if label == "Right" else "Red"
            img = draw_complex_orb(img, pos, tech_type, frame_counter)

        # 4. HOLLOW PURPLE MERGE LOGIC
        if len(hand_data) == 2:
            p1, p2 = hand_data[0][0], hand_data[1][0]
            dist = int(np.linalg.norm(np.array(p1) - np.array(p2)))
            center = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)

            if dist < 140:
                # --- ACTIVATE SIX EYES GLOW ---
                # Only draw if face mesh found the eyes
                if left_eye_pos and right_eye_pos:
                    img = draw_eye_glow(img, left_eye_pos[0], left_eye_pos[1], right_eye_pos[0], right_eye_pos[1])

                # Audio & Timer Logic
                if purple_start_time is None:
                    purple_start_time = time.time()
                    if purple_sound: purple_sound.play()
                
                elapsed = time.time() - purple_start_time

                # Screen Shake
                if 3.0 < elapsed < 4.0:
                    shift_x = random.randint(-20, 20)
                    shift_y = random.randint(-20, 20)
                    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
                    img = cv2.warpAffine(img, M, (w, h))

                # Launch Logic
                if elapsed > 4.0:
                    launch_mode = True
                    launch_scale += 0.8
                
                # Purple Visuals
                overlay = img.copy()
                curr_radius = int(dist * 2.0 * launch_scale)
                cv2.circle(overlay, center, curr_radius, (200, 0, 150), -1)
                
                if not launch_mode: # Lightning
                    for _ in range(4):
                        off_x = random.randint(-100, 100)
                        off_y = random.randint(-100, 100)
                        cv2.line(img, center, (center[0]+off_x, center[1]+off_y), (255, 255, 255), 2)

                alpha = min(0.95, 0.5 + (launch_scale * 0.05))
                img = cv2.addWeighted(cv2.GaussianBlur(overlay, (61, 61), 0), alpha, img, 1 - alpha, 0)
                cv2.circle(img, center, int(dist * 0.6 * launch_scale), (255, 255, 255), -1)

                if launch_mode:
                    cv2.putText(img, "HALO: PURPLE", (w//4, h//2), 
                                cv2.FONT_HERSHEY_TRIPLEX, 2, (255, 255, 255), 3)
            else:
                # Reset if hands move apart
                purple_start_time = None
                launch_mode = False
                launch_scale = 1.0
                if purple_sound: purple_sound.stop()
    
    cv2.imshow("Gojo S-Rank + Six Eyes", img)
    if cv2.waitKey(1) & 0xFF == ord('q') or launch_scale > 25: break

cap.release()
cv2.destroyAllWindows()