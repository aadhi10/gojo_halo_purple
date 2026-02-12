# 🤞 KYOSHIKI: MURASAKI : HALO PURPLE (Gojo Satoru FX)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?style=for-the-badge&logo=opencv)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Hand%20Tracking-orange?style=for-the-badge&logo=google)
![Status](https://img.shields.io/badge/Status-Special%20Grade-red?style=for-the-badge)

> *"Throughout Heaven and Earth, I alone am the honored one."*

A high-performance Computer Vision project that uses **AI Hand Tracking**, **Face Mesh**, and **Background Segmentation** to recreate Gojo Satoru's Cursed Techniques in real-time.



## ⚡ Features

* **🟦 Cursed Technique Lapse: Blue**
    * Tracks your **Right Index Finger**.
    * Generates a swirling, attractive blue energy orb with particle effects.
* **🟥 Cursed Technique Reversal: Red**
    * Tracks your **Left Index Finger**.
    * Generates a chaotic, repulsive red energy orb.
* **🟣 Hollow Technique: Purple**
    * **Merge Logic:** triggered when hands are brought together.
    * **Physics:** Lightning arcs and particle swarms intensify as hands get closer.
    * **Launch Sequence:** Hold the pose for 4 seconds to fire the "Hollow Purple" at the screen.
* **👀 Six Eyes Mode**
    * **Face Mesh Integration:** Detects your irises in real-time.
    * **Glow Effect:** Adds an intense blue aura to your eyes when the technique is active.

---

## 🛠️ Tech Stack

* **Language:** Python 3.x
* **Computer Vision:** OpenCV (`cv2`)
* **AI Models:** MediaPipe (Hands, Selfie Segmentation, Face Mesh)
* **Audio:** Pygame (for SFX)
* **Math:** NumPy (for vector calculations and particle physics)

---

## 🚀 Installation

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/yourusername/gojo-fx.git](https://github.com/yourusername/gojo-fx.git)
    cd gojo-fx
    ```

2.  **Install Dependencies**
    You need a few Python libraries to run the Six Eyes.
    ```bash
    pip install opencv-python mediapipe numpy pygame
    ```

3.  **Add Audio**
    * Download a "Hollow Purple" sound effect (mp3).
    * Rename it to `purple.mp3`.
    * Place it in the same folder as the script.

---

## 🎮 How to Use

1.  **Run the Script**
    ```bash
    python gojo.py
    ```
2.  **Controls**
    * **Right Hand:** Controls **Blue**.
    * **Left Hand:** Controls **Red**.
    * **Combine:** Bring index fingers close together to form **Purple**.
    * **Launch:** Hold the combined pose for **4 seconds** to trigger the blast.
    * **Quit:** Press `q` to exit the Domain.

---

## 📂 Project Structure

```text
gojo-fx/
├── gojo.py           # Main source code
├── purple.mp3        # Audio file (you must add this)
└── README.md         # Documentation
