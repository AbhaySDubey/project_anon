# project_anon

### Problem Statement - WHCS02 / Open Innovation

---

## **Introduction**
- ### ***project_anon*** is an effective way to anonymize people or any sensitive sectors in an image.

- ### The project has been created with the sole purpose of eliminating the risk of accidental or intentional invasion of other's privacy.

- ### To use it:

    - Upload an image/video
    - Use the filters provided:
        > - age (classified into age groups as: **tbd**)

        > - gender (classified into **male/female**)

    - or select a sensitive sector (e.g. faces or any objects) that you wish to blur out or privatise, i.e.

        > - choose from the **faces** shown as a list:

        > - the app will automatically detect and blur out the face everytime it locates it anywhere in the video stream.

---

## **Workfolow**

### *From what is decided, so far*:

- **Face Detection:** To Begin with we'll first have to detect the faces occuring in the given ***frame*** using any of the following:

    - Haar Cascade Classifier
    - #### (Using this)[Ultra Light Fast Generic Face Detector - 1MB](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB)
    - [YOLOv8 - Face Detection](https://huggingface.co/arnabdhar/YOLOv8-Face-Detection)
    - [EgoBlur](https://www.projectaria.com/tools/egoblur/)

- **We'll be caching these faces for further use (probably use Redis with time-stamps or frame numbers):**

- **Age/Gender Detection and Blurring:**

    - I still haven't decided on what model to use for this or to be fair even what approach to take.
    - Should I go for a pre-trained model that is capable of detecting age and gender without training or should I instead fine-tune a CNN model myself.
    - The latter approach is definitely going to be overkill, given the fact that I've limited time and there are a few models available that are pretty good at estimation of age and gender.
    - I'm sure about one thing however, that it'll be a better idea to cache faces w.r.t. time-stamps, frame numbers, etc. and use mathematical or ML-based ligthweight methods to compare and evaluate the faces using the cache instead of calling upon the heavy CNN model over and over again.
    - The previous point also allows me to provide users with the ability to select certain face(s) on the basis of which blur operation will be performed (viz. blurring the selected face(s) or blurring all faces except the selected face(s)).

- **Object Detection and NSFW blurring:**
    - I also wish to add this, but I'm 80% sure that implementing the above functionalities and preparing the PPT, Video would take up most of the time.