[![image](https://img.shields.io/discord/772774097734074388?label=Discord&logo=discord)](https://discord.com/servers/software-engineering-courses-secourses-772774097734074388) [![Hits](https://hits.sh/github.com/FurkanGozukara/SECourses_Musubi_Trainer.svg?style=plastic&label=Hits%20Since%2025.08.27&labelColor=007ec6&logo=SECourses)](https://hits.sh/github.com/FurkanGozukara/Stable-Diffusion/)
[![Patreon](https://img.shields.io/badge/Patreon-Support%20Me-F2EB0E?style=for-the-badge&logo=patreon)](https://www.patreon.com/c/SECourses) [![BuyMeACoffee](https://img.shields.io/badge/Buy%20Me%20a%20Coffee-ffdd00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black)](https://www.buymeacoffee.com/DrFurkan) [![Furkan Gözükara Medium](https://img.shields.io/badge/Medium-Follow%20Me-800080?style=for-the-badge&logo=medium&logoColor=white)](https://medium.com/@furkangozukara) [![Codio](https://img.shields.io/static/v1?style=for-the-badge&message=Articles&color=4574E0&logo=Codio&logoColor=FFFFFF&label=CivitAI)](https://civitai.com/user/SECourses/articles) [![Furkan Gözükara Medium](https://img.shields.io/badge/DeviantArt-Follow%20Me-990000?style=for-the-badge&logo=deviantart&logoColor=white)](https://www.deviantart.com/monstermmorpg)

[![YouTube Channel](https://img.shields.io/badge/YouTube-SECourses-C50C0C?style=for-the-badge&logo=youtube)](https://www.youtube.com/SECourses)  [![Furkan Gözükara LinkedIn](https://img.shields.io/badge/LinkedIn-Follow%20Me-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/furkangozukara/)   [![Udemy](https://img.shields.io/static/v1?style=for-the-badge&message=Stable%20Diffusion%20Course&color=A435F0&logo=Udemy&logoColor=FFFFFF&label=Udemy)](https://www.udemy.com/course/stable-diffusion-dreambooth-lora-zero-to-hero/?referralCode=E327407C9BDF0CEA8156) [![Twitter Follow Furkan Gözükara](https://img.shields.io/badge/Twitter-Follow%20Me-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white)](https://twitter.com/GozukaraFurkan)

# SECourses Musubi Tuner - 1-Click to Install App for LoRA Training and Full Fine Tuning Qwen Image, Qwen Image Edit, Wan 2.1 and Wan 2.2 Models with Musubi Tuner with Ready Presets

## APP Download Link : https://www.patreon.com/posts/137551634

- This is an interface app based on the famous Kohya Musubi Tuner, enhanced with extra features, a high level of detail, and an easy-to-use interface.

### Quick Links
*   **Patreon:** [Exclusive Posts Index](https://github.com/FurkanGozukara/Stable-Diffusion/blob/main/Patreon-Posts-Index.md) | [Scripts Update History](https://github.com/FurkanGozukara/Stable-Diffusion/blob/main/Patreon-Posts-Index.md) | [Special Generative Scripts List](https://github.com/FurkanGozukara/Stable-Diffusion/blob/main/Amazing-Generative-AI-Scripts.md)
*   **GitHub:** [Stable Diffusion & Generative AI Repository](https://github.com/FurkanGozukara/Stable-Diffusion) (Please Star, Watch, and Fork!)
*   **Community:** [SECourses Discord](https://discord.com/servers/software-engineering-courses-secourses-772774097734074388) | [Reddit Subreddit](https://www.reddit.com/r/SECourses/)
*   **Connect:** [LinkedIn](https://www.linkedin.com/in/furkangozukara/)

---

## Latest Release

**Latest Zip File:** [**SECourses_Musubi_Trainer_v3.zip**](https://www.patreon.com/posts/137551634)

### Release Notes (v1 - Initial Release)

*   **App Screenshots Gallery:** [View on Reddit](https://www.reddit.com/r/SECourses/comments/1n2qo9g/secourses_musubi_tuner_v1_published_a_training/)
*   **Current Focus:** Full research is underway to prepare the very best presets for **Qwen Image LoRA training**.
    *   The goal is to enable Qwen Image LoRA training on GPUs with as low as 6 or 8 GB of VRAM using block swapping and other optimizations.
*   **Easy Installation:** 1-click installers are available for Windows, RunPod, and Massed Compute.
    *   Includes a 1-click model downloader script for necessary models (`qwen_2.5_vl_7b_fp16.safetensors`, `qwen_image_bf16.safetensors`, `qwen_train_vae.safetensors`).
    *   **Important:** Please use the provided model downloader to avoid issues with incorrect model versions.
    *   Moreover the downloader script verifies SHA 256 hash of the models and prevents any possibly corrupted model downloads
    *   The model downloader uses a UGET-like method for ultra-fast and robust downloads, replacing the standard Hugging Face downloader.
    *   The Musubi Tuner automatically handles FP8 and FP8 scaled conversion when loading the BF16 model into RAM so BF16 models are used.
*   **Technical Foundation:** This app is an interface based on the official [Kohya Musubi Tuner](https://github.com/kohya-ss/musubi-tuner), incorporating all its features plus additional enhancements.
*   **Modern Tech Stack:** The installer comes with Torch 2.7, CUDA 12.8, and pre-compiled libraries for xFormers, Triton, Flash Attention, and Sage Attention.
    *   **Broad GPU Support:** Supports a wide range of GPUs including RTX 3000, 4000, 5000 series, A40, L40, A100, H100, B200, etc.
    *   *Note:* Flash Attention and Sage Attention may not work on RTX 2000 or 1000 series GPUs, but a solution is planned.
*   **Current Capabilities:**
    *   Fully supports **Qwen Image model LoRA training**.
    *   Supports **Qwen2.5-VL image captioning**
*   **Important Note:** The original `Musubi Tuner` tab from the fork is not tested or supported. Please use the **Qwen Image LoRA** and **Image Captioning** tabs.
*   The included `test1.toml` is a basic test file to confirm functionality and is not an optimal configuration.

### What's Coming Next
*   Qwen Image Edit model LoRA training.
*   Qwen Image model full Fine Tuning / DreamBooth.
*   Wan 2.2 LoRA training and Fine Tuning / DreamBooth.
*   Wan 2.1 LoRA training and Fine Tuning / DreamBooth.

---

## System Requirements

### Windows
*   **Python:** 3.10.11
*   **NVIDIA:** CUDA 12.8, cuDNN 9.7 or above
*   **Tools:** FFmpeg, C++ tools, MSVC, and Git
*   *Note:* CUDA 12.8 is compatible with all modern GPUs. If you encounter issues, follow this tutorial precisely: [https://youtu.be/DrhUHnYfwC0](https://youtu.be/DrhUHnYfwC0).

---

## How To Install and Use

### Windows
*   A full tutorial is coming soon.
*   Use the `Windows_Install_and_Update.bat` script for installation and updates.
*   Follow the same folder logic as Kohya's trainer (e.g., `Parent Folder > 1_ohwx man`). Use the **Generate Dataset Configuration** button to handle the setup.

### Massed Compute (Recommended Cloud)
1.  **Sign Up:** [Register via this link](https://vm.massedcompute.com/signup?linkId=lp_034338&sourceId=secourses&tenantId=massed-compute).
    *   Use coupon code **`SECourses`** for a discount on all GPUs.
    *   For more details on GPUs and pricing, read [this post](https://www.patreon.com/posts/126671823).
2.  **Select GPU:** Choose an RTX A6000 or better (e.g., L40S, A6000 ADA, A100, H100, RTX 6000 PRO).
3.  **Select Image:** Choose **SECourses** from the "Creator" dropdown menu.
4.  **Follow Instructions:** Refer to the `Massed_Compute_Instructions_READ.txt` file in the repository.
5.  **Video Tutorial:** [How to use Massed Compute (starts at 12:58)](https://youtu.be/KW-MHmoNcqo?si=G1WbG-Qw4ujWvOtG&t=778)

### RunPod (Cloud)
1.  **Sign Up:** [Register via this link](https://get.runpod.io/955rkuppqv4h).
2.  **Follow Instructions:** Refer to the `Runpod_Instructions_READ.txt` file and use the template provided within it.
3.  **Video Tutorial:** [How to use RunPod (starts at 22:03)](https://youtu.be/KW-MHmoNcqo?si=QN8X8Sjn13ZYu-EU&t=1323)

---

## App Screenshots

<img width="2000" height="1159" alt="14" src="https://github.com/user-attachments/assets/96fa5212-5861-4afd-8ae7-29a6ae1fa293" />
<img width="1600" height="1450" alt="13" src="https://github.com/user-attachments/assets/fd6561d4-4f4c-4e43-8404-759c9ddff1cf" />
<img width="2799" height="1817" alt="12" src="https://github.com/user-attachments/assets/5a25a9f3-6690-46e8-ad68-2676799e2d3f" />
<img width="3840" height="4597" alt="11" src="https://github.com/user-attachments/assets/39d30306-4bcf-4066-86cb-1264a366ca67" />
<img width="3535" height="2519" alt="10" src="https://github.com/user-attachments/assets/82a437c5-99b0-40b2-b5ad-c090559248d2" />
<img width="3461" height="3125" alt="09" src="https://github.com/user-attachments/assets/4612f5d0-d8c7-44d3-83f2-28b98f99548e" />
<img width="3444" height="1564" alt="08" src="https://github.com/user-attachments/assets/a6d78fc1-11df-438d-9576-5f5eaac9223b" />
<img width="3454" height="1826" alt="07" src="https://github.com/user-attachments/assets/38c70bde-1f32-42d3-b270-55d73dad013a" />
<img width="3424" height="3593" alt="06" src="https://github.com/user-attachments/assets/414b57e2-1a14-4081-9f18-874f7cde9bc2" />
<img width="3489" height="3127" alt="05" src="https://github.com/user-attachments/assets/82b33f1d-c41b-4307-a83e-2b67c3a6463f" />
<img width="3499" height="2620" alt="04" src="https://github.com/user-attachments/assets/b0f589ee-3100-43a5-84fb-77f35b2f8e77" />
<img width="3585" height="2536" alt="03" src="https://github.com/user-attachments/assets/8f5ffbb9-a093-4913-9a4e-35e217209a1e" />
<img width="3696" height="2053" alt="02" src="https://github.com/user-attachments/assets/91b7c0d8-4ae8-43f0-8aa3-432c9eb0245e" />
<img width="3840" height="2343" alt="01" src="https://github.com/user-attachments/assets/67e68ed4-2523-42f7-be03-dd3d1f7265b2" />

Updates to V7
<img width="1175" height="1724" alt="updates" src="https://github.com/user-attachments/assets/f686ded6-1843-4833-87f2-094ae63b279f" />

