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

## Updates

![005](https://github.com/user-attachments/assets/fc0f58c5-bdda-499a-9983-1db1f80236a1)
<img width="1502" height="1000" alt="0010" src="https://github.com/user-attachments/assets/696f344a-d904-4005-bdef-d9ab2520272a" />


<img width="1572" height="1695" alt="44" src="https://github.com/user-attachments/assets/ad6b8c54-cbf2-4cc1-b8cb-91f4ae70d5ab" />
<img width="2803" height="1526" alt="43" src="https://github.com/user-attachments/assets/b801e2ee-c4a1-4cbc-b83d-4cde74b40fab" />
<img width="1825" height="1526" alt="42" src="https://github.com/user-attachments/assets/2bf3cfbc-b943-4268-8f2e-7fce65d23eed" />
<img width="2763" height="1478" alt="41" src="https://github.com/user-attachments/assets/b6b5f050-65d6-4dc6-acb5-bb68fe2bec49" />
<img width="1829" height="1487" alt="40" src="https://github.com/user-attachments/assets/acae6299-1d28-44b2-920a-1d04a3b85a3c" />


## Stage 1 Training Published Config Results Examples

30+ examples shared here : https://medium.com/@furkangozukara/qwen-image-lora-trainings-stage-1-results-and-pre-made-configs-published-as-low-as-training-with-ba0d41d76a05

<img width="1328" height="1328" alt="04" src="https://github.com/user-attachments/assets/3c7c5fd0-fc0d-4978-8800-ed485054db22" />
<img width="2656" height="2656" alt="03" src="https://github.com/user-attachments/assets/1e73012e-06f3-4181-ac49-ef10294d4119" />
<img width="2656" height="2656" alt="02" src="https://github.com/user-attachments/assets/ce432e3f-5f74-4df2-bb43-0f911ab55184" />
<img width="2656" height="2656" alt="01" src="https://github.com/user-attachments/assets/fe40513f-cfb9-4c91-9eb3-cfe45689e7a6" />
<img width="2656" height="2656" alt="06" src="https://github.com/user-attachments/assets/f254bd98-1171-46e6-b1cb-8c5e1837dd5c" />
<img width="2656" height="2656" alt="05" src="https://github.com/user-attachments/assets/cfa561ef-6323-4040-a8e2-05b43109ee1c" />



## App Screenshots

<img width="2000" height="1389" alt="14" src="https://github.com/user-attachments/assets/11158887-696c-4eec-b5ee-6077f7a47873" />
<img width="2291" height="1246" alt="13" src="https://github.com/user-attachments/assets/6f8e02ef-e37c-4da8-8eeb-41f873c236e9" />
<img width="3840" height="4621" alt="12" src="https://github.com/user-attachments/assets/4b9d4871-4f24-4d09-b257-ea05b5e0ae58" />
<img width="3477" height="2445" alt="11" src="https://github.com/user-attachments/assets/cbd2afaf-d4df-409c-a341-334122151e0b" />
<img width="3468" height="2286" alt="10" src="https://github.com/user-attachments/assets/6cc09542-5c01-4685-a8d7-e8f90039fc8a" />
<img width="3472" height="2800" alt="09" src="https://github.com/user-attachments/assets/e0e49752-4e32-46f1-b6ad-ea0f132fe461" />
<img width="3463" height="1550" alt="08" src="https://github.com/user-attachments/assets/5cf90326-2351-4fb9-b486-a9244eddf81f" />
<img width="3438" height="1855" alt="07" src="https://github.com/user-attachments/assets/cc010ec9-bfd6-4c35-b1da-b8fec426ffbf" />
<img width="3436" height="3643" alt="06" src="https://github.com/user-attachments/assets/8ec0f71f-d4c4-4f34-a3d1-bedb062c3c69" />
<img width="3486" height="3937" alt="05" src="https://github.com/user-attachments/assets/c6d12948-a2d5-413a-959e-be683b82f196" />
<img width="3473" height="2758" alt="04" src="https://github.com/user-attachments/assets/55c43c0e-77dd-4435-91cb-a17b4960e4ad" />
<img width="3471" height="1511" alt="03" src="https://github.com/user-attachments/assets/ed0b7055-dc99-419a-ae03-f377705a8a03" />
<img width="3461" height="1571" alt="02" src="https://github.com/user-attachments/assets/4c6685ba-146e-4b1a-af50-3b93da24f472" />
<img width="3840" height="3022" alt="01" src="https://github.com/user-attachments/assets/33ecd80c-6a53-41a1-bdc2-e35719bb9726" />





