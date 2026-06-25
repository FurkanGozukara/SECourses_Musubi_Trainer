# SECourses Musubi Tuner - 1-Click to Install App for LoRA Training and Full Fine Tuning Qwen Image, Qwen Image Edit, Wan 2.1 and Wan 2.2, FLUX Klein, FLUX 2, Z Image Base and Turbo Models with Musubi Tuner with Ready Presets

## App Installer Download Link : https://www.patreon.com/SECourses/posts/secourses-musubi-137551634
    
-   **Latest Zip File :** [**SECourses\_Musubi\_Trainer\_v28.zip**](https://www.patreon.com/file?h=137551634&m=648430324)
        
   -   **Main training tutorial to learn training and how to use this app (mandatory to watch) :** [**https://youtu.be/DPX3eBTuO\_Y**](https://youtu.be/DPX3eBTuO_Y)        
   -   Wan 2.2 training tutorial : [https://youtu.be/ocEkhAsPOs4](https://youtu.be/ocEkhAsPOs4)
   -   **SwarmUI :** [**https://www.patreon.com/posts/114517862**](https://www.patreon.com/posts/114517862)            
   -   ComfyUI : [https://www.patreon.com/posts/105023709](https://www.patreon.com/posts/105023709)              
  
    
-   Currently 1-click to install on Windows, RunPod and Massed Compute with uv installation - ultra fast
        
   -   We have model auto downloader that supports below models (run Windows\_Download\_Training\_Model\_Files.bat)            
   -   Qwen Image Training Models (Old and new 2512) with Torch Compile
       -   Qwen Image Edit Plus (2509 and 2511) Training Models
       -    Wan 2.1 Text to Video Training Models with Torch Compile
       -   Wan 2.2 Text to Video Training Models with Torch Compile
       -    Wan 2.2 Image to Video Training Models with Torch Compile
       -   Z-Image Turbo LoRA training with Torch Compile
       -   Z-Image Base Fine Tuning with Torch Compile
       -   Z-Image Base LoRA Training with Torch Compile
       -   FLUX 2 Dev LoRA Training with Torch Compile
       -   FLUX Klein 9B and 4B LoRA Training with Torch Compile
                
   -    Please use model downloader to not have any issues because your selected models may be wrong            
      -   Moreover, the Musubi Tuner automatically does FP8 and FP8 scaled conversion while loading BF16 model into RAM so we always use BF16 models for training                
      -   I specifically developed this model downaloder with like UGET method so that it is both ultra fast and ultra robust - not using Hugging Face downloader anymore            
      -   Moreover the downloader script verifies SHA 256 hash of the models and prevents any possibly corrupted model downloads                
   -   This is an interface app that is based on famous Kohya Musubi Tuner        
        -   It has all the features of Kohya Musubi Tuner + extra features with huge amount of details and very easy to use            
        -   Musubi Tuner Official repo : [https://github.com/kohya-ss/musubi-tuner](https://github.com/kohya-ss/musubi-tuner)            
   -   The installer will install with Torch 2.8 and CUDA 12.9 and pre-compiled xFormers, Triton, Flash Attention and Sage Attention libraries        
        -   We literally support all of the GPUs like RTX 3000, 4000, 5000 series or, A40, L40, A100, H100, B200 etc            
        -   Flash Attention and Sage Attention may not work on RTX 2000 or 1000 series but I have plan for them hopefully soon with newest libraries compiling myself            
   -   Currently it fully supports Qwen Image Models training, Wan 2.2 models training and Wan 2.1 models training        
        -   Also has automatic Qwen VL captioning


## Updates
**- Click images to see their full sizes**

### 25 June 2026 V28.4 Update

- Quantize model backend code fully updated
- Quantize presets fully updated and vastly improved
- Queue system added so set your settings and click start conversion it will auto queue
- Following presets fully added and default quality set to Normal - pretty good quality

<img height="600" alt="image" src="https://github.com/user-attachments/assets/6dfa218f-e053-4d0c-aad6-2acda950f637" />

- Full new screenshot of the quantize page as below
- I am currently testing FP8 Krea 2 Tensor vs Blockwise vs INT8 Blockwise
- Just run Windows_Install_and_Update.bat

<img height="600" alt="image" src="https://github.com/user-attachments/assets/43d31e3a-1839-4d16-b6be-7b09ea180b08" />

---

<img   height="600" alt="0010" src="https://github.com/user-attachments/assets/fc0f58c5-bdda-499a-9983-1db1f80236a1" />

---

<img   height="600" alt="0010" src="https://github.com/user-attachments/assets/696f344a-d904-4005-bdef-d9ab2520272a" />

---

<img   height="600" alt="44" src="https://github.com/user-attachments/assets/ad6b8c54-cbf2-4cc1-b8cb-91f4ae70d5ab" />

---

<img   height="600" alt="43" src="https://github.com/user-attachments/assets/b801e2ee-c4a1-4cbc-b83d-4cde74b40fab" />

---

<img   height="600" alt="42" src="https://github.com/user-attachments/assets/2bf3cfbc-b943-4268-8f2e-7fce65d23eed" />

---

<img  height="600" alt="41" src="https://github.com/user-attachments/assets/b6b5f050-65d6-4dc6-acb5-bb68fe2bec49" />

---

<img   height="600" alt="40" src="https://github.com/user-attachments/assets/acae6299-1d28-44b2-920a-1d04a3b85a3c" />

## Stage 1 Training Published Config Results Examples

30+ examples shared here : https://medium.com/@furkangozukara/qwen-image-lora-trainings-stage-1-results-and-pre-made-configs-published-as-low-as-training-with-ba0d41d76a05

<img   height="600" alt="04" src="https://github.com/user-attachments/assets/3c7c5fd0-fc0d-4978-8800-ed485054db22" />

---

<img   height="600" alt="03" src="https://github.com/user-attachments/assets/1e73012e-06f3-4181-ac49-ef10294d4119" />

---

<img   height="600" alt="02" src="https://github.com/user-attachments/assets/ce432e3f-5f74-4df2-bb43-0f911ab55184" />

---

<img   height="600" alt="01" src="https://github.com/user-attachments/assets/fe40513f-cfb9-4c91-9eb3-cfe45689e7a6" />

---

<img   height="600" alt="06" src="https://github.com/user-attachments/assets/f254bd98-1171-46e6-b1cb-8c5e1837dd5c" />

---

<img   height="600" alt="05" src="https://github.com/user-attachments/assets/cfa561ef-6323-4040-a8e2-05b43109ee1c" />


## App Screenshots

<img   height="600" alt="14" src="https://github.com/user-attachments/assets/11158887-696c-4eec-b5ee-6077f7a47873" />

---

<img   height="600" alt="13" src="https://github.com/user-attachments/assets/6f8e02ef-e37c-4da8-8eeb-41f873c236e9" />

---

<img   height="600" alt="12" src="https://github.com/user-attachments/assets/4b9d4871-4f24-4d09-b257-ea05b5e0ae58" />

---

<img   height="600" alt="11" src="https://github.com/user-attachments/assets/cbd2afaf-d4df-409c-a341-334122151e0b" />

---

<img   height="600" alt="10" src="https://github.com/user-attachments/assets/6cc09542-5c01-4685-a8d7-e8f90039fc8a" />

---

<img   height="600" alt="09" src="https://github.com/user-attachments/assets/e0e49752-4e32-46f1-b6ad-ea0f132fe461" />

---

<img   height="600" alt="08" src="https://github.com/user-attachments/assets/5cf90326-2351-4fb9-b486-a9244eddf81f" />

---

<img   height="600" alt="07" src="https://github.com/user-attachments/assets/cc010ec9-bfd6-4c35-b1da-b8fec426ffbf" />

---

<img   height="600" alt="06" src="https://github.com/user-attachments/assets/8ec0f71f-d4c4-4f34-a3d1-bedb062c3c69" />

---

<img   height="600" alt="05" src="https://github.com/user-attachments/assets/c6d12948-a2d5-413a-959e-be683b82f196" />

---

<img  height="600" alt="04" src="https://github.com/user-attachments/assets/55c43c0e-77dd-4435-91cb-a17b4960e4ad" />

---

<img   height="600" alt="03" src="https://github.com/user-attachments/assets/ed0b7055-dc99-419a-ae03-f377705a8a03" />

---

<img   height="600" alt="02" src="https://github.com/user-attachments/assets/4c6685ba-146e-4b1a-af50-3b93da24f472" />

---

<img   height="600" alt="01" src="https://github.com/user-attachments/assets/33ecd80c-6a53-41a1-bdc2-e35719bb9726" />





