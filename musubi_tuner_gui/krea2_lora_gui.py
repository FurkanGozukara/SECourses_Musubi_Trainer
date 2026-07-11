from .modern_image_lora_gui import modern_image_lora_tab


def krea2_lora_tab(headless=False, config={}):
    return modern_image_lora_tab("krea2", headless=headless, config=config)
