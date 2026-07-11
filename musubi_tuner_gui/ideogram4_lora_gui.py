from .modern_image_lora_gui import modern_image_lora_tab


def ideogram4_lora_tab(headless=False, config={}):
    return modern_image_lora_tab("ideogram4", headless=headless, config=config)
