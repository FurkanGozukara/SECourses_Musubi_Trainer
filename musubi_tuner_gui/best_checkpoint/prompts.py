"""Test prompts of the best-checkpoint finder: the defaults (from SECourses' Qwen_Training_Tutorial_Prompts
"Grid_Find_Best_Checkpoint" set, written for a person subject "ohwx man"), the subject read from the dataset folder
names, and the prompt file the trainer samples from (one line per prompt and seed, --d set)."""

from __future__ import annotations

import os
import re

DEFAULT_SUBJECT = "ohwx man"

DEFAULT_PROMPTS = [
    "A classic and timeless studio headshot of ohwx man, focusing on trust and reliability. He is wearing a traditional black suit, a pristine white shirt, and a sophisticated, subtly textured burgundy tie and eyeglasses. The lighting is a simple, yet masterful \"loop lighting\" setup, where the key light creates a small \"loop\" of shadow from the nose, a technique known for being flattering and professional. A subtle hair light separates him from the classic, mottled, deep blue or grey painted canvas backdrop. His expression is a calm, gentle smile, looking directly at the viewer with warmth and sincerity, making the portrait perfect for a consultant, financial advisor, or any role where building client trust is paramount.",
    "A cinematic, wide-angle photograph of a futuristic starship pilot ohwx man, seated in the command chair on the bridge of his interstellar vessel as it navigates through a dense, vibrant nebula. He is dressed in a sleek, form-fitting navy blue flight suit adorned with glowing cyan piping and mission patches from various star systems. His hands rest confidently on the holographic controls projected from the armrests of his chair, displaying complex navigational data and star charts. He is wearing his signature eyeglasses, their lenses reflecting the swirling cosmic colors of the nebula outside the massive viewport that dominates the scene. The bridge is a marvel of advanced technology, with consoles, holographic displays, and crew stations all visible and in crisp focus, creating a sense of a fully operational and lived-in environment. The lighting is entirely diegetic, sourced from the soft glow of the control panels and the spectacular, multi-colored light of the purple, pink, and blue nebula pouring through the main viewport, which acts as a massive, natural softbox. The shot is taken from a low angle to emphasize the scale of the bridge and the captain's command. The camera used is a Canon EOS R5 with a 15-35mm f/2.8L lens, set to 18mm and an aperture of f/8 to ensure the entire bridge, from the foreground consoles to the distant stars, is perfectly sharp. The ISO is 800, and the shutter speed is 1/60s, capturing the static scene with immense detail and vibrant color.",
    "A clean, professional photograph of a news anchor ohwx man seated at his news desk in a modern television studio. He is dressed in a sharp, well-fitting suit and tie, looking directly into the camera as if delivering the news. He is holding a set of papers, and his expression is serious and authoritative. He is wearing his signature eyeglasses. The news desk is sleek and modern, with built-in screens. The background is the high-tech set of the news studio, with large video screens displaying world maps and graphics, and studio cameras visible, all in sharp focus. The lighting is the bright, even, and flattering lighting of a professional television studio, created with multiple softboxes and fill lights to eliminate all shadows. The photograph is taken with a Canon EOS C300 Mark III cinema camera, shooting a high-resolution still frame. The lens is a Canon CN-E 50mm T1.3 L F cinema lens, set to an aperture of f/8 to keep the entire set in focus. The ISO is 800, which is the camera's native ISO. The shutter speed is 1/120s. The image is a polished, professional portrait that conveys authority, trustworthiness, and the fast-paced world of broadcast journalism.",
    "A clean, high-key studio photograph of a master chef ohwx man, presented in a modern and professional style. He is dressed in a pristine, white chef's uniform, complete with a knotted neckerchief and a clean apron. He stands with his arms crossed, holding a well-honed chef's knife in one hand, its blade gleaming. He is wearing his signature eyeglasses, which are spotless and reflect the bright studio lights. The background is pure white and seamless, creating a completely distraction-free environment that emphasizes the chef's professionalism and cleanliness. The lighting is bright, even, and shadowless, achieved by using multiple large softboxes and umbrellas to surround the subject with light, bouncing it off the white walls, floor, and ceiling. This high-key setup creates a fresh, airy, and optimistic mood. The photograph is taken with a Canon EOS R6 Mark II and an RF 85mm f/1.2 L USM lens, known for its exceptional sharpness, but set to an aperture of f/11 to ensure every crease in his uniform and the sharp edge of his knife are perfectly in focus from tip to tip. The ISO is 100, and the shutter speed is 1/160s, synchronized with the powerful studio strobes. The final image is crisp, vibrant, and perfect for the cover of a cookbook or a feature in a culinary magazine.",
    "A powerful, environmental portrait of a seasoned firefighter ohwx man, a man in his early 40s with a soot-stained face and a look of weary heroism, standing in front of his fire engine in the station's garage. He has just returned from a call, and his expression is a mixture of exhaustion and pride. He wears a pair of durable, safety-rated eyeglasses, smudged but intact, a practical necessity of his job. He is still in his heavy, bunker gear, the reflective stripes catching the light, and he has taken off his helmet, which he holds under his arm. The fire engine behind him is a character in itself, a massive, gleaming red machine with all its chrome, gauges, and equipment rendered in sharp, intricate detail. The background of the fire station is also in clear focus, showing other equipment, hoses, and the concrete floor of the bay. The lighting is dramatic and slightly stylized, using the station's own harsh fluorescent overheads as a base, but augmented with a large, gridded softbox placed to the side to create a strong, directional key light that sculpts his face and highlights the texture of his gear. A second, bare-bulb strobe is placed behind him to create a sharp rim light, separating him from the engine and adding a heroic feel. This photograph is taken with a Nikon D850 and a Sigma Art 50mm f/1.4 lens set to f/8 for a deep and detailed image, capturing the grit, strength, and dedication of the subject.",
    "a photo of ohwx man wearing eyeglasses and sitting on a rooftop patio at twilight, with the city skyline lighting up behind him. This is a perfect use case for an iPhone's Portrait Mode, which has amplified city lights. He is in sharp focus, holding a orange juice and looking directly at the camera with a relaxed smile. The light is a mix of the fading blue ambient light from the sky and the warm string lights on the patio, and the iPhone's AWB (Auto White Balance) handles the combination beautifully. The edge detection on the portrait mode is good, but a keen eye can spot a few imperfections around his shoulders, a tell-tale sign of computational photography. The image feels stylish, modern, and celebratory, capturing the magic of a city evening.",
]

# a prompt without the trigger: the trained subject should not appear in it (identity leakage); {cls} is the class word
LEAKAGE_PROMPT = (
    "A candid photograph of a {cls} sitting at a small table in a busy cafe, soft natural window light, "
    "shallow depth of field, other customers blurred in the background."
)
FIRST_SEED = 1001
OPTION_RE = re.compile(r"(?<!\S)--?[a-zA-Z]{1,3}(?:\s|=|$)")

_REPEAT_RE = re.compile(r"^\d+_(.+)$")


def subject_from_dataset(folder: str) -> str:
    """The trigger and class of the first kohya-style sub-folder (``10_ohwx man`` -> ``ohwx man``), else ''."""
    try:
        names = sorted(entry.name for entry in os.scandir(folder) if entry.is_dir())
    except OSError:
        return ""
    for name in names:
        match = _REPEAT_RE.match(name)
        if match and match.group(1).strip():
            return match.group(1).strip()
    return ""


def suggest_subject(folder: str) -> str:
    """The subject for the prompts: the dataset folder's trigger, or the defaults' ``ohwx man`` when the folder only
    names its first word (``1_ohwx``) or nothing."""
    found = subject_from_dataset(folder) if folder else ""
    if not found or DEFAULT_SUBJECT.startswith(found.casefold()):
        return DEFAULT_SUBJECT
    return found


def class_word(subject: str) -> str:
    """The last word of the subject (``ohwx man`` -> ``man``): what the prompt means without the trigger."""
    words = subject.split()
    return words[-1] if len(words) > 1 else ""


def apply_subject(prompts: list[str], subject: str) -> list[str]:
    """Replace the defaults' ``ohwx man`` with the user's subject (case-insensitive); other prompts stay as typed."""
    subject = subject.strip()
    if not subject or subject.casefold() == DEFAULT_SUBJECT:
        return list(prompts)
    pattern = re.compile(re.escape(DEFAULT_SUBJECT), re.IGNORECASE)
    return [pattern.sub(subject, prompt) for prompt in prompts]


def clean_prompts(rows) -> list[str]:
    """Prompts from the UI table / text (one per row or line); empty rows, comments and <lora:...> tags dropped."""
    if isinstance(rows, str):
        rows = rows.splitlines()
    prompts = []
    for row in rows or []:
        text = row[0] if isinstance(row, (list, tuple)) else row
        text = re.sub(r"<lora:[^>]*>|[\x00-\x08\x0b-\x1f\x7f]", "", str(text or "")).strip()
        if text and not text.startswith("#") and text not in prompts:
            prompts.append(text)
    return prompts


def prompt_file_lines(prompts: list[str], seeds: list[int]) -> list[str]:
    """One sampling line per prompt and seed; the seed is explicit, so every checkpoint draws the same noise."""
    return [f"{prompt} --d {seed}" for prompt in prompts for seed in seeds]


def leakage_prompt(subject: str) -> str:
    return LEAKAGE_PROMPT.format(cls=class_word(subject) or "person")


def seeds(count: int) -> list[int]:
    """The same seeds for every prompt and checkpoint: the samples differ only by the weights."""
    return [FIRST_SEED + index for index in range(max(1, int(count)))]


def prompt_text(line: str) -> str:
    """A sample prompt line without its options (``--w 1024 --d 42 ...``): what the image should show."""
    match = OPTION_RE.search(line)
    return (line[: match.start()] if match else line).strip()
