import os
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter

BASE_DIR = Path(__file__).resolve().parent.parent
CORPUS_DIR = BASE_DIR / "data" / "corpus"
CACHE_FILE = BASE_DIR / "data" / "vision_cache.json"

CORPUS_DIR.mkdir(parents=True, exist_ok=True)

CATEGORIES_SPEC = {
    "fox": {
        "count": 10,
        "base_color": (217, 83, 30),  # Red-orange
        "subject": "red fox",
        "category": "animal",
        "captions": [
            "A vibrant red fox standing in an autumn forest clearing.",
            "Close-up of a wild red fox with alert pointed ears and bushy tail.",
            "A red fox hunting through fresh winter snow.",
            "Wild Vulpes vulpes foraging quietly among pine needles.",
            "A slender red fox resting under golden birch trees.",
            "Wild red fox kit playing in tall meadow grasses.",
            "A sharp portrait of a red fox with vibrant reddish-orange fur.",
            "A red fox stalking small prey across an open woodland.",
            "Adult red fox walking along a misty forest trail.",
            "A curious red fox looking directly toward the camera."
        ],
        "attributes": ["orange fur", "bushy tail", "wildlife", "forest", "canid", "carnivore"]
    },
    "wolf": {
        "count": 10,
        "base_color": (110, 115, 120),  # Gray
        "subject": "gray wolf",
        "category": "animal",
        "captions": [
            "A magnificent gray wolf standing atop a rocky winter cliff.",
            "A solitary timber wolf prowling through a cold northern forest.",
            "Pack of gray wolves resting together in snowy pine woods.",
            "Profile portrait of an adult gray wolf with piercing amber eyes.",
            "A majestic gray wolf trotting through misty tundra terrain.",
            "A wild wolf howling against a backdrop of spruce trees.",
            "Close-up of a gray wolf with thick silver-grey winter coat.",
            "A lone gray wolf surveying the northern wilderness.",
            "An alert gray wolf scanning the horizon at dusk.",
            "Adult wolf pausing by a frozen mountain stream."
        ],
        "attributes": ["gray fur", "pack predator", "wilderness", "carnivore", "winter", "howling"]
    },
    "dog": {
        "count": 10,
        "base_color": (220, 190, 140),  # Golden / tan
        "subject": "domestic dog",
        "category": "animal",
        "captions": [
            "A friendly golden retriever dog sitting in a sunny suburban backyard.",
            "A domestic puppy playing happily with a toy on a green lawn.",
            "A loyal family dog resting indoors by a warm hearth.",
            "Close-up portrait of a happy domestic dog panting with a smile.",
            "A golden retriever fetching a ball in an urban dog park.",
            "A well-groomed domestic dog walking politely on a leash.",
            "A playful dog splashing through shallow lake waters.",
            "A sleeping domestic dog curled up comfortably on a rug.",
            "A gentle pet dog sitting attentively beside its companion.",
            "A cheerful golden puppy exploring backyard garden flowers."
        ],
        "attributes": ["domesticated", "pet", "golden fur", "friendly", "canine", "companion"]
    },
    "bear": {
        "count": 8,
        "base_color": (90, 60, 40),  # Brown
        "subject": "brown bear",
        "category": "animal",
        "captions": [
            "A massive brown grizzly bear catching wild salmon in a river.",
            "A brown bear mother guiding two playful cubs across a meadow.",
            "A large brown bear foraging for wild berries in coastal valleys.",
            "Close-up of a grizzly bear with thick brown fur standing in shallow water.",
            "A brown bear scratching its back against a tall cedar tree.",
            "A solitary brown bear traversing an alpine hillside.",
            "A grizzly bear resting peacefully in tall grass near a river.",
            "An adult brown bear scanning the rapids for migrating fish."
        ],
        "attributes": ["large mammal", "brown fur", "grizzly", "river", "salmon", "wildlife"]
    },
    "deer": {
        "count": 8,
        "base_color": (160, 120, 80),  # Fawn / brown
        "subject": "white-tailed deer",
        "category": "animal",
        "captions": [
            "A majestic white-tailed deer buck with large antlers in morning mist.",
            "A graceful white-tailed deer doe grazing peacefully in a meadow.",
            "A pair of wild deer standing alert at the forest edge.",
            "A spotted deer fawn resting quietly in sheltered tall ferns.",
            "A white-tailed stag leaping gracefully over a woodland creek.",
            "Close-up portrait of a wild deer with velvet-covered antlers.",
            "A female deer foraging for tender leaves in late autumn woods.",
            "A group of white-tailed deer grazing in a golden sunrise meadow."
        ],
        "attributes": ["antlers", "herbivore", "white-tailed", "meadow", "forest", "graceful"]
    }
}

EDGE_CASES = [
    {
        "filename": "blurry_mystery_01.jpg",
        "color": (80, 80, 80),
        "blur": True,
        "metadata": {
            "subject": "unknown",
            "category": "unidentified",
            "attributes": ["blurry", "low contrast", "unclear"],
            "caption": "A blurry, low-contrast image where the subject cannot be determined.",
            "confidence": 0.42
        }
    },
    {
        "filename": "abstract_noise_02.jpg",
        "color": (60, 60, 70),
        "blur": True,
        "metadata": {
            "subject": "abstract pattern",
            "category": "distortion",
            "attributes": ["noise", "grain", "indistinct"],
            "caption": "An abstract, grainy image with severe visual distortion.",
            "confidence": 0.38
        }
    }
]

def generate_corpus():
    vision_cache = {}

    for cat_key, spec in CATEGORIES_SPEC.items():
        base_color = spec["base_color"]
        subject = spec["subject"]
        category = spec["category"]
        captions = spec["captions"]
        attributes = spec["attributes"]

        for i in range(spec["count"]):
            filename = f"{cat_key}_{i+1:02d}.jpg"
            img_path = CORPUS_DIR / filename

            # Generate image with subtle variations
            r_var = (i * 7) % 30 - 15
            g_var = (i * 11) % 30 - 15
            b_var = (i * 13) % 30 - 15
            color = (
                max(0, min(255, base_color[0] + r_var)),
                max(0, min(255, base_color[1] + g_var)),
                max(0, min(255, base_color[2] + b_var))
            )

            img = Image.new("RGB", (320, 240), color=color)
            draw = ImageDraw.Draw(img)
            # Draw simple geometric shapes
            draw.rectangle([40, 40, 280, 200], outline=(255, 255, 255), width=2)
            draw.ellipse([80, 60, 240, 180], fill=(color[0] // 2, color[1] // 2, color[2] // 2))

            img.save(img_path, format="JPEG", quality=85)

            caption = captions[i]
            conf = 0.90 + ((i % 8) * 0.01)

            vision_cache[filename] = {
                "subject": subject,
                "category": category,
                "attributes": attributes,
                "caption": caption,
                "confidence": round(conf, 2)
            }

    # Edge cases
    for ec in EDGE_CASES:
        filename = ec["filename"]
        img_path = CORPUS_DIR / filename
        img = Image.new("RGB", (320, 240), color=ec["color"])
        draw = ImageDraw.Draw(img)
        draw.line([(0, 0), (320, 240)], fill=(120, 120, 120), width=4)
        if ec.get("blur"):
            img = img.filter(ImageFilter.GaussianBlur(radius=8))
        img.save(img_path, format="JPEG", quality=60)
        vision_cache[filename] = ec["metadata"]

    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(vision_cache, f, indent=2)

    total_images = len(list(CORPUS_DIR.glob("*.jpg")))
    print(f"Successfully generated {total_images} images in {CORPUS_DIR}")
    print(f"Generated vision cache with {len(vision_cache)} entries in {CACHE_FILE}")

if __name__ == "__main__":
    generate_corpus()
