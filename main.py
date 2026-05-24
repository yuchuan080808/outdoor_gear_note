#!/usr/bin/env python3
"""
Drip-feed Amazon outdoor gear content pipeline.

Syncs Outdoor Recreation / Camping & Hiking BSR categories into tracking.json,
claims pending leaf categories in small batches, caches scraper output, generates
Markdown with an OpenAI-compatible LLM, and exports Hugo content files.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import shlex
import subprocess
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parent
DEFAULT_CATEGORY_JSON = ROOT / "data" / "outdoor_camping_bsr_urls.json"
TRACKING_JSON = ROOT / "data" / "tracking.json"
BESTSELLER_CACHE = ROOT / ".cache" / "bestsellers"
PRODUCT_CACHE = ROOT / ".cache" / "products"
OUTPUT_DIR = ROOT / "content"

LOGGER = logging.getLogger("amazon_outdoor_pipeline")
ASIN_RE = re.compile(
    r"(?:/dp/|/gp/product/|/product/|asin=)([A-Z0-9]{10})|"
    r"(?:^|[^A-Z0-9])([A-Z0-9]{10})(?:[^A-Z0-9]|$)"
)
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


@dataclass(frozen=True)
class CategoryTask:
    node_id: str
    category_path: str
    category_name: str
    bsr_url: str
    section: str


@dataclass(frozen=True)
class PublishedArticle:
    title: str
    url: str
    section: str
    category_name: str
    category_path: str
    source_path: Path | None = None


@dataclass(frozen=True)
class AuthorityResource:
    title: str
    url: str
    note: str
    keywords: tuple[str, ...]


@dataclass(frozen=True)
class SEOTopic:
    title: str
    description: str
    keywords: tuple[str, ...]
    faqs: tuple[tuple[str, str], ...]


class AutoCLITimeoutError(TimeoutError):
    def __init__(self, message: str, stdout: str = "", stderr: str = "") -> None:
        super().__init__(message)
        self.stdout = stdout
        self.stderr = stderr


def configure_logging() -> None:
    log_dir = ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    file_handler = logging.FileHandler(log_dir / "pipeline.log", encoding="utf-8")
    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    logging.basicConfig(
        level=logging.INFO,
        handlers=[stream_handler, file_handler],
        force=True,
    )


def load_dotenv(path: Path = ROOT / ".env") -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def slugify(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-") or "outdoor-gear"


def safe_json_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def section_label(section: str) -> str:
    return "Camping & Hiking" if section == "camping-hiking" else "Outdoor Recreation"


class TaskManager:
    VALID_STATUSES = {"pending", "processing", "completed", "failed"}
    SOURCE_SECTIONS = ("outdoor_recreation", "camping_hiking")

    def __init__(self, tracking_path: Path = TRACKING_JSON) -> None:
        self.tracking_path = tracking_path
        self.tracking_path.parent.mkdir(parents=True, exist_ok=True)
        self.state = self._load_state()

    def close(self) -> None:
        self.save()

    def _load_state(self) -> dict[str, Any]:
        if not self.tracking_path.exists():
            return {"version": 1, "last_synced": None, "categories": []}
        with self.tracking_path.open("r", encoding="utf-8") as f:
            state = json.load(f)
        state.setdefault("version", 1)
        state.setdefault("last_synced", None)
        state.setdefault("categories", [])
        return state

    def save(self) -> None:
        tmp_path = self.tracking_path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(self.state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp_path.replace(self.tracking_path)

    def sync_category_tree(self, category_json_path: Path) -> int:
        with category_json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        existing = {str(item["node_id"]): item for item in self.state["categories"]}
        now = utc_now()
        synced_count = 0

        for source_key in self.SOURCE_SECTIONS:
            nodes = data.get(source_key, [])
            leaves = self._find_leaf_categories(nodes)
            section = source_key.replace("_", "-")
            for node in leaves:
                node_id = str(node["node_id"])
                payload = {
                    "node_id": node_id,
                    "category_path": node["category_path"],
                    "category_name": node["category_name"],
                    "bsr_url": node["bsr_url"],
                    "section": section,
                }
                if node_id not in existing:
                    existing[node_id] = {**payload, "status": "pending", "last_updated": now}
                else:
                    existing[node_id].update(payload)
                    if existing[node_id].get("status") not in self.VALID_STATUSES:
                        existing[node_id]["status"] = "pending"
                        existing[node_id]["last_updated"] = now
                synced_count += 1

        self.state["categories"] = sorted(existing.values(), key=self._sort_key)
        self.state["last_synced"] = now
        self.save()
        LOGGER.info("Synced %s leaf rows into %s", synced_count, self.tracking_path)
        LOGGER.info("Tracking file now has %s unique tasks", len(self.state["categories"]))
        return synced_count

    @staticmethod
    def _find_leaf_categories(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        paths = [node.get("category_path", "") for node in nodes]
        leaves = []
        for node in nodes:
            path = node.get("category_path", "")
            if not path or not node.get("bsr_url"):
                continue
            is_parent = any(other != path and other.startswith(f"{path} >") for other in paths)
            if not is_parent:
                leaves.append(node)
        return leaves

    @classmethod
    def _sort_key(cls, item: dict[str, Any]) -> tuple[int, int, str, str]:
        path = item.get("category_path", "")
        section_rank = 0 if item.get("section") == "camping-hiking" else 1
        topic_rank = 0 if cls._is_tent_or_core_camping(path) else 1
        return (section_rank, topic_rank, item.get("last_updated", ""), item.get("node_id", ""))

    @staticmethod
    def _is_tent_or_core_camping(path: str) -> bool:
        text = path.lower()
        keywords = (
            "tent", "shelter", "camping furniture", "sleeping bag", "camp bedding",
            "camp kitchen", "lantern", "stove", "hydration", "backpack", "trekking pole",
        )
        return any(keyword in text for keyword in keywords)

    def get_next_batch(self, limit: int = 5) -> list[CategoryTask]:
        pending = [item for item in self.state["categories"] if item.get("status") == "pending"]
        pending.sort(key=self._sort_key)
        batch = pending[:limit]
        if not batch:
            return []

        now = utc_now()
        for item in batch:
            item["status"] = "processing"
            item["last_updated"] = now
        self.save()
        LOGGER.info("Claimed %s pending categories for processing", len(batch))
        return [
            CategoryTask(
                node_id=item["node_id"],
                category_path=item["category_path"],
                category_name=item["category_name"],
                bsr_url=item["bsr_url"],
                section=item["section"],
            )
            for item in batch
        ]

    def mark_completed(self, node_id: str, article_path: Path, article_url: str, title: str) -> None:
        self._set_status(
            node_id,
            "completed",
            {
                "article_path": str(article_path.relative_to(ROOT)),
                "article_url": article_url,
                "article_title": title,
            },
        )

    def mark_failed(self, node_id: str) -> None:
        self._set_status(node_id, "failed")

    def reset_failed_to_pending(self) -> int:
        return self._reset_status("failed")

    def reset_processing_to_pending(self) -> int:
        return self._reset_status("processing")

    def _reset_status(self, status: str) -> int:
        reset_count = 0
        now = utc_now()
        for item in self.state["categories"]:
            if item.get("status") == status:
                item["status"] = "pending"
                item["last_updated"] = now
                reset_count += 1
        if reset_count:
            self.save()
        return reset_count

    def get_related_articles(self, task: CategoryTask, limit: int = 5) -> list[PublishedArticle]:
        completed = [
            item for item in self.state["categories"]
            if item.get("status") == "completed"
            and item.get("article_title")
            and item.get("article_url")
            and item.get("node_id") != task.node_id
        ]
        same_section = [item for item in completed if item.get("section") == task.section]
        task_terms = self._path_terms(task.category_path)
        same_section.sort(
            key=lambda item: (
                len(task_terms & self._path_terms(item.get("category_path", ""))),
                item.get("last_updated", ""),
            ),
            reverse=True,
        )
        return [
            PublishedArticle(
                title=item["article_title"],
                url=item["article_url"],
                section=item["section"],
                category_name=item["category_name"],
                category_path=item.get("category_path", ""),
            )
            for item in same_section[:limit]
        ]

    @staticmethod
    def _path_terms(category_path: str) -> set[str]:
        stopwords = {"sports", "outdoors", "outdoor", "recreation", "camping", "hiking", "and", "the", "for"}
        terms = re.findall(r"[a-z0-9]+", category_path.lower())
        return {term for term in terms if term not in stopwords and len(term) > 2}

    def _set_status(self, node_id: str, status: str, extra: dict[str, Any] | None = None) -> None:
        if status not in self.VALID_STATUSES:
            raise ValueError(f"Invalid status: {status}")
        for item in self.state["categories"]:
            if item.get("node_id") == node_id:
                item["status"] = status
                item["last_updated"] = utc_now()
                if extra:
                    item.update(extra)
                self.save()
                return
        raise KeyError(f"Unknown node_id: {node_id}")


class SEOTopicCatalog:
    """Creates stable, human-readable SEO metadata for every outdoor leaf page."""

    OVERRIDES: dict[str, SEOTopic] = {
        "tents": SEOTopic(
            title="Best Tents for Camping, Rain, and Family Campsites",
            description="Compare camping tents by capacity, weather protection, ventilation, setup time, packed size, and buyer complaint patterns.",
            keywords=("best camping tents", "family camping tent", "tent buying guide", "waterproof camping tent", "easy setup tent"),
            faqs=(
                ("What should I check before buying a camping tent?", "Start with real sleeping capacity, rainfly coverage, ventilation, peak height, packed size, and whether the listing gives enough detail about stakes and guy lines."),
                ("Should I buy a larger tent than the listed capacity?", "Usually yes for car camping. A four-person tent often feels comfortable for two adults plus gear, while backpackers may accept tighter space to save weight."),
                ("What tent complaints matter most?", "Leaks, broken poles, weak zippers, condensation, and hard setup are the complaints that affect actual trips more than small cosmetic issues."),
            ),
        ),
        "backpacking-stoves": SEOTopic(
            title="Best Backpacking Stoves for Trail Cooking and Pack Weight",
            description="Compare backpacking stoves by fuel type, boil speed claims, stability, wind handling, packed size, and trail cooking tradeoffs.",
            keywords=("best backpacking stove", "lightweight backpacking stove", "canister stove", "trail cooking stove", "camp stove buying guide"),
            faqs=(
                ("What type of backpacking stove is easiest for beginners?", "Canister stoves are usually simplest because they are compact and easy to light, but wind, cold, and pot stability still matter."),
                ("Can one backpacking stove cook for a group?", "Only if the burner, pot support, and fuel plan match the group size. Tiny ultralight burners are better for boiling water than cooking real meals for several people."),
                ("What stove safety details should I check?", "Use stoves outside the tent, protect them from tipping, follow fuel instructions, and verify whether the pot and burner combination is stable before a trip."),
            ),
        ),
        "camping-stoves": SEOTopic(
            title="Best Camping Stoves for Car Camping and Group Meals",
            description="Compare camping stoves by burner layout, simmer control, wind resistance, fuel needs, cleanup, and campsite cooking fit.",
            keywords=("best camping stove", "two burner camp stove", "propane camping stove", "car camping stove", "camp kitchen stove"),
            faqs=(
                ("Is a two-burner camping stove worth it?", "For car camping and family meals, yes. Two burners make it easier to cook a main dish and heat water or sides at the same time."),
                ("What matters more than BTU claims?", "Pot stability, wind blocking, simmer control, fuel connection quality, and easy cleaning often matter more in everyday campsite cooking."),
                ("Can camping stoves be used inside a tent?", "No. Use fuel-burning stoves outdoors with ventilation and follow manufacturer instructions to avoid fire and carbon monoxide hazards."),
            ),
        ),
        "sleeping-bags": SEOTopic(
            title="Best Sleeping Bags for Camping Comfort and Cold Nights",
            description="Compare sleeping bags by temperature claims, shape, fill type, packability, zipper comfort, and real-use warmth cautions.",
            keywords=("best sleeping bag", "camping sleeping bag", "cold weather sleeping bag", "backpacking sleeping bag", "sleeping bag buying guide"),
            faqs=(
                ("How should I read sleeping bag temperature ratings?", "Treat them as guidance, not a guarantee. Your sleeping pad, clothing, wind exposure, metabolism, and moisture can change how warm a bag feels."),
                ("Is a mummy or rectangular sleeping bag better?", "Mummy bags save warmth and packed space, while rectangular bags feel roomier for car camping and restless sleepers."),
                ("What sleeping bag mistake causes the most regret?", "Buying only by the lowest advertised temperature without checking fit, zipper comfort, packed size, and whether the bag suits the actual season."),
            ),
        ),
        "coolers": SEOTopic(
            title="Best Coolers for Camping, Road Trips, and Ice Retention",
            description="Compare camping coolers by capacity, ice retention claims, portability, latch quality, drain design, and road-trip usability.",
            keywords=("best camping cooler", "cooler for camping", "ice retention cooler", "road trip cooler", "cooler buying guide"),
            faqs=(
                ("What cooler size should I buy for camping?", "Match capacity to trip length, group size, and whether food and drinks share space. Bigger coolers hold more ice but become heavy fast."),
                ("Are premium coolers always worth it?", "Not always. Premium insulation helps on long hot trips, but weekend campers may care more about weight, handles, drainage, and storage space."),
                ("How do I get better ice retention?", "Pre-chill the cooler, use block ice when possible, limit warm items, keep it shaded, and avoid opening it constantly."),
            ),
        ),
        "electric-lanterns": SEOTopic(
            title="Best Electric Lanterns for Camping, Power Outages, and Tents",
            description="Compare electric camping lanterns by brightness, runtime claims, recharge options, hanging design, durability, and tent safety.",
            keywords=("best electric lantern", "camping lantern", "rechargeable lantern", "tent lantern", "lantern for power outage"),
            faqs=(
                ("How bright should a camping lantern be?", "For tent use, lower modes matter as much as maximum output. Too much brightness in a small tent is annoying and drains batteries faster."),
                ("Are rechargeable lanterns better than battery lanterns?", "Rechargeable lanterns reduce disposable battery waste, but replaceable batteries can be easier on long trips without reliable charging."),
                ("Can I hang a lantern inside a tent?", "Electric lanterns are the safest choice inside tents. Keep heat-producing lanterns and fuel lanterns outside living spaces."),
            ),
        ),
        "fuel-lanterns": SEOTopic(
            title="Best Fuel Lanterns for Campsites and Cold-Weather Light",
            description="Compare fuel lanterns by brightness, fuel type, mantle care, stability, ventilation needs, and campsite safety tradeoffs.",
            keywords=("best fuel lantern", "propane lantern", "camping fuel lantern", "mantle lantern", "camp lantern buying guide"),
            faqs=(
                ("Are fuel lanterns safe inside tents?", "No. Fuel lanterns need outdoor ventilation and careful placement because flame, heat, and carbon monoxide risk are serious campsite hazards."),
                ("Why choose a fuel lantern over electric?", "Fuel lanterns can provide strong area light and work well in cold conditions, but they require mantles, fuel, ventilation, and more care."),
                ("What fuel lantern parts should I check?", "Look at the globe, mantle setup, fuel connection, base stability, and whether replacement parts are easy to find."),
            ),
        ),
        "water-filters": SEOTopic(
            title="Best Water Filters for Backpacking, Camping, and Emergencies",
            description="Compare outdoor water filters by flow rate, treatment limits, bottle compatibility, cleaning needs, and backcountry safety fit.",
            keywords=("best backpacking water filter", "camping water filter", "water filter for hiking", "emergency water filter", "backcountry water treatment"),
            faqs=(
                ("Do outdoor water filters remove every water risk?", "No. Filters vary by pore size and treatment target. Check whether the product addresses bacteria, protozoa, viruses, chemicals, or only taste."),
                ("What filter is easiest for beginners?", "Squeeze and bottle filters are usually simple, while gravity filters can be better for groups. Match the system to your water source and group size."),
                ("How do I avoid clogging a water filter?", "Pre-filter silty water when possible, backflush as instructed, avoid freezing wet filters, and store the filter clean and dry."),
            ),
        ),
        "backpacks": SEOTopic(
            title="Best Hiking Backpacks for Day Hikes, Travel, and Gear Fit",
            description="Compare hiking backpacks by capacity, torso fit, pocket layout, hydration support, rain protection, and comfort complaints.",
            keywords=("best hiking backpack", "day hiking backpack", "camping backpack", "hydration backpack", "backpack buying guide"),
            faqs=(
                ("What size backpack do I need for day hiking?", "Many day hikes work with 15 to 30 liters, depending on weather layers, water, food, first aid, and whether you carry gear for kids or pets."),
                ("What backpack fit details matter most?", "Torso length, shoulder strap shape, hip belt usefulness, back ventilation, and load stability matter more than a long feature list."),
                ("Should I buy a waterproof backpack?", "Most packs are water resistant, not truly waterproof. For wet trips, check rain-cover support and use dry bags for critical gear."),
            ),
        ),
        "pack-covers": SEOTopic(
            title="Best Backpack Rain Covers for Hiking and Wet Weather",
            description="Compare backpack rain covers by pack size fit, elastic grip, drainage, tear resistance, visibility, and storm-use limitations.",
            keywords=("best backpack rain cover", "pack cover", "hiking rain cover", "waterproof backpack cover", "rain cover for backpacking"),
            faqs=(
                ("Do backpack rain covers keep everything dry?", "They help with rain on the outside of the pack, but water can still run down your back panel or straps. Use dry bags for critical items."),
                ("How should a pack cover fit?", "It should wrap the loaded pack without overstretching and stay tight in wind. Oversized covers can flap, snag, or collect water."),
                ("Are bright pack covers useful?", "Yes for visibility in rain, roadside walking, or low light, though color matters less than secure fit and durable fabric."),
            ),
        ),
    }

    GROUP_PROFILES = (
        (
            ("tent", "shelter", "tarp", "canopy", "stake", "footprint", "privacy"),
            "Camping Weather Protection and Easy Setup",
            "weather protection, setup difficulty, ventilation, packed size, stability, and common durability complaints",
            ("weather protection", "easy setup", "camping shelter", "rain and wind", "packed size"),
            (
                ("What matters most for shelter gear?", "Prioritize real weather protection, stable pitch, ventilation, and whether the size fits your group and gear, not just the product photo."),
                ("What shelter specs are easy to overvalue?", "Peak height, capacity labels, and waterproof wording can be misleading without packed size, pole quality, stake quality, and user complaint patterns."),
            ),
        ),
        (
            ("sleep", "sleeping", "blanket", "cot", "hammock", "pillow", "pad", "bedding"),
            "Camping Sleep Comfort and Packability",
            "warmth, comfort, packed size, support, setup time, cleaning, and likely comfort tradeoffs",
            ("camp sleep system", "camping comfort", "packed size", "warmth", "support"),
            (
                ("How do I avoid regretting sleep gear?", "Match the item to your sleep style, expected temperature, packed-space limits, and whether you are car camping or carrying it on trail."),
                ("What sleep gear details matter after the first night?", "Noise, zipper feel, width, support, inflation or setup effort, and how easily the item dries or cleans can matter more than headline claims."),
            ),
        ),
        (
            ("stove", "cook", "kitchen", "cooler", "grill", "food", "utensil", "pot", "pan", "griddle"),
            "Camp Kitchen Setup and Meal Prep",
            "capacity, fuel or storage needs, cleanup, portability, durability, and campsite meal-prep tradeoffs",
            ("camp kitchen", "camp cooking", "car camping", "meal prep", "cleanup"),
            (
                ("What should I check before buying camp kitchen gear?", "Check group size, fuel or ice needs, cleanup effort, storage space, and whether the item is stable enough for the meals you actually cook."),
                ("What camp kitchen mistake is most common?", "Buying for an ideal trip instead of your real habits. If you cook simply, compact and easy-to-clean gear usually beats oversized specialty gear."),
            ),
        ),
        (
            ("lantern", "flashlight", "headlamp", "light", "glow", "battery", "solar"),
            "Camping Light, Runtime, and Emergency Use",
            "brightness modes, runtime claims, battery or charging options, hanging design, durability, and low-light safety",
            ("camp lighting", "runtime", "emergency light", "tent light", "brightness modes"),
            (
                ("How much brightness do I need for camping?", "You need enough light for cooking and walking, but low modes are important inside tents and around camp when you want battery life and less glare."),
                ("What lighting features matter most in bad weather?", "Water resistance, secure hanging or hands-free use, reliable switches, and a battery plan matter more than maximum lumen claims."),
            ),
        ),
        (
            ("water", "filter", "hydration", "canteen", "bottle", "reservoir", "bladder"),
            "Hiking Hydration and Backcountry Water Planning",
            "capacity, flow, cleaning, leak risk, pack compatibility, treatment limits, and trip-length fit",
            ("hiking hydration", "water capacity", "backcountry water", "leak prevention", "filter cleaning"),
            (
                ("How much water capacity should I carry?", "Capacity depends on heat, distance, water sources, and group needs. Carry more margin when conditions are hot or water access is uncertain."),
                ("What hydration complaints matter most?", "Leaks, hard cleaning, poor flow, plastic taste, and poor pack fit are the problems that get annoying quickly on trail."),
            ),
        ),
        (
            ("backpack", "daypack", "pack", "bag", "sack", "dry bag", "compression", "cover"),
            "Hiking Carry Comfort and Gear Organization",
            "capacity, fit, pocket layout, weather protection, compression, durability, and load comfort",
            ("hiking backpack", "gear organization", "pack fit", "rain protection", "trail comfort"),
            (
                ("What pack feature should I check first?", "Capacity matters, but fit and load comfort matter more. Look at straps, back panel, hip support, and whether pockets match your use."),
                ("How do I keep gear dry inside a pack?", "Use a rain cover for the outside and dry bags or liners for critical items. A cover alone is not a full waterproofing plan."),
            ),
        ),
        (
            ("fire", "starter", "first aid", "emergency", "survival", "compass", "whistle", "knife"),
            "Outdoor Safety, Backup Use, and Trip Prep",
            "reliability, packability, weather limits, ease of use, maintenance, and emergency backup value",
            ("outdoor safety", "emergency gear", "trip prep", "backup gear", "reliability"),
            (
                ("What safety gear should every hiker think about?", "Start with navigation, light, warmth, first aid, water, food, fire or emergency signaling, and weather awareness for the actual trip."),
                ("Why is cheap emergency gear risky?", "Emergency gear fails at the worst time if it is flimsy, hard to use, or untested. Practice with it before relying on it outdoors."),
            ),
        ),
    )

    DEFAULT_FAQS = (
        ("What should I check before buying this outdoor gear?", "Start with the real use case, weather exposure, packed size, weight, setup effort, cleaning, and the complaint pattern buyers mention most often."),
        ("Are Amazon bestseller rankings enough to choose gear?", "No. Bestseller signals help surface popular products, but the better choice depends on fit, safety, durability, and whether the product solves your specific trip problem."),
        ("How do I compare outdoor gear when prices change?", "Use broad price tiers, expected lifespan, replacement risk, and trip consequences. Cheap gear is fine when failure is low-impact and frustrating when it protects sleep, water, or safety."),
    )

    @classmethod
    def for_task(cls, task: CategoryTask) -> SEOTopic:
        return cls.for_values(task.section, task.category_name, task.category_path)

    @classmethod
    def for_article(cls, article: PublishedArticle) -> SEOTopic:
        return cls.for_values(article.section, article.category_name, article.category_path)

    @classmethod
    def for_values(cls, section: str, category_name: str, category_path: str = "") -> SEOTopic:
        slug = slugify(category_name)
        if slug in cls.OVERRIDES:
            return cls.OVERRIDES[slug]

        category_label = cls._title_case(category_name)
        lower_category = category_name.strip().lower()
        angle, focus, keyword_bits, group_faqs = cls._profile_for(category_name, category_path)
        section_phrase = cls._section_phrase(section)
        title = cls._trim_title(f"Best {category_label} for {angle}", category_label)
        description = cls._trim_description(
            f"Compare {lower_category} for {section_phrase} by {focus}."
        )
        keywords = cls._keywords(lower_category, section_phrase, keyword_bits)
        faqs = (
            (f"What should I check before buying {lower_category}?", f"Start with {focus}. Then check whether the product matches your trip length, weather, group size, and storage limits."),
            *group_faqs,
            (f"Who should skip budget {lower_category}?", "Skip the cheapest option when failure would affect sleep, water, cooking, warmth, or safety. Budget gear is safest when the downside is only inconvenience."),
        )
        return SEOTopic(title=title, description=description, keywords=keywords, faqs=faqs[:3])

    @classmethod
    def _profile_for(cls, category_name: str, category_path: str) -> tuple[str, str, tuple[str, ...], tuple[tuple[str, str], ...]]:
        haystack = f"{category_name} {category_path}".lower()
        for needles, angle, focus, keyword_bits, faqs in cls.GROUP_PROFILES:
            if any(needle in haystack for needle in needles):
                return angle, focus, keyword_bits, faqs
        return (
            "Camping, Hiking, and Outdoor Use",
            "fit, durability, portability, weather limits, setup friction, and common buyer regrets",
            ("outdoor gear", "camping gear", "hiking gear", "buyer cautions", "durability"),
            cls.DEFAULT_FAQS[:2],
        )

    @staticmethod
    def _title_case(value: str) -> str:
        title = re.sub(r"\s+", " ", value.replace("&", "and")).strip().title()
        return title.replace(" And ", " and ")

    @staticmethod
    def _trim_title(title: str, category_label: str) -> str:
        if len(title) <= 70:
            return title
        fallback = f"Best {category_label}: Buying Guide and Buyer Cautions"
        return fallback if len(fallback) <= 70 else f"Best {category_label}: Buying Guide"

    @staticmethod
    def _trim_description(description: str) -> str:
        description = re.sub(r"\s+", " ", description).strip()
        if len(description) <= 155:
            return description
        clipped = description[:152].rsplit(" ", 1)[0].rstrip(".,;:")
        clipped = re.sub(r"\b(?:and|or|for|with|by|to|the|a|an)$", "", clipped).rstrip(" ,;:-")
        return f"{clipped}."

    @staticmethod
    def _section_phrase(section: str) -> str:
        return section_label(section).lower().replace("&", "and")

    @staticmethod
    def _keywords(lower_category: str, section_phrase: str, keyword_bits: tuple[str, ...]) -> tuple[str, ...]:
        base = (
            f"best {lower_category}",
            f"{lower_category} buying guide",
            f"{lower_category} reviews",
            f"{lower_category} for {section_phrase}",
        )
        return tuple(dict.fromkeys((*base, *keyword_bits)))[:8]


class SEOResourceLinker:
    """Adds crawlable, contextual internal links and vetted authority links."""

    INTERNAL_LINK_LIMIT = 3
    RELATED_SECTION_RE = re.compile(
        r"\n*#{2,3}[ \t]+Related Resources[ \t]*\n.*?(?=\n#{2,3}[ \t]+(?:Comparison Table|Deep Reviews|Final Summary)\b|\Z)",
        re.DOTALL,
    )
    INSERT_TARGETS = (
        re.compile(r"(?m)^#{2,3}\s+Comparison Table\b"),
        re.compile(r"(?m)^#{2,3}\s+Deep Reviews\b"),
        re.compile(r"(?m)^#{2,3}\s+Final Summary\b"),
    )
    MARKDOWN_LINK_RE = re.compile(r"(?<!!)\[[^\]]+\]\((https?://[^)]+|/[^)]+)\)")
    FRONTMATTER_RE = re.compile(r"\A(---\s*\n.*?\n---\s*\n?)(.*)\Z", re.DOTALL)

    AUTHORITY_RESOURCES = (
        AuthorityResource(
            title="National Park Service Ten Essentials",
            url="https://www.nps.gov/articles/10essentials.htm",
            note="Baseline trip-prep context for safety, navigation, light, water, food, and emergency layers.",
            keywords=("default", "safety", "emergency", "backpack", "daypack", "hiking", "survival"),
        ),
        AuthorityResource(
            title="Weather.gov outdoor safety guidance",
            url="https://www.weather.gov/safety/",
            note="Weather-safety context for storms, heat, cold, wind, and lightning planning.",
            keywords=("weather", "tent", "shelter", "lightning", "rain", "cold", "heat", "lantern"),
        ),
        AuthorityResource(
            title="Leave No Trace Seven Principles",
            url="https://lnt.org/why/7-principles/",
            note="Low-impact camping guidance for campsites, cooking, waste, fires, and shared outdoor spaces.",
            keywords=("camp", "kitchen", "stove", "food", "fire", "default"),
        ),
        AuthorityResource(
            title="USDA Forest Service campfire safety",
            url="https://www.fs.usda.gov/visit/know-before-you-go/campfire-safety",
            note="Fire-safety background for stoves, fire starters, fuel, and campsite decisions.",
            keywords=("fire", "starter", "stove", "fuel", "grill", "lantern"),
        ),
        AuthorityResource(
            title="REI Expert Advice on the Ten Essentials",
            url="https://www.rei.com/learn/expert-advice/ten-essentials.html",
            note="Practical checklist context for packing and evaluating outdoor gear systems.",
            keywords=("pack", "backpack", "daypack", "hiking", "emergency", "navigation"),
        ),
        AuthorityResource(
            title="REI Expert Advice on backcountry water treatment",
            url="https://www.rei.com/learn/expert-advice/water-treatment-backcountry.html",
            note="Useful context for choosing filters, purifiers, and hydration plans.",
            keywords=("water", "filter", "hydration", "canteen", "bottle", "reservoir"),
        ),
        AuthorityResource(
            title="REI Expert Advice on choosing tents",
            url="https://www.rei.com/learn/expert-advice/family-base-camping-tent.html",
            note="Background on tent capacity, setup, weather protection, and campsite fit.",
            keywords=("tent", "shelter", "privacy", "canopy", "tarp"),
        ),
        AuthorityResource(
            title="REI Expert Advice on camp stoves",
            url="https://www.rei.com/learn/expert-advice/camp-stove.html",
            note="Fuel, burner, and camp-cooking context for comparing stove styles.",
            keywords=("stove", "kitchen", "cook", "grill", "fuel"),
        ),
    )
    RELATED_TOPIC_GROUPS = (
        ("camp shelter systems", frozenset(("tent", "shelter", "tarp", "canopy", "stake", "privacy"))),
        ("camp sleep systems", frozenset(("sleep", "sleeping", "blanket", "cot", "hammock", "pad", "pillow"))),
        ("camp kitchen", frozenset(("stove", "cook", "kitchen", "cooler", "food", "pot", "pan", "griddle"))),
        ("camp lighting", frozenset(("lantern", "flashlight", "headlamp", "light", "glow", "battery"))),
        ("hiking hydration", frozenset(("water", "filter", "hydration", "canteen", "bottle", "reservoir"))),
        ("packs and storage", frozenset(("backpack", "daypack", "pack", "bag", "sack", "cover", "compression"))),
        ("outdoor safety", frozenset(("fire", "starter", "first", "aid", "emergency", "survival", "compass"))),
    )

    @classmethod
    def enrich(
        cls,
        markdown_body: str,
        task: CategoryTask,
        related_articles: list[PublishedArticle] | None = None,
        current_url: str | None = None,
    ) -> str:
        body = cls._remove_related_resources(markdown_body)
        resource_section = cls._build_resource_section(
            task=task,
            related_articles=related_articles or [],
            body=body,
            current_url=current_url or cls._url_for(task),
        )
        return cls._insert_resource_section(body, resource_section)

    @classmethod
    def refresh_existing_content(cls, content_dir: Path = OUTPUT_DIR) -> int:
        articles = cls.collect_published_articles(content_dir)
        changed_count = 0
        for article in articles:
            if not article.source_path:
                continue
            original = article.source_path.read_text(encoding="utf-8")
            updated = cls.enrich_document(original, article, articles)
            if updated != original:
                article.source_path.write_text(updated, encoding="utf-8")
                changed_count += 1
        return changed_count

    @classmethod
    def collect_published_articles(cls, content_dir: Path = OUTPUT_DIR) -> list[PublishedArticle]:
        articles: list[PublishedArticle] = []
        for path in sorted(content_dir.rglob("*.md")):
            if path.name == "_index.md":
                continue
            document = path.read_text(encoding="utf-8")
            fields = cls._parse_frontmatter(document)
            if str(fields.get("draft", "")).lower() == "true":
                continue
            section = str(fields.get("section") or path.parent.name).strip().lower()
            if section not in {"camping-hiking", "outdoor-recreation"}:
                continue
            title = str(fields.get("title") or cls._title_from_filename(path.stem))
            category_path = str(fields.get("category_path") or "")
            category_name = cls._category_name(title, category_path)
            slug = str(fields.get("slug") or path.stem).strip("/")
            articles.append(
                PublishedArticle(
                    title=title,
                    url=f"/{section}/{slug}/",
                    section=section,
                    category_name=category_name,
                    category_path=category_path,
                    source_path=path,
                )
            )
        return articles

    @classmethod
    def enrich_document(cls, document: str, article: PublishedArticle, all_articles: list[PublishedArticle]) -> str:
        frontmatter, body = cls._split_frontmatter(document)
        task = CategoryTask(
            node_id="",
            category_path=article.category_path,
            category_name=article.category_name,
            bsr_url="",
            section=article.section,
        )
        body_without_resource_section = cls._remove_related_resources(body)
        related_articles = cls._rank_related_articles(task, all_articles, article.url, body_without_resource_section)
        updated_body = cls.enrich(body, task, related_articles=related_articles, current_url=article.url).strip()
        if not frontmatter:
            return updated_body + "\n"
        if updated_body == body.strip():
            return document
        return cls._touch_lastmod(frontmatter) + "\n\n" + updated_body + "\n"

    @classmethod
    def _build_resource_section(
        cls,
        task: CategoryTask,
        related_articles: list[PublishedArticle],
        body: str,
        current_url: str,
    ) -> str:
        lines = []
        for article in cls._rank_related_articles(task, related_articles, current_url, body)[: cls.INTERNAL_LINK_LIMIT]:
            lines.append(
                f"- **Related Guide:** [{article.title}]({article.url}) - "
                f"{cls._internal_link_note(task, article)}"
            )

        authority = cls._select_authority_resource(task, body)
        lines.append(f"- **Authority Reference:** [{authority.title}]({authority.url}) - {authority.note}")
        return "## Related Resources\n\n" + "\n".join(lines)

    @classmethod
    def _rank_related_articles(
        cls,
        task: CategoryTask,
        related_articles: list[PublishedArticle],
        current_url: str,
        body: str,
    ) -> list[PublishedArticle]:
        existing_urls = cls._extract_link_urls(body)
        task_terms = cls._link_terms(f"{task.category_name} {task.category_path}")
        task_groups = cls._topic_groups(task_terms)
        scored: list[tuple[int, str, PublishedArticle]] = []
        for article in related_articles:
            if article.url == current_url:
                continue
            article_terms = cls._link_terms(f"{article.title} {article.category_name} {article.category_path}")
            article_groups = cls._topic_groups(article_terms)
            same_section = 2 if article.section == task.section else 0
            overlap = len(task_terms & article_terms)
            topic_overlap = len(task_groups & article_groups)
            common_path_depth = cls._common_path_depth(task.category_path, article.category_path)
            score = (overlap * 10) + (topic_overlap * 8) + (common_path_depth * 4) + same_section
            if article.url in existing_urls:
                score -= 3
            scored.append((score, article.title, article))

        scored = [item for item in scored if item[0] > 0]
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [article for _, _, article in scored[: cls.INTERNAL_LINK_LIMIT]]

    @classmethod
    def _select_authority_resource(cls, task: CategoryTask, body: str) -> AuthorityResource:
        haystack = f"{task.category_name} {task.category_path}".lower()
        existing_urls = cls._extract_link_urls(body)

        def score(resource: AuthorityResource) -> int:
            return sum(1 for keyword in resource.keywords if keyword != "default" and keyword in haystack)

        ordered = sorted(cls.AUTHORITY_RESOURCES, key=lambda resource: (-score(resource), "default" in resource.keywords))
        matched = [resource for resource in ordered if score(resource) > 0]
        for resource in matched:
            if resource.url not in existing_urls:
                return resource
        if matched:
            return matched[0]
        for resource in ordered:
            if "default" in resource.keywords and resource.url not in existing_urls:
                return resource
        return ordered[0]

    @classmethod
    def _remove_related_resources(cls, markdown_body: str) -> str:
        return cls._collapse_blank_lines(cls.RELATED_SECTION_RE.sub("\n\n", markdown_body))

    @classmethod
    def _insert_resource_section(cls, markdown_body: str, resource_section: str) -> str:
        body = markdown_body.strip()
        for pattern in cls.INSERT_TARGETS:
            match = pattern.search(body)
            if match:
                prefix = body[: match.start()].rstrip()
                suffix = body[match.start() :].lstrip()
                return f"{prefix}\n\n{resource_section}\n\n{suffix}".strip()
        return f"{body}\n\n{resource_section}".strip()

    @classmethod
    def _parse_frontmatter(cls, document: str) -> dict[str, str]:
        match = cls.FRONTMATTER_RE.match(document)
        if not match:
            return {}
        frontmatter = match.group(1)
        fields: dict[str, str] = {}
        for line in frontmatter.splitlines():
            if not line or line == "---" or line.startswith(" ") or ":" not in line:
                continue
            key, value = line.split(":", 1)
            fields[key.strip()] = cls._frontmatter_value(value.strip())
        return fields

    @classmethod
    def _split_frontmatter(cls, document: str) -> tuple[str, str]:
        match = cls.FRONTMATTER_RE.match(document)
        if not match:
            return "", document
        return match.group(1).strip(), match.group(2)

    @staticmethod
    def _frontmatter_value(value: str) -> str:
        if not value:
            return ""
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return value.strip('"').strip("'")
        return str(decoded)

    @staticmethod
    def _touch_lastmod(frontmatter: str) -> str:
        now = utc_now()
        if re.search(r"(?m)^lastmod:\s*.*$", frontmatter):
            return re.sub(r"(?m)^lastmod:\s*.*$", f'lastmod: "{now}"', frontmatter, count=1)
        return frontmatter.replace("\n---", f'\nlastmod: "{now}"\n---', 1)

    @staticmethod
    def _extract_link_urls(markdown_body: str) -> set[str]:
        return {match.group(1).strip() for match in SEOResourceLinker.MARKDOWN_LINK_RE.finditer(markdown_body)}

    @staticmethod
    def _link_terms(value: str) -> set[str]:
        stopwords = {
            "and", "best", "buying", "camping", "for", "gear", "guide", "hiking", "outdoor",
            "outdoors", "recreation", "review", "reviews", "sports", "the", "use",
        }
        normalized_terms = set()
        for term in re.findall(r"[a-z0-9]+", value.lower()):
            if term in stopwords or len(term) <= 2:
                continue
            if term.endswith("ies") and len(term) > 4:
                term = term[:-3] + "y"
            elif term.endswith("s") and len(term) > 3:
                term = term[:-1]
            normalized_terms.add(term)
        return normalized_terms

    @classmethod
    def _topic_groups(cls, terms: set[str]) -> set[str]:
        return {group_name for group_name, group_terms in cls.RELATED_TOPIC_GROUPS if terms & group_terms}

    @staticmethod
    def _common_path_depth(left: str, right: str) -> int:
        ignored = {"sports & outdoors", "outdoor recreation", "camping & hiking"}
        left_parts = [part.strip().lower() for part in left.split(">") if part.strip() and part.strip().lower() not in ignored]
        right_parts = [part.strip().lower() for part in right.split(">") if part.strip() and part.strip().lower() not in ignored]
        depth = 0
        for left_part, right_part in zip(left_parts, right_parts):
            if left_part != right_part:
                break
            depth += 1
        return depth

    @staticmethod
    def _category_name(title: str, category_path: str) -> str:
        if category_path:
            return category_path.split(">")[-1].strip()
        match = re.match(r"Best\s+(.+?)(?:\s+for\b|:)", title, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return title.removeprefix("Best ").strip()

    @staticmethod
    def _title_from_filename(stem: str) -> str:
        words = stem.removeprefix("best-").replace("-", " ").title()
        return f"Best {words} for Outdoor Buyers"

    @staticmethod
    def _url_for(task: CategoryTask) -> str:
        return f"/{task.section}/best-{slugify(task.category_name)}/"

    @classmethod
    def _internal_link_note(cls, task: CategoryTask, article: PublishedArticle) -> str:
        shared_context = cls._shared_context(task.category_path, article.category_path)
        if shared_context == "outdoor gear":
            shared_groups = sorted(
                cls._topic_groups(cls._link_terms(f"{task.category_name} {task.category_path}"))
                & cls._topic_groups(cls._link_terms(f"{article.title} {article.category_name} {article.category_path}"))
            )
            if shared_groups:
                shared_context = shared_groups[0]
        return f"Compare nearby {shared_context} tradeoffs before you buy."

    @staticmethod
    def _shared_context(left: str, right: str) -> str:
        ignored = {"sports & outdoors", "outdoor recreation", "camping & hiking"}
        left_parts = [part.strip().lower() for part in left.split(">") if part.strip()]
        right_parts = [part.strip().lower() for part in right.split(">") if part.strip()]
        shared = [part for part in left_parts if part in right_parts and part not in ignored]
        return shared[-1] if shared else "outdoor gear"

    @staticmethod
    def _collapse_blank_lines(value: str) -> str:
        return re.sub(r"\n{3,}", "\n\n", value).strip()


class SEOArticleOptimizer:
    LEADING_H1_RE = re.compile(r"\A\s*#\s+.+?\n+")
    FAQ_SECTION_RE = re.compile(
        r"\n*## Common Questions Before Buying\n.*?(?=\n#{2,3}\s+(?:Related Resources|Comparison Table|Deep Reviews|Final Summary)\b|\Z)",
        re.DOTALL,
    )
    INSERT_TARGETS = (
        re.compile(r"(?m)^#{2,3}\s+Related Resources\b"),
        re.compile(r"(?m)^#{2,3}\s+Comparison Table\b"),
        re.compile(r"(?m)^#{2,3}\s+Deep Reviews\b"),
        re.compile(r"(?m)^#{2,3}\s+Final Summary\b"),
    )

    @classmethod
    def enrich_body(cls, markdown_body: str, task: CategoryTask) -> str:
        return cls._insert_faq_section(markdown_body, SEOTopicCatalog.for_task(task))

    @classmethod
    def refresh_existing_content(cls, content_dir: Path = OUTPUT_DIR) -> int:
        articles = SEOResourceLinker.collect_published_articles(content_dir)
        changed_count = 0
        for article in articles:
            if not article.source_path:
                continue
            original = article.source_path.read_text(encoding="utf-8")
            updated = cls.enrich_document(original, article)
            if updated != original:
                article.source_path.write_text(updated, encoding="utf-8")
                changed_count += 1
        return changed_count

    @classmethod
    def enrich_document(cls, document: str, article: PublishedArticle) -> str:
        frontmatter, body = SEOResourceLinker._split_frontmatter(document)
        if not frontmatter:
            return document
        topic = SEOTopicCatalog.for_article(article)
        updated_frontmatter = cls._update_frontmatter(frontmatter, topic)
        updated_body = cls._insert_faq_section(body, topic).strip()
        if updated_frontmatter == frontmatter and updated_body == body.strip():
            return document
        updated_frontmatter = SEOResourceLinker._touch_lastmod(updated_frontmatter)
        return f"{updated_frontmatter}\n\n{updated_body}\n"

    @classmethod
    def _insert_faq_section(cls, markdown_body: str, topic: SEOTopic) -> str:
        body = cls._remove_leading_h1(markdown_body)
        body = cls.FAQ_SECTION_RE.sub("\n\n", body).strip()
        faq_section = cls._faq_section(topic)
        for pattern in cls.INSERT_TARGETS:
            match = pattern.search(body)
            if match:
                prefix = body[: match.start()].rstrip()
                suffix = body[match.start() :].lstrip()
                return f"{prefix}\n\n{faq_section}\n\n{suffix}".strip()
        return f"{body}\n\n{faq_section}".strip()

    @classmethod
    def _remove_leading_h1(cls, markdown_body: str) -> str:
        return cls.LEADING_H1_RE.sub("", markdown_body, count=1).strip()

    @staticmethod
    def _faq_section(topic: SEOTopic) -> str:
        blocks = ["## Common Questions Before Buying"]
        for question, answer in topic.faqs:
            blocks.append(f"### {question}\n\n{answer}")
        return "\n\n".join(blocks)

    @classmethod
    def _update_frontmatter(cls, frontmatter: str, topic: SEOTopic) -> str:
        lines = frontmatter.splitlines()
        lines = cls._remove_yaml_block(lines, "keywords")
        lines = cls._set_scalar(lines, "title", topic.title)
        lines = cls._set_scalar(lines, "description", topic.description)
        insert_at = cls._line_index(lines, "description")
        keyword_lines = ["keywords:", *[f"  - {MarkdownExporter._yaml_quote(keyword)}" for keyword in topic.keywords]]
        if insert_at is None:
            insert_at = 1
        lines[insert_at + 1 : insert_at + 1] = keyword_lines
        return "\n".join(lines)

    @staticmethod
    def _line_index(lines: list[str], key: str) -> int | None:
        for index, line in enumerate(lines):
            if re.match(rf"^{re.escape(key)}\s*:", line):
                return index
        return None

    @classmethod
    def _set_scalar(cls, lines: list[str], key: str, value: str) -> list[str]:
        quoted_value = MarkdownExporter._yaml_quote(value)
        index = cls._line_index(lines, key)
        if index is None:
            lines.insert(1, f"{key}: {quoted_value}")
        else:
            lines[index] = f"{key}: {quoted_value}"
        return lines

    @staticmethod
    def _remove_yaml_block(lines: list[str], key: str) -> list[str]:
        output: list[str] = []
        index = 0
        while index < len(lines):
            line = lines[index]
            if re.match(rf"^{re.escape(key)}\s*:", line):
                index += 1
                while index < len(lines) and (lines[index].startswith(" ") or not lines[index].strip()):
                    index += 1
                continue
            output.append(line)
            index += 1
        return output


class ScraperEngine:
    def __init__(
        self,
        bestseller_cache: Path = BESTSELLER_CACHE,
        product_cache: Path = PRODUCT_CACHE,
        timeout_seconds: int = 180,
        bestsellers_command_template: str | None = None,
        product_command_template: str | None = None,
        autocli_path: str | None = None,
    ) -> None:
        self.bestseller_cache = bestseller_cache
        self.product_cache = product_cache
        self.timeout_seconds = timeout_seconds
        self.autocli_path = (autocli_path or os.environ.get("AUTOCLI_PATH") or "autocli").strip()
        autocli_command = self._quote_command_token(self.autocli_path)
        self.bestsellers_command_template = (
            bestsellers_command_template
            or os.environ.get("AUTOCLI_BESTSELLERS_COMMAND")
            or f"{autocli_command} amazon bestsellers {{url}} -f json"
        )
        self.product_command_template = (
            product_command_template
            or os.environ.get("AUTOCLI_PRODUCT_COMMAND")
            or f"{autocli_command} amazon product {{asin}} -f json"
        )
        self.bestseller_cache.mkdir(parents=True, exist_ok=True)
        self.product_cache.mkdir(parents=True, exist_ok=True)
        self._log_autocli_resolution()

    def scrape_category(self, task: CategoryTask, top_n: int = 20, min_success: int = 10) -> list[dict[str, Any]]:
        LOGGER.info("Scraping %s (%s)", task.category_name, task.node_id)
        bestseller_payload = self._cached_autocli_json(
            self._format_command(self.bestsellers_command_template, url=task.bsr_url, node_id=task.node_id),
            self.bestseller_cache / f"{safe_json_filename(task.node_id)}.json",
        )
        asins = self._extract_top_asins(bestseller_payload, limit=top_n)
        LOGGER.info("Found %s ASIN candidates for %s", len(asins), task.node_id)

        products: list[dict[str, Any]] = []
        for index, asin in enumerate(asins, start=1):
            LOGGER.info("Fetching product detail %s/%s for %s: %s", index, len(asins), task.node_id, asin)
            try:
                payload = self._cached_autocli_json(
                    self._format_command(self.product_command_template, asin=asin),
                    self.product_cache / f"{asin}.json",
                )
            except Exception as exc:
                LOGGER.warning("Skipping ASIN %s after product fetch failure: %s", asin, exc)
                continue
            compact = self._compact_product_payload(payload, asin)
            if compact:
                products.append(compact)

        if len(products) < min_success:
            raise RuntimeError(f"Only fetched {len(products)} usable products; need at least {min_success}")
        LOGGER.info("Fetched %s usable products for %s", len(products), task.node_id)
        return products

    def _cached_autocli_json(self, command: list[str], cache_path: Path) -> Any:
        cached = self._read_cache(cache_path)
        if cached is not None:
            LOGGER.info("Cache hit: %s", cache_path)
            return cached

        LOGGER.info("Cache miss; running: %s", " ".join(command))
        try:
            stdout, stderr = self._run_command_with_timeout(command)
        except OSError as exc:
            LOGGER.error("Failed to start AutoCLI command: %s", " ".join(command))
            LOGGER.error("AutoCLI start error: %s", exc)
            raise
        except AutoCLITimeoutError as exc:
            stdout, stderr = exc.stdout, exc.stderr
            payload = self._load_autocli_json(stdout, stderr, strict=False)
            if payload is not None:
                LOGGER.warning("AutoCLI timed out but returned valid JSON; saving cache anyway: %s", cache_path)
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                LOGGER.info("Saved cache: %s", cache_path)
                return payload
            self._save_autocli_raw_output(command, cache_path, stdout, stderr, "timeout_without_json")
            raise
        except subprocess.CalledProcessError as exc:
            LOGGER.error("AutoCLI failed with exit code %s", exc.returncode)
            if exc.stdout:
                LOGGER.error("AutoCLI stdout:\n%s", str(exc.stdout)[-4000:])
            if exc.stderr:
                LOGGER.error("AutoCLI stderr:\n%s", str(exc.stderr)[-4000:])
            payload = self._load_autocli_json(str(exc.stdout or ""), str(exc.stderr or ""), strict=False)
            if payload is not None:
                LOGGER.warning("AutoCLI failed but output contained valid JSON; saving cache anyway: %s", cache_path)
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                LOGGER.info("Saved cache: %s", cache_path)
                return payload
            self._save_autocli_raw_output(
                command,
                cache_path,
                str(exc.stdout or ""),
                str(exc.stderr or ""),
                f"exit_{exc.returncode}_without_json",
            )
            raise

        payload = self._load_autocli_json(stdout, stderr, strict=False)
        if payload is None:
            self._save_autocli_raw_output(command, cache_path, stdout, stderr, "success_without_json")
            LOGGER.error("AutoCLI exited successfully but no JSON payload could be parsed.")
            raise json.JSONDecodeError("AutoCLI did not return valid JSON", stdout or stderr or "", 0)

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        LOGGER.info("Saved cache: %s", cache_path)
        return payload

    @staticmethod
    def _load_autocli_json(stdout: str, stderr: str = "", strict: bool = True) -> Any | None:
        sources = [stdout or "", stderr or "", f"{stdout or ''}\n{stderr or ''}"]
        payloads: list[Any] = []
        for raw_text in sources:
            text = ScraperEngine._clean_autocli_text(raw_text)
            if not text:
                continue
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                pass

            for line in reversed(text.splitlines()):
                candidate = line.strip().rstrip(",")
                if candidate.startswith(("{", "[")):
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        continue

            payloads.extend(ScraperEngine._scan_json_payloads(text))
        if payloads:
            return max(payloads, key=ScraperEngine._json_payload_size)

        if strict:
            LOGGER.error("AutoCLI returned non-JSON stdout:\n%s", (stdout or "")[-4000:])
            if stderr.strip():
                LOGGER.error("AutoCLI stderr:\n%s", stderr[-4000:])
            raise json.JSONDecodeError("AutoCLI did not return valid JSON", stdout or "", 0)
        return None

    @staticmethod
    def _clean_autocli_text(text: str) -> str:
        text = ANSI_ESCAPE_RE.sub("", text or "")
        return text.replace("\x00", "").lstrip("\ufeff").strip()

    @staticmethod
    def _scan_json_payloads(text: str) -> list[Any]:
        decoder = json.JSONDecoder()
        payloads: list[Any] = []
        for index, char in enumerate(text):
            if char not in "{[":
                continue
            try:
                payload, _ = decoder.raw_decode(text[index:])
            except json.JSONDecodeError:
                continue
            payloads.append(payload)
        return payloads

    @staticmethod
    def _json_payload_size(payload: Any) -> int:
        try:
            return len(json.dumps(payload, ensure_ascii=False))
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _save_autocli_raw_output(command: list[str], cache_path: Path, stdout: str, stderr: str, reason: str) -> None:
        debug_dir = ROOT / "logs" / "autocli_raw"
        debug_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        safe_reason = safe_json_filename(reason) or "autocli"
        base = f"{stamp}-{safe_json_filename(cache_path.stem)}-{safe_reason}"
        metadata = {
            "reason": reason,
            "cache_path": str(cache_path),
            "command": command,
            "stdout_bytes": len((stdout or "").encode("utf-8", errors="replace")),
            "stderr_bytes": len((stderr or "").encode("utf-8", errors="replace")),
        }
        (debug_dir / f"{base}.meta.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        (debug_dir / f"{base}.stdout.txt").write_text(stdout or "", encoding="utf-8", errors="replace")
        (debug_dir / f"{base}.stderr.txt").write_text(stderr or "", encoding="utf-8", errors="replace")
        LOGGER.error("Saved raw AutoCLI output for inspection: %s", debug_dir / f"{base}.*")

    def _run_command_with_timeout(self, command: list[str]) -> tuple[str, str]:
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        use_shell = os.name == "nt"
        popen_command: list[str] | str = self._windows_shell_join(command) if use_shell else command
        start_time = time.monotonic()
        stdout_fd, stdout_name = tempfile.mkstemp(prefix="autocli-stdout-", suffix=".log")
        stderr_fd, stderr_name = tempfile.mkstemp(prefix="autocli-stderr-", suffix=".log")
        os.close(stdout_fd)
        os.close(stderr_fd)
        stdout_path = Path(stdout_name)
        stderr_path = Path(stderr_name)
        timed_out = False
        try:
            with stdout_path.open("w", encoding="utf-8", errors="replace") as stdout_file, stderr_path.open(
                "w", encoding="utf-8", errors="replace"
            ) as stderr_file:
                process = subprocess.Popen(
                    popen_command,
                    stdout=stdout_file,
                    stderr=stderr_file,
                    creationflags=creationflags,
                    shell=use_shell,
                )
                LOGGER.info("AutoCLI started with PID %s", process.pid)
                deadline = start_time + self.timeout_seconds
                while process.poll() is None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        timed_out = True
                        self._kill_process_tree(process)
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                        break
                    try:
                        process.wait(timeout=min(15, remaining))
                    except subprocess.TimeoutExpired:
                        LOGGER.info("AutoCLI still running after %.0fs (PID %s)", time.monotonic() - start_time, process.pid)

            stdout = stdout_path.read_text(encoding="utf-8", errors="replace")
            stderr = stderr_path.read_text(encoding="utf-8", errors="replace")
            elapsed = time.monotonic() - start_time
            if timed_out:
                LOGGER.error("AutoCLI timed out after %s seconds: %s", self.timeout_seconds, " ".join(command))
                if stdout:
                    LOGGER.error("AutoCLI stdout before timeout:\n%s", stdout[-4000:])
                if stderr:
                    LOGGER.error("AutoCLI stderr before timeout:\n%s", stderr[-4000:])
                raise AutoCLITimeoutError(
                    f"AutoCLI timed out after {self.timeout_seconds} seconds",
                    stdout=stdout,
                    stderr=stderr,
                )

            LOGGER.info("AutoCLI exited with code %s in %.1fs", process.returncode, elapsed)
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, command, output=stdout, stderr=stderr)
            return stdout, stderr
        finally:
            for temp_path in (stdout_path, stderr_path):
                try:
                    if temp_path.exists():
                        temp_path.unlink()
                except OSError:
                    LOGGER.debug("Could not remove temp AutoCLI log file: %s", temp_path)

    @staticmethod
    def _kill_process_tree(process: subprocess.Popen[str]) -> None:
        if process.poll() is not None:
            return
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(process.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            if process.poll() is None:
                process.kill()
            return
        process.kill()

    @staticmethod
    def _quote_command_token(token: str) -> str:
        token = token.strip()
        if not token:
            return "autocli"
        if len(token) >= 2 and token[0] in {"'", '"'} and token[-1] == token[0]:
            return token
        if os.name == "nt":
            return subprocess.list2cmdline([token])
        return shlex.quote(token)

    @staticmethod
    def _windows_shell_join(command: list[str]) -> str:
        parts: list[str] = []
        for arg in command:
            quoted = subprocess.list2cmdline([arg])
            if not quoted.startswith('"') and re.search(r'[&|<>^()]', arg):
                quoted = f'"{arg}"'
            parts.append(quoted)
        return " ".join(parts)

    def _log_autocli_resolution(self) -> None:
        executable = self.autocli_path.strip().strip('"').strip("'")
        if Path(executable).is_file():
            LOGGER.info("Using AutoCLI executable: %s", executable)
            return
        resolved = shutil.which(executable)
        if resolved:
            LOGGER.info("Using AutoCLI executable from PATH: %s", resolved)
            return
        LOGGER.warning(
            "AutoCLI executable was not found on PATH: %s. Set AUTOCLI_PATH to the full autocli executable path.",
            executable,
        )

    @staticmethod
    def _format_command(template: str, **values: str) -> list[str]:
        return shlex.split(template.format(**{key: shlex.quote(value) for key, value in values.items()}))

    @staticmethod
    def _read_cache(cache_path: Path) -> Any | None:
        if not cache_path.exists() or cache_path.stat().st_size == 0:
            return None
        try:
            with cache_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            LOGGER.warning("Ignoring invalid JSON cache file: %s", cache_path)
            return None

    def _extract_top_asins(self, payload: Any, limit: int) -> list[str]:
        asins: list[str] = []
        for item in self._iter_product_like_items(payload):
            asin = self._extract_asin(item)
            if asin and asin not in asins:
                asins.append(asin)
            if len(asins) >= limit:
                break
        return asins

    @staticmethod
    def _iter_product_like_items(payload: Any) -> Iterable[Any]:
        if isinstance(payload, list):
            yield from payload
            return
        if not isinstance(payload, dict):
            return
        for key in ("products", "items", "results", "bestsellers", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                yield from value
                return
        yield payload

    def _extract_asin(self, item: Any) -> str | None:
        if isinstance(item, dict):
            for key in ("asin", "ASIN", "product_asin", "productAsin"):
                value = item.get(key)
                if isinstance(value, str) and re.fullmatch(r"[A-Z0-9]{10}", value):
                    return value
            for key in ("url", "link", "product_url", "productUrl", "href"):
                value = item.get(key)
                if isinstance(value, str):
                    asin = self._extract_asin_from_text(value)
                    if asin:
                        return asin
            return self._extract_asin_from_text(json.dumps(item, ensure_ascii=False))
        if isinstance(item, str):
            return self._extract_asin_from_text(item)
        return None

    @staticmethod
    def _extract_asin_from_text(text: str) -> str | None:
        match = ASIN_RE.search(text)
        if not match:
            return None
        return match.group(1) or match.group(2)

    @staticmethod
    def _compact_product_payload(payload: Any, asin: str) -> dict[str, Any]:
        source = payload[0] if isinstance(payload, list) and payload and isinstance(payload[0], dict) else payload
        if not isinstance(source, dict):
            return {"asin": asin, "raw": source}

        def first(*keys: str) -> Any:
            for key in keys:
                if key in source and source[key] not in (None, "", []):
                    return source[key]
            return None

        image = first("image", "image_url", "imageUrl", "main_image", "mainImage", "thumbnail")
        if isinstance(image, dict):
            image = image.get("url") or image.get("src")

        return {
            "asin": asin,
            "title": first("title", "name", "product_title", "productTitle"),
            "price": first("price", "current_price", "currentPrice", "display_price", "displayPrice"),
            "rating": first("rating", "stars", "average_rating", "averageRating"),
            "review_count": first("review_count", "reviewCount", "ratings_count", "ratingsCount"),
            "image_url": image,
            "customers_say": first("customers_say", "customersSay", "customer_summary", "customerSummary"),
            "star_distribution": first("star_distribution", "starDistribution"),
            "reviews": first("reviews", "review_snippets", "reviewSnippets"),
            "features": first("features", "bullets", "bullet_points", "bulletPoints"),
            "specifications": first("specifications", "specs", "technical_details", "technicalDetails"),
            "ai_vision_report": first("ai_vision_report", "aiVisionReport", "vision_report", "visionReport"),
        }


class ContentGenerator:
    SYSTEM_PROMPT = """
You are a senior outdoor gear reviewer, technical camping editor, people-first SEO editor, and skeptical buyer advocate.
Write in English for US campers, hikers, road-trippers, family campers, overlanders, and beginner-to-intermediate outdoor buyers.

Banned phrases and tone rules:
- NEVER use these AI cliches: "delve into", "a testament to", "crucial", "in conclusion", "vital", "elevate", "realm", "bustling", "moreover", "furthermore", "tapestry", "game-changer", "unleash", "picture this", "navigate", "symphony", "undeniable", "paramount".
- Do not write a catalog rewrite. Write like a practical gear editor helping a buyer avoid a bad-fit purchase.
- Use short paragraphs, direct sentences, and scannable bold labels.
- Do not use robotic transitional phrases or summary paragraphs that add no new decision value.
- Every product section needs a distinct reason to exist; do not repeat the same praise.

Google-first SEO and helpful-content strategy:
- Match the provided primary SEO title, search intent, and target keywords naturally. Do not keyword-stuff.
- Build original decision value: buyer-fit matching, tradeoffs, red flags, use-case sorting, setup/storage/care tips, and mistake prevention.
- Use question-based H3 headers where they fit the buyer's long-tail search intent.
- Make the page useful even if the reader never clicks Amazon.
- Treat "best" as scenario-based, not absolute. Recommend by buyer need, trip type, and failure risk.
- Do not write primarily to manipulate search rankings; write to satisfy the real buyer's task.
- Do not create a standalone FAQ section or Related Resources section; the publishing pipeline adds those consistently after generation.

Evidence rules:
- Use only Product JSON facts and optional AI Vision Report fields. Do not invent field testing, measurements, waterproof ratings, season ratings, materials, certifications, dimensions, or warranty terms.
- Separate evidence from interpretation. Prefer phrases like "product-page details suggest", "customer-summary signals point to", and "this is likely best for".
- Do not claim personal field testing unless the input explicitly says hands-on testing was done.
- The How We Read This List section must explicitly say no hands-on field testing was conducted unless Product JSON says otherwise.
- Do not present Amazon bestseller status as proof of quality; treat it only as a marketplace popularity signal.
- Never output exact prices, discounts, coupons, sale language, or deal language. Convert price into broad tiers only.
- For safety-sensitive gear, remind readers to match equipment to weather, terrain, group size, skill level, and manufacturer limits.
- Do not make guaranteed survival, waterproof, warmth, food-safety, fuel-safety, or emergency-performance claims.

Link and compliance rules:
- Every purchase link must be exactly: [Check Price on Amazon](https://www.amazon.com/dp/{ASIN})
- Do not use affiliate, tracking, shortened, redirected, ref=, tag=, ascsubtag=, linkCode=, creative=, camp=, or query-parameter links.
- Use descriptive anchor text for internal links; avoid "click here".
- Use only the provided internal article URLs; do not invent internal links.
- Do not create a standalone authority-links block. If an authority link is truly useful inside the Buying Guide, use only nps.gov, weather.gov, noaa.gov, rei.com/learn, lnt.org, fs.usda.gov, cdc.gov, or redcross.org.

Image rules:
- For every individual product section, place the product image immediately under that product heading using Markdown:
  ![{SEO alt text with the product type and buyer use case}]({image_url})
- Alt text must be specific and functional, not just a product name. Prefer a use-case phrase such as "How to line a 45 qt cooler with {ProductName}" or "{ProductName} for hands-free campsite lighting".
- Keep alt text accurate to the image, under 145 characters, and free of keyword stuffing.
- If a product has no image_url, omit only the image line for that product.

Required Markdown structure:
- Do not output a Markdown H1. Hugo frontmatter supplies the page H1.
- Use these exact H2 headings without numeric prefixes:
  ## How We Read This List
  ## Quick Picks
  ## Buying Guide
  ## Comparison Table
  ## Deep Reviews
  ## Final Summary
- Before "## How We Read This List", write 2-4 tight introduction paragraphs, pain-first, no "Introduction" heading.
- Quick Picks: compact bullets naming the best product for 4-6 specific buyer needs.
- Buying Guide: practical criteria, red flags, fit/safety notes, and buying mistakes.
- Comparison Table: product, best for, standout upside, buyer caution, skip-if. Do not include price.
- Deep Reviews: exactly 10 products. Use H3 product headings. For each, include image if available, short verdict, best for, skip it if, what buyers may regret, complaint/watch-out pattern, pros, cons, Expert Tip, and clean Amazon link.
- Keep the complaint/watch-out pattern as one clean paragraph. Do not add star bars, score tables, Product schema, Review schema, price tables, or rating widgets in the generated body; the pipeline adds standardized visible feedback summaries and JSON-LD after generation.
- Final Summary: brief, scenario-based wrap-up.

Output Markdown body only. Do not output YAML frontmatter.
""".strip()

    def __init__(self, model: str | None = None, base_url: str | None = None, api_key: str | None = None) -> None:
        self.model = model or os.environ.get("LLM_MODEL") or os.environ.get("OPENAI_MODEL") or "qwen-plus"
        self.base_url = base_url or os.environ.get("LLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
        self.api_key = (
            api_key
            or os.environ.get("LLM_API_KEY")
            or os.environ.get("DASHSCOPE_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        )
        self.max_tokens = int(os.environ.get("LLM_MAX_TOKENS", "8000"))
        self.draft_model = os.environ.get("GEMINI_MODEL")
        self.draft_base_url = os.environ.get("GEMINI_BASE_URL")
        self.draft_api_key = os.environ.get("GEMINI_API_KEY")

    def generate(
        self,
        task: CategoryTask,
        products: list[dict[str, Any]],
        related_articles: list[PublishedArticle] | None = None,
    ) -> str:
        topic = SEOTopicCatalog.for_task(task)
        prompt_products = self._prepare_products_for_prompt(products)[:10]
        compact_json = json.dumps(prompt_products, ensure_ascii=False, separators=(",", ":"))
        seo_brief = (
            f"Primary SEO title: {topic.title}\n"
            f"Meta description: {topic.description}\n"
            f"Canonical URL path: {MarkdownExporter.url_for(task)}\n"
            f"Target keywords: {', '.join(topic.keywords)}\n"
            f"FAQ questions reserved for pipeline: {json.dumps([q for q, _ in topic.faqs], ensure_ascii=False)}\n"
            f"Category path: {task.category_path}\n"
            f"Category name: {task.category_name}\n"
            f"Site section: {section_label(task.section)}\n"
            "Search intent: help outdoor buyers choose the right product, avoid bad-fit purchases, and understand trip-specific tradeoffs before buying.\n"
        )
        user_prompt = (
            f"{seo_brief}"
            "Important: do not answer the reserved FAQ questions in a standalone FAQ section; the pipeline adds that section later.\n"
            f"Related internal articles already published:\n{self._related_articles_json(related_articles or [])}\n"
            f"Product JSON:\n{compact_json}\n"
        )
        LOGGER.info("Generating Markdown with model %s for %s", self.model, task.node_id)

        has_draft_model = bool(self.draft_model and self.draft_api_key)
        is_proxy_claude = "claude" in self.model.lower() and self.base_url
        is_native_anthropic = "claude" in self.model.lower() and not self.base_url

        if has_draft_model:
            LOGGER.info("Using %s to generate the initial article draft.", self.draft_model)
            draft_prompt = user_prompt + "\n\nCRITICAL DIRECTIVE: Generate the entire Markdown article with all 7 required sections."
            old_model, old_base_url, old_api_key = self.model, self.base_url, self.api_key
            self.model, self.base_url, self.api_key = self.draft_model, self.draft_base_url, self.draft_api_key
            try:
                draft_markdown = self._generate_openai(self.SYSTEM_PROMPT, draft_prompt)
            finally:
                self.model, self.base_url, self.api_key = old_model, old_base_url, old_api_key

            if os.environ.get("SKIP_CLAUDE_REFINEMENT", "").lower() == "true":
                body = draft_markdown
            else:
                refinement_prompt = (
                    "Rewrite and polish this draft to match the required editorial voice, SEO logic, and evidence limits. "
                    "Do not change product facts, ASINs, or the required 6 H2 headings. "
                    "Do not add a Markdown H1, FAQ section, or Related Resources section.\n\n"
                    f"=== ORIGINAL SEO BRIEF ===\n{seo_brief}=== END SEO BRIEF ===\n\n"
                    f"=== DRAFT ARTICLE ===\n{draft_markdown}\n=== END DRAFT ==="
                )
                body = self._generate_long_article(refinement_prompt, is_proxy_claude=is_proxy_claude)
        elif is_proxy_claude:
            body = self._generate_long_article(user_prompt, is_proxy_claude=True)
        elif is_native_anthropic:
            try:
                body = self._generate_claude(self.SYSTEM_PROMPT, user_prompt)
            except Exception as exc:
                LOGGER.warning("Claude messages streaming failed: %s; falling back to chat/completions", exc)
                body = self._generate_openai(self.SYSTEM_PROMPT, user_prompt)
        else:
            body = self._generate_openai(self.SYSTEM_PROMPT, user_prompt)

        body = self._enforce_clean_amazon_links(body)
        body = self._sanitize_external_links(body)
        body = self._sanitize_exact_prices(body)
        body = self._sanitize_unsupported_claims(body)
        body = self._sanitize_unsupported_vision_claims(body)
        body = GeneratedArticleEnhancer.enhance_body(body, task, products)
        body = SEOArticleOptimizer.enrich_body(body, task)
        body = SEOResourceLinker.enrich(body, task, related_articles=related_articles)
        return self._sanitize_external_links(body)

    def _generate_long_article(self, user_prompt: str, is_proxy_claude: bool) -> str:
        if not is_proxy_claude:
            return self._generate_openai(self.SYSTEM_PROMPT, user_prompt)

        LOGGER.info("Using chunked generation for proxy Claude to reduce timeout risk.")
        p1 = user_prompt + "\n\nCRITICAL DIRECTIVE: Only generate the intro plus these H2 sections: ## How We Read This List, ## Quick Picks, ## Buying Guide, and ## Comparison Table. Stop after the Comparison Table. Do not use numbered headings."
        body1 = self._generate_openai(self.SYSTEM_PROMPT, p1)
        p2 = user_prompt + "\n\nCRITICAL DIRECTIVE: Only generate ## Deep Reviews for the first 5 products. Start with exactly '## Deep Reviews'. Use H3 product headings. Do not generate intro, Quick Picks, Buying Guide, Comparison Table, FAQ, Related Resources, or Final Summary."
        body2 = self._generate_openai(self.SYSTEM_PROMPT, p2)
        p3 = user_prompt + "\n\nCRITICAL DIRECTIVE: Only generate the remaining product H3 reviews, then ## Final Summary. Do not output ## Deep Reviews again. Do not repeat intro, Quick Picks, Buying Guide, Comparison Table, FAQ, or Related Resources."
        body3 = self._generate_openai(self.SYSTEM_PROMPT, p3)
        return f"{body1}\n\n{body2}\n\n{body3}"

    def _generate_claude(self, system_prompt: str, user_prompt: str) -> str:
        import requests as _requests

        base = (self.base_url or "https://api.anthropic.com/v1").rstrip("/")
        if not base.endswith("/v1"):
            base += "/v1"
        url = f"{base}/messages"
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "stream": True,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
        }
        resp = self._post_with_retries(
            _requests,
            url,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "anthropic-version": "2023-06-01",
            },
            payload=payload,
        )
        if resp.status_code != 200:
            LOGGER.error("API error %s: %s", resp.status_code, resp.text[:2000])
            resp.raise_for_status()

        chunks: list[str] = []
        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            data_str = line[len("data: "):].strip()
            if data_str == "[DONE]":
                break
            try:
                event = json.loads(data_str)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "content_block_delta":
                delta = event.get("delta", {})
                if delta.get("type") == "text_delta":
                    chunks.append(delta.get("text", ""))
            elif event.get("type") == "error":
                raise RuntimeError(f"Claude stream error event: {event}")
            elif "choices" in event:
                for choice in event["choices"]:
                    delta = choice.get("delta", {})
                    if delta.get("content"):
                        chunks.append(delta["content"])
            elif isinstance(event.get("content"), str) and event["content"]:
                chunks.append(event["content"])

        if not chunks:
            raise RuntimeError("Claude streaming returned no content")
        return "".join(chunks).strip()

    def _generate_openai(self, system_prompt: str, user_prompt: str) -> str:
        if self.base_url:
            return self._generate_openai_streaming(system_prompt, user_prompt)

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("OpenAI SDK is not installed. Run: pip install -r requirements.txt") from exc

        client_kwargs: dict[str, str] = {}
        if self.api_key:
            client_kwargs["api_key"] = self.api_key
        client = OpenAI(**client_kwargs)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        if hasattr(client, "responses"):
            response = client.responses.create(model=self.model, input=messages)
            return response.output_text.strip()
        response = client.chat.completions.create(model=self.model, messages=messages)
        return response.choices[0].message.content.strip()

    def _generate_openai_streaming(self, system_prompt: str, user_prompt: str) -> str:
        import requests as _requests

        base = (self.base_url or "").rstrip("/")
        if not base.endswith("/v1") and "openai" not in base.lower():
            base += "/v1"
        url = f"{base}/chat/completions"
        payload = {
            "model": self.model,
            "stream": True,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        resp = self._post_with_retries(
            _requests,
            url,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
            payload=payload,
        )
        if resp.status_code != 200:
            LOGGER.error("API error %s: %s", resp.status_code, resp.text[:2000])
            resp.raise_for_status()

        resp.encoding = "utf-8"
        chunks: list[str] = []
        sample_events: list[str] = []
        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            data_str = line[len("data: "):].strip()
            if data_str == "[DONE]":
                break
            if len(sample_events) < 8:
                sample_events.append(data_str[:1000])
            try:
                event = json.loads(data_str)
            except json.JSONDecodeError:
                continue
            if "choices" in event:
                for choice in event["choices"]:
                    delta = choice.get("delta", {})
                    message = choice.get("message", {})
                    if delta.get("content"):
                        chunks.append(delta["content"])
                    elif message.get("content"):
                        chunks.append(message["content"])
            elif isinstance(event.get("content"), str) and event["content"]:
                chunks.append(event["content"])

        if not chunks:
            LOGGER.error("Streaming chat/completions returned no content. Sample events: %s", sample_events)
            raise RuntimeError("Streaming chat/completions returned no content")
        return "".join(chunks).strip()

    @staticmethod
    def _post_with_retries(_requests, url: str, headers: dict[str, str], payload: dict[str, Any]):
        last_resp = None
        for attempt in range(1, 4):
            resp = _requests.post(url, headers=headers, json=payload, timeout=600, stream=True)
            last_resp = resp
            if resp.status_code < 500:
                return resp
            LOGGER.warning("API returned %s on attempt %s/3; retrying", resp.status_code, attempt)
            try:
                resp.close()
            except Exception:
                pass
            import time as _time

            _time.sleep(10 * attempt)
        return last_resp

    @staticmethod
    def _related_articles_json(related_articles: list[PublishedArticle]) -> str:
        payload = [
            {
                "title": article.title,
                "url": article.url,
                "section": article.section,
                "category_name": article.category_name,
                "category_path": article.category_path,
            }
            for article in related_articles
        ]
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    @classmethod
    def _prepare_products_for_prompt(cls, products: list[dict[str, Any]]) -> list[dict[str, Any]]:
        prompt_products: list[dict[str, Any]] = []
        for product in products:
            item = dict(product)
            raw_price = item.pop("price", None)
            item["price_tier"] = cls._price_to_tier(raw_price)
            prompt_products.append(cls._compact_prompt_product(item))
        return prompt_products

    @classmethod
    def _compact_prompt_product(cls, item: dict[str, Any]) -> dict[str, Any]:
        limits = {
            "title": 220,
            "customers_say": 700,
            "reviews": 1000,
            "features": 900,
            "specifications": 900,
            "star_distribution": 500,
            "ai_vision_report": 900,
        }
        keep_keys = (
            "asin",
            "title",
            "rating",
            "review_count",
            "image_url",
            "customers_say",
            "star_distribution",
            "reviews",
            "features",
            "specifications",
            "ai_vision_report",
            "price_tier",
        )
        compact = {}
        for key in keep_keys:
            value = item.get(key)
            if value in (None, "", [], {}):
                continue
            compact[key] = cls._clip_jsonish(value, limits.get(key, 600))
        return compact

    @classmethod
    def _clip_jsonish(cls, value: Any, limit: int) -> Any:
        if isinstance(value, str):
            return cls._clip_text(value, limit)
        if isinstance(value, list):
            return [cls._clip_jsonish(item, max(160, limit // 4)) for item in value[:6]]
        if isinstance(value, dict):
            clipped = {}
            for key, subvalue in list(value.items())[:12]:
                clipped[key] = cls._clip_jsonish(subvalue, max(160, limit // 4))
            return clipped
        return value

    @staticmethod
    def _clip_text(value: str, limit: int) -> str:
        value = re.sub(r"\s+", " ", value).strip()
        if len(value) <= limit:
            return value
        return value[:limit].rsplit(" ", 1)[0] + "..."

    @staticmethod
    def _price_to_tier(raw_price: Any) -> str:
        if raw_price in (None, "", []):
            return "Price varies"
        price_text = json.dumps(raw_price, ensure_ascii=False) if not isinstance(raw_price, str) else raw_price
        numbers = [float(value.replace(",", "")) for value in re.findall(r"\d+(?:,\d{3})*(?:\.\d+)?", price_text)]
        if not numbers:
            return "Price varies"
        price = min(numbers)
        if price < 30:
            return "$ / Budget-friendly"
        if price < 100:
            return "$$ / Mid-range"
        return "$$$ / Premium price"

    @staticmethod
    def _enforce_clean_amazon_links(markdown: str) -> str:
        def clean_url(match: re.Match[str]) -> str:
            label, url = match.group(1), match.group(2)
            parsed = urlparse(url)
            if "amazon." not in parsed.netloc:
                return match.group(0)
            asin_match = re.search(r"/dp/([A-Z0-9]{10})", parsed.path)
            if not asin_match:
                return match.group(0)
            return f"[{label}](https://www.amazon.com/dp/{asin_match.group(1)})"

        return re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", clean_url, markdown)

    @staticmethod
    def _sanitize_external_links(markdown: str) -> str:
        allowed_domains = (
            "nps.gov",
            "weather.gov",
            "noaa.gov",
            "rei.com",
            "lnt.org",
            "fs.usda.gov",
            "cdc.gov",
            "redcross.org",
        )

        def sanitize(match: re.Match[str]) -> str:
            label, url = match.group(1), match.group(2).strip()
            if url.startswith(("/", "#")):
                return match.group(0)
            parsed = urlparse(url)
            host = parsed.netloc.lower().removeprefix("www.")
            if "amazon." in host:
                asin_match = re.search(r"/dp/([A-Z0-9]{10})", parsed.path)
                if asin_match:
                    return f"[{label}](https://www.amazon.com/dp/{asin_match.group(1)})"
                return label
            if any(host == domain or host.endswith(f".{domain}") for domain in allowed_domains):
                return match.group(0)
            return label

        return re.sub(r"(?<!!)\[([^\]]+)\]\((https?://[^)]+|/[^)]+|#[^)]+)\)", sanitize, markdown)

    @staticmethod
    def _sanitize_exact_prices(markdown: str) -> str:
        markdown = re.sub(r"(?i)(?:US\$|\$)\s*\d+(?:,\d{3})*(?:\.\d{1,2})?", "price tier", markdown)
        markdown = re.sub(r"(?i)\bUSD\s*\d+(?:,\d{3})*(?:\.\d{1,2})?", "price tier", markdown)
        markdown = re.sub(
            r"(?i)\b(?:sale|discount|deal|coupon|was|now)\b[^.\n]*(?:US\$|\$|USD)\s*\d+(?:,\d{3})*(?:\.\d{1,2})?",
            "price may vary",
            markdown,
        )
        return markdown

    @staticmethod
    def _sanitize_unsupported_claims(markdown: str) -> str:
        markdown = re.sub(
            r"(?i)\b(?:we|our team)\s+(?:field-tested|tested in the field|camped with|hiked with)\b",
            "we evaluated the available product-page evidence for",
            markdown,
        )
        markdown = re.sub(
            r"(?i)\b(?:waterproof to|rated to|tested to)\s+\d+[^.\n]*",
            "weather-resistance claims should be verified against the manufacturer listing",
            markdown,
        )
        markdown = re.sub(
            r"(?i)\b(?:guaranteed|guarantees)\s+(?:to\s+)?(?:keep you dry|keep food cold|work in all weather)[^.\n]*",
            "should be matched to the manufacturer's stated limits and the expected conditions",
            markdown,
        )
        return markdown

    @staticmethod
    def _sanitize_unsupported_vision_claims(markdown: str) -> str:
        markdown = re.sub(
            r"(?i)\b(?:we|our team|our ai(?: visual)? scanner)\s+(?:scanned|analyzed|reviewed)\s+"
            r"(?:over\s+|more than\s+)?\d[\d,]*\s+(?:customer\s+)?(?:photos|images|pictures|reviews)\b",
            "we reviewed the available customer-summary signals",
            markdown,
        )
        markdown = re.sub(
            r"(?i)\b(?:over\s+|more than\s+)?\d[\d,]*\s+(?:real\s+)?(?:customer\s+)?(?:photos|images|pictures)\b",
            "customer image signals",
            markdown,
        )
        return markdown


class GeneratedArticleEnhancer:
    """Post-processes generated reviews for image SEO, feedback summaries, and schema data."""

    ASIN_LINK_RE = re.compile(r"\[Check Price on Amazon\]\(https://www\.amazon\.com/dp/([A-Z0-9]{10})\)")
    IMAGE_RE = re.compile(r"!\[([^\]]*)\]\s*\((https?://[^)]+)\)")
    H3_RE = re.compile(r"^###\s+(.+?)\s*$")
    SUMMARY_BLOCK_RE = re.compile(
        r"\n*\*\*User Feedback Summary:\*\*\n\n\| Signal \| Read \|\n\| :--- \| :--- \|\n(?:\| .+? \|\n?){1,6}",
        re.DOTALL,
    )
    COMPLAINT_RE = re.compile(r"(?m)^(\*\*(?:Complaint/Watch-Out Pattern|Complaint Pattern(?: / Watch-Out Theme)?):\*\*.*)$")
    FALLBACK_RE = re.compile(r"(?m)^(\*\*(?:What Buyers May Regret|What buyers may regret):\*\*.*)$")

    @classmethod
    def enhance_body(cls, markdown_body: str, task: CategoryTask, products: list[dict[str, Any]]) -> str:
        metadata = cls.product_metadata(products)
        by_asin = {item["asin"]: item for item in metadata if item.get("asin")}
        body = cls.optimize_image_alt_text(markdown_body, task)
        return cls.insert_feedback_summaries(body, by_asin)

    @classmethod
    def refresh_existing_content(cls, content_dir: Path = OUTPUT_DIR, product_cache: Path = PRODUCT_CACHE) -> int:
        changed_count = 0
        for path in sorted(content_dir.rglob("*.md")):
            if path.name == "_index.md":
                continue
            original = path.read_text(encoding="utf-8")
            frontmatter, body = SEOResourceLinker._split_frontmatter(original)
            if not frontmatter:
                continue
            fields = SEOResourceLinker._parse_frontmatter(original)
            section = str(fields.get("section") or path.parent.name).strip()
            category_path = str(fields.get("category_path") or "")
            category_name = SEOResourceLinker._category_name(str(fields.get("title") or path.stem), category_path)
            task = CategoryTask("", category_path, category_name, "", section)
            products = cls._products_from_body(body, product_cache)
            metadata = cls.product_metadata(products)
            by_asin = {item["asin"]: item for item in metadata if item.get("asin")}
            updated_body = cls.insert_feedback_summaries(cls.optimize_image_alt_text(body, task), by_asin)
            updated_frontmatter = cls.update_products_frontmatter(frontmatter, metadata)
            if updated_frontmatter != frontmatter or updated_body.strip() != body.strip():
                updated_frontmatter = SEOResourceLinker._touch_lastmod(updated_frontmatter)
                path.write_text(f"{updated_frontmatter}\n\n{updated_body.strip()}\n", encoding="utf-8")
                changed_count += 1
        return changed_count

    @classmethod
    def optimize_image_alt_text(cls, markdown_body: str, task: CategoryTask) -> str:
        lines = markdown_body.splitlines()
        current_heading = ""
        output: list[str] = []
        for line in lines:
            heading_match = cls.H3_RE.match(line.strip())
            if heading_match:
                current_heading = heading_match.group(1).strip()
                output.append(line)
                continue

            image_match = cls.IMAGE_RE.search(line)
            if image_match and current_heading:
                url = image_match.group(2)
                alt = cls._functional_alt(current_heading, task)
                output.append(cls.IMAGE_RE.sub(f"![{alt}]({url})", line, count=1))
                continue
            output.append(line)
        return "\n".join(output).strip()

    @classmethod
    def insert_feedback_summaries(cls, markdown_body: str, product_by_asin: dict[str, dict[str, Any]]) -> str:
        if not product_by_asin:
            return markdown_body
        markdown_body = cls._remove_feedback_summaries(markdown_body)

        def enhance_section(match: re.Match[str]) -> str:
            section = match.group(0)
            asin_match = cls.ASIN_LINK_RE.search(section)
            if not asin_match:
                return section
            product = product_by_asin.get(asin_match.group(1))
            if not product:
                return section
            summary = cls._feedback_summary(product)
            inserted = False

            def insert_once(line_match: re.Match[str]) -> str:
                nonlocal inserted
                if inserted:
                    return line_match.group(0)
                inserted = True
                return f"{line_match.group(1)}\n\n{summary}"

            updated = cls.COMPLAINT_RE.sub(insert_once, section, count=1)
            if not inserted:
                updated = cls.FALLBACK_RE.sub(insert_once, section, count=1)
            if inserted:
                return updated
            return section.replace("\n[Check Price on Amazon]", f"\n{summary}\n\n[Check Price on Amazon]", 1)

        section_re = re.compile(r"(?ms)^###\s+.+?(?=^###\s+|\Z)")
        return section_re.sub(enhance_section, markdown_body).strip()

    @classmethod
    def _remove_feedback_summaries(cls, markdown_body: str) -> str:
        """Remove previously inserted feedback tables, including partial rows from older runs."""
        summary_rows = re.compile(
            r"^\|\s*(?:Signal|Marketplace rating|Positive signal|Pros signal|Evidence depth|Complaint risk|Complaint control|Complaint pressure|Watch-out pressure|Price tier)\s*\|"
        )
        lines = markdown_body.splitlines()
        output: list[str] = []
        index = 0

        while index < len(lines):
            line = cls._strip_inline_summary_fragment(lines[index])
            if line.strip() == "**User Feedback Summary:**":
                while output and not output[-1].strip():
                    output.pop()
                index += 1
                while index < len(lines) and not lines[index].strip():
                    index += 1
                while index < len(lines) and lines[index].lstrip().startswith("|"):
                    index += 1
                while index < len(lines) and not lines[index].strip():
                    index += 1
                if output and index < len(lines) and output[-1].strip() and lines[index].strip():
                    output.append("")
                continue
            if summary_rows.match(line.strip()):
                index += 1
                continue
            output.append(line)
            index += 1

        cleaned = "\n".join(output)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

    @staticmethod
    def _strip_inline_summary_fragment(line: str) -> str:
        review_label = (
            r"\*\*(?:What Buyers May Regret|What buyers may regret|"
            r"Complaint/Watch-Out Pattern|Complaint Pattern(?: / Watch-Out Theme)?):\*\*"
        )
        if re.match(rf"^{review_label}", line):
            line = re.sub(r"\s+\[[#-]{5}\]\s+[^|\n]*\|\s*$", "", line)
        return line

    @classmethod
    def product_metadata(cls, products: list[dict[str, Any]]) -> list[dict[str, Any]]:
        metadata: list[dict[str, Any]] = []
        seen: set[str] = set()
        for product in products:
            asin = str(product.get("asin") or "").strip()
            if not re.fullmatch(r"[A-Z0-9]{10}", asin) or asin in seen:
                continue
            seen.add(asin)
            raw_price = product.get("price")
            metadata.append(
                {
                    "asin": asin,
                    "title": cls._clean_text(str(product.get("title") or asin), 180),
                    "image": str(product.get("image_url") or "").strip(),
                    "amazon_url": f"https://www.amazon.com/dp/{asin}",
                    "rating": cls._parse_float(product.get("rating")),
                    "review_count": cls._parse_int(product.get("review_count")),
                    "rating_source": "amazon_marketplace",
                    "price_tier": ContentGenerator._price_to_tier(raw_price),
                    "feedback_note": cls._clean_text(str(product.get("customers_say") or ""), 220),
                }
            )
        return metadata

    @classmethod
    def update_products_frontmatter(cls, frontmatter: str, products: list[dict[str, Any]]) -> str:
        lines = cls._remove_yaml_block(frontmatter.splitlines(), "products")
        if not products:
            return "\n".join(lines)
        product_lines = ["products:"]
        for product in products[:10]:
            product_lines.append(f"  - asin: {MarkdownExporter._yaml_quote(str(product['asin']))}")
            product_lines.append(f"    title: {MarkdownExporter._yaml_quote(str(product['title']))}")
            if product.get("image"):
                product_lines.append(f"    image: {MarkdownExporter._yaml_quote(str(product['image']))}")
            product_lines.append(f"    amazon_url: {MarkdownExporter._yaml_quote(str(product['amazon_url']))}")
            if product.get("rating") is not None:
                product_lines.append(f"    rating: {product['rating']}")
            if product.get("review_count") is not None:
                product_lines.append(f"    review_count: {product['review_count']}")
            if product.get("rating_source"):
                product_lines.append(f"    rating_source: {MarkdownExporter._yaml_quote(str(product['rating_source']))}")
            if product.get("price_tier"):
                product_lines.append(f"    price_tier: {MarkdownExporter._yaml_quote(str(product['price_tier']))}")

        insert_at = len(lines) - 1 if lines and lines[-1] == "---" else len(lines)
        lines[insert_at:insert_at] = product_lines
        return "\n".join(lines)

    @classmethod
    def _products_from_body(cls, markdown_body: str, product_cache: Path) -> list[dict[str, Any]]:
        products: list[dict[str, Any]] = []
        seen: set[str] = set()
        for asin in cls.ASIN_LINK_RE.findall(markdown_body):
            if asin in seen:
                continue
            seen.add(asin)
            cache_path = product_cache / f"{asin}.json"
            if not cache_path.exists():
                products.append({"asin": asin})
                continue
            try:
                payload = json.loads(cache_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                products.append({"asin": asin})
                continue
            products.append(ScraperEngine._compact_product_payload(payload, asin))
        return products

    @classmethod
    def _feedback_summary(cls, product: dict[str, Any]) -> str:
        rating = product.get("rating")
        review_count = product.get("review_count")
        price_tier = str(product.get("price_tier") or "Price varies")
        note = str(product.get("feedback_note") or "").strip()
        rating_text = cls._rating_text(rating, review_count)
        evidence_score = cls._evidence_score(review_count)
        pros_score = cls._pros_score(rating, review_count)
        complaint_pressure = 6 - cls._cons_score(rating, note)
        risk_note = cls._risk_note(note)
        return (
            "**User Feedback Summary:**\n\n"
            "| Signal | Read |\n"
            "| :--- | :--- |\n"
            f"| Pros signal | {cls._score_label(pros_score, 'pros')} - {rating_text} |\n"
            f"| Evidence depth | {cls._score_label(evidence_score, 'evidence')} - {cls._evidence_text(review_count)} |\n"
            f"| Complaint pressure | {cls._score_label(complaint_pressure, 'complaint')} - {risk_note} |\n"
            f"| Price tier | {price_tier}; exact Amazon prices change frequently. |"
        )

    @staticmethod
    def _functional_alt(product_name: str, task: CategoryTask) -> str:
        name = GeneratedArticleEnhancer._clean_text(product_name, 90)
        category = task.category_name.lower()
        path = task.category_path.lower()
        if "cooler accessories" in category or "ice pack" in name.lower():
            alt = f"How to line a 45 qt camping cooler with {name}"
        elif "cooler" in category:
            alt = f"How to pack {name} for car camping food storage"
        elif "blanket" in category:
            alt = f"How to use {name} as a campsite blanket or ground cover"
        elif "flashlight" in category or "lantern" in path:
            alt = f"How to use {name} for hands-free campsite lighting"
        else:
            alt = f"How to use {name} for {category} outdoors"
        return GeneratedArticleEnhancer._clean_text(alt, 145)

    @staticmethod
    def _score_label(score: int, signal: str) -> str:
        score = max(1, min(5, score))
        labels = {
            "pros": {
                5: "Excellent buyer signal",
                4: "Strong buyer signal",
                3: "Moderate buyer signal",
                2: "Mixed buyer signal",
                1: "Weak buyer signal",
            },
            "evidence": {
                5: "Very strong evidence",
                4: "Strong evidence",
                3: "Moderate evidence",
                2: "Thin evidence",
                1: "Limited evidence",
            },
            "complaint": {
                5: "High complaint pressure",
                4: "Elevated complaint pressure",
                3: "Moderate complaint pressure",
                2: "Low complaint pressure",
                1: "Very low complaint pressure",
            },
        }
        return labels.get(signal, labels["evidence"])[score]

    @staticmethod
    def _pros_score(rating: float | None, review_count: int | None) -> int:
        if rating is None:
            return 3
        if rating >= 4.7 and (review_count or 0) >= 1000:
            return 5
        if rating >= 4.5:
            return 4
        if rating >= 4.2:
            return 3
        return 2

    @staticmethod
    def _cons_score(rating: float | None, note: str) -> int:
        risk_terms = ("mixed", "leak", "crack", "break", "thaw", "doesn't", "not last", "fragile", "puncture", "fail")
        risk_count = sum(1 for term in risk_terms if term in note.lower())
        base = 5 if rating and rating >= 4.6 else 4 if rating and rating >= 4.3 else 3
        return max(1, base - min(3, risk_count))

    @staticmethod
    def _evidence_score(review_count: int | None) -> int:
        if review_count is None:
            return 2
        if review_count >= 5000:
            return 5
        if review_count >= 1000:
            return 4
        if review_count >= 250:
            return 3
        if review_count >= 50:
            return 2
        return 1

    @staticmethod
    def _rating_text(rating: float | None, review_count: int | None) -> str:
        if rating is None:
            return "No reliable aggregate rating was available in the scraped product data."
        if review_count:
            return f"{rating:g}/5 across {review_count:,} Amazon ratings."
        return f"{rating:g}/5 Amazon rating signal; review volume was not available."

    @staticmethod
    def _evidence_text(review_count: int | None) -> str:
        if review_count is None:
            return "Limited review-volume data; treat patterns as directional."
        if review_count >= 5000:
            return "Very strong sample size for marketplace pattern reading."
        if review_count >= 1000:
            return "Strong sample size for recurring praise and complaint patterns."
        if review_count >= 250:
            return "Moderate sample size; useful but not exhaustive."
        return "Thin sample size; watch for pattern changes over time."

    @staticmethod
    def _risk_note(note: str) -> str:
        note = (note or "").strip()
        if not note or note.upper() == "N/A":
            return "No clear customer-summary complaint signal was available."
        complaint_patterns = [
            r"(?i)\b(?:however|but|while|although)\b[^.!?]*(?:mixed|others|some|report|complain|issue|problem|difficult|hard|leak|break|crack|rip|tear|fail|fragile|small|heavy|overpriced|smell|odor|scratchy|not\s+\w+)[^.!?]*[.!?]?",
            r"(?i)\b(?:the|its|their)?\s*(?:durability|fit|size|zipper|water resistance|brightness|value|taste|texture|comfort|assembly|weight|odor|smell|battery|leakage)\s+(?:receives?|gets?|is|are)\s+mixed[^.!?]*[.!?]?",
            r"(?i)\b(?:some|others|several|many)\s+(?:customers|buyers|reviewers|users)?[^.!?]*(?:report|say|note|mention|find|complain)[^.!?]*(?:break|broke|leak|rip|tear|fail|small|heavy|hard|difficult|overpriced|smell|odor|scratchy|not|issue|problem)[^.!?]*[.!?]?",
        ]
        for pattern in complaint_patterns:
            match = re.search(pattern, note)
            if match:
                return GeneratedArticleEnhancer._clean_text(match.group(0), 170)
        if re.search(r"(?i)\b(?:mixed|others report|some report|however|but)\b", note):
            sentence = re.split(r"(?<=[.!?])\s+", note)[-1]
            return GeneratedArticleEnhancer._clean_text(sentence, 170)
        return "No clear recurring complaint theme surfaced in the customer-summary data."

    @staticmethod
    def _parse_float(value: Any) -> float | None:
        if value in (None, "", []):
            return None
        match = re.search(r"\d+(?:\.\d+)?", str(value))
        return float(match.group(0)) if match else None

    @staticmethod
    def _parse_int(value: Any) -> int | None:
        if value in (None, "", []):
            return None
        match = re.search(r"\d[\d,]*", str(value))
        return int(match.group(0).replace(",", "")) if match else None

    @staticmethod
    def _clean_text(value: str, limit: int) -> str:
        value = re.sub(r"[\[\]\n\r]+", " ", value)
        value = re.sub(r"\s+", " ", value).strip().strip('"')
        if len(value) <= limit:
            return value
        return value[:limit].rsplit(" ", 1)[0].rstrip(".,;:") + "..."

    @staticmethod
    def _remove_yaml_block(lines: list[str], key: str) -> list[str]:
        output: list[str] = []
        index = 0
        while index < len(lines):
            line = lines[index]
            if re.match(rf"^{re.escape(key)}\s*:", line):
                index += 1
                while index < len(lines) and (lines[index].startswith(" ") or not re.match(r"^[A-Za-z_][A-Za-z0-9_-]*\s*:", lines[index])):
                    if lines[index] == "---":
                        break
                    index += 1
                continue
            output.append(line)
            index += 1
        return output


class MarkdownExporter:
    def __init__(self, output_dir: Path = OUTPUT_DIR) -> None:
        self.output_dir = output_dir

    def export(self, task: CategoryTask, markdown_body: str, products: list[dict[str, Any]] | None = None) -> tuple[Path, str, str]:
        target_dir = self.output_dir / task.section
        target_dir.mkdir(parents=True, exist_ok=True)
        filename = f"best-{slugify(task.category_name)}.md"
        target_path = target_dir / filename
        title = self.title_for(task)
        frontmatter = self._frontmatter(task, title, products=products or [])
        target_path.write_text(f"{frontmatter}\n\n{markdown_body.strip()}\n", encoding="utf-8")
        LOGGER.info("Wrote Markdown: %s", target_path)
        return target_path, self.url_for(task), title

    @staticmethod
    def title_for(task: CategoryTask) -> str:
        return SEOTopicCatalog.for_task(task).title

    @staticmethod
    def url_for(task: CategoryTask) -> str:
        return f"/{task.section}/best-{slugify(task.category_name)}/"

    @staticmethod
    def _frontmatter(task: CategoryTask, title: str, products: list[dict[str, Any]] | None = None) -> str:
        topic = SEOTopicCatalog.for_task(task)
        description = topic.description
        path_tags = [
            part.strip().lower()
            for part in task.category_path.split(">")
            if part.strip() and part.strip().lower() not in {"sports & outdoors", "outdoor recreation"}
        ]
        tags = list(
            dict.fromkeys(
                [
                    task.section,
                    task.category_name.lower(),
                    *path_tags,
                    "outdoor gear",
                    "camping gear",
                    "amazon best sellers",
                ]
            )
        )
        yaml_tags = "\n".join(f"  - {MarkdownExporter._yaml_quote(tag)}" for tag in tags)
        yaml_keywords = "\n".join(f"  - {MarkdownExporter._yaml_quote(keyword)}" for keyword in topic.keywords)
        product_lines = GeneratedArticleEnhancer.update_products_frontmatter("---\n---", GeneratedArticleEnhancer.product_metadata(products or [])).splitlines()
        yaml_products = "\n".join(line for line in product_lines if line not in {"---"})
        now = utc_now()
        return (
            "---\n"
            f"title: {MarkdownExporter._yaml_quote(title)}\n"
            f"description: {MarkdownExporter._yaml_quote(description)}\n"
            "keywords:\n"
            f"{yaml_keywords}\n"
            f"slug: {MarkdownExporter._yaml_quote(f'best-{slugify(task.category_name)}')}\n"
            f"date: {MarkdownExporter._yaml_quote(now)}\n"
            f"lastmod: {MarkdownExporter._yaml_quote(now)}\n"
            "draft: false\n"
            "categories:\n"
            f"  - {MarkdownExporter._yaml_quote(section_label(task.section))}\n"
            "tags:\n"
            f"{yaml_tags}\n"
            f"section: {MarkdownExporter._yaml_quote(task.section)}\n"
            f"amazon_node_id: {MarkdownExporter._yaml_quote(task.node_id)}\n"
            f"category_path: {MarkdownExporter._yaml_quote(task.category_path)}\n"
            f"{yaml_products + chr(10) if yaml_products else ''}"
            "---"
        )

    @staticmethod
    def _yaml_quote(value: str) -> str:
        return json.dumps(value, ensure_ascii=False)


def run_pipeline(args: argparse.Namespace) -> int:
    load_dotenv()
    if args.refresh_links_only:
        changed_count = SEOResourceLinker.refresh_existing_content(OUTPUT_DIR)
        LOGGER.info("Refreshed SEO resource links in %s article files", changed_count)
        return 0
    if args.refresh_content_only:
        changed_count = GeneratedArticleEnhancer.refresh_existing_content(OUTPUT_DIR, PRODUCT_CACHE)
        LOGGER.info("Refreshed generated article enhancements in %s article files", changed_count)
        return 0
    if args.refresh_seo_only:
        content_changed_count = GeneratedArticleEnhancer.refresh_existing_content(OUTPUT_DIR, PRODUCT_CACHE)
        seo_changed_count = SEOArticleOptimizer.refresh_existing_content(OUTPUT_DIR)
        link_changed_count = SEOResourceLinker.refresh_existing_content(OUTPUT_DIR)
        LOGGER.info(
            "Refreshed article enhancements in %s files, SEO metadata/FAQ in %s files, and resource links in %s files",
            content_changed_count,
            seo_changed_count,
            link_changed_count,
        )
        return 0

    task_manager = TaskManager(args.tracking_json)
    try:
        if args.reset_processing:
            LOGGER.info("Reset %s processing categories to pending", task_manager.reset_processing_to_pending())
        if args.retry_failed:
            LOGGER.info("Reset %s failed categories to pending", task_manager.reset_failed_to_pending())

        task_manager.sync_category_tree(args.category_json)
        if args.sync_only:
            LOGGER.info("Sync-only mode complete. No scraping or article generation was run.")
            return 0

        scraper = ScraperEngine(
            timeout_seconds=args.timeout,
            bestsellers_command_template=args.autocli_bestsellers_command,
            product_command_template=args.autocli_product_command,
            autocli_path=args.autocli_path,
        )
        generator = ContentGenerator(model=args.model, base_url=args.base_url, api_key=args.api_key)
        exporter = MarkdownExporter()

        batch = task_manager.get_next_batch(limit=args.batch_size)
        if not batch:
            LOGGER.info("No pending categories found. Nothing to do.")
            return 0

        success_count = 0
        failed_count = 0
        for task in batch:
            try:
                related_articles = task_manager.get_related_articles(task, limit=args.related_limit)
                products = scraper.scrape_category(task, top_n=args.top_n, min_success=args.min_products)
                markdown = generator.generate(task, products, related_articles=related_articles)
                article_path, article_url, title = exporter.export(task, markdown, products=products)
                task_manager.mark_completed(task.node_id, article_path=article_path, article_url=article_url, title=title)
                LOGGER.info("Completed category %s", task.node_id)
                success_count += 1
            except Exception as exc:
                LOGGER.exception("Failed category %s: %s", task.node_id, exc)
                task_manager.mark_failed(task.node_id)
                failed_count += 1
        if failed_count and not success_count:
            LOGGER.error("All %s claimed categories failed; stopping with exit code 1", failed_count)
            return 1
        if failed_count:
            LOGGER.warning("Completed %s categories with %s failures", success_count, failed_count)
        return 0
    finally:
        task_manager.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Drip-feed Amazon outdoor gear content pipeline")
    parser.add_argument("--batch-size", type=int, default=4, help="Number of pending leaf categories to process")
    parser.add_argument("--category-json", type=Path, default=DEFAULT_CATEGORY_JSON, help="Path to outdoor_camping_bsr_urls.json")
    parser.add_argument("--tracking-json", type=Path, default=TRACKING_JSON, help="Git-trackable status file path")
    parser.add_argument("--sync-only", action="store_true", help="Only sync category JSON into tracking.json; do not scrape or generate articles")
    parser.add_argument("--model", default=os.environ.get("LLM_MODEL") or os.environ.get("OPENAI_MODEL"), help="LLM model name")
    parser.add_argument("--base-url", default=os.environ.get("LLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL"), help="OpenAI-compatible API base URL")
    parser.add_argument("--api-key", default=os.environ.get("LLM_API_KEY"), help="LLM API key; defaults also check DASHSCOPE_API_KEY and OPENAI_API_KEY")
    parser.add_argument("--top-n", type=int, default=20, help="Top ASINs to fetch from each bestseller page")
    parser.add_argument("--min-products", type=int, default=10, help="Minimum successful product details per category")
    parser.add_argument("--related-limit", type=int, default=5, help="Completed articles to offer as internal-link candidates")
    parser.add_argument("--autocli-path", default=os.environ.get("AUTOCLI_PATH"), help="Full path to autocli if it is not on PATH")
    parser.add_argument("--autocli-bestsellers-command", default=os.environ.get("AUTOCLI_BESTSELLERS_COMMAND"), help='Command template, e.g. "autocli amazon bestsellers {url} -f json"')
    parser.add_argument("--autocli-product-command", default=os.environ.get("AUTOCLI_PRODUCT_COMMAND"), help='Command template, e.g. "autocli amazon product {asin} -f json"')
    parser.add_argument("--timeout", type=int, default=180, help="AutoCLI timeout in seconds per request")
    parser.add_argument("--reset-processing", action="store_true", help="Reset stuck processing rows back to pending before claiming a new batch")
    parser.add_argument("--retry-failed", action="store_true", help="Reset failed rows back to pending before claiming a new batch")
    parser.add_argument(
        "--refresh-links-only",
        action="store_true",
        help="Rebuild SEO Related Resources sections for existing content and exit",
    )
    parser.add_argument(
        "--refresh-content-only",
        action="store_true",
        help="Refresh image alt text, user feedback summaries, and product schema metadata for existing content, then exit",
    )
    parser.add_argument(
        "--refresh-seo-only",
        action="store_true",
        help="Refresh article titles, descriptions, keywords, FAQ sections, and SEO resource links, then exit",
    )
    return parser.parse_args()


def main() -> int:
    configure_logging()
    return run_pipeline(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
