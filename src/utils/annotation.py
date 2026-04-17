from __future__ import annotations

import random
import string
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from .data_utils import sentence_split, tokenize_text


@dataclass
class AnnotatedSentence:
    topic: str
    tokens: list[str]
    pos_tags: list[str]
    ner_tags: list[str]


POS_TAGS = [
    "NOUN",
    "VERB",
    "ADJ",
    "ADV",
    "PRON",
    "DET",
    "CONJ",
    "POST",
    "NUM",
    "PUNC",
    "UNK",
]

NER_TAGS = ["B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG", "B-MISC", "I-MISC", "O"]

PRONOUNS = {
    "mai",
    "main",
    "hum",
    "ham",
    "tum",
    "aap",
    "wo",
    "yeh",
    "yah",
    "unhon",
    "inhon",
    "mujhe",
    "humein",
    "us",
    "iss",
}

DETERMINERS = {
    "ye",
    "woh",
    "is",
    "us",
    "koi",
    "har",
    "kuch",
    "tamam",
    "aksar",
    "ziyada",
    "kam",
}

CONJUNCTIONS = {
    "aur",
    "lekin",
    "magar",
    "ya",
    "agar",
    "to",
    "phir",
    "balke",
    "jab",
    "kyunke",
    "tahum",
}

POSTPOSITIONS = {
    "mein",
    "main",
    "par",
    "per",
    "se",
    "tak",
    "ke",
    "ki",
    "ka",
    "ko",
    "liye",
    "baad",
    "pehle",
}

VERB_SUFFIXES = ("na", "ta", "ti", "te", "ga", "gi", "ge", "raha", "rahi", "rahe")
ADJ_SUFFIXES = ("i", "een", "ana", "iya", "dar")
ADV_SUFFIXES = ("tor", "war", "an", "se")


def infer_topics_from_metadata(metadata: object, num_docs: int) -> list[str]:
    """Infer per-document topic labels from metadata with schema tolerance."""

    def pick_label(item: dict) -> str | None:
        for key in ["topic", "category", "label", "section", "genre"]:
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    labels: list[str] = []
    if isinstance(metadata, list):
        for item in metadata:
            if isinstance(item, dict):
                lbl = pick_label(item)
                if lbl:
                    labels.append(lbl)
    elif isinstance(metadata, dict):
        if isinstance(metadata.get("articles"), list):
            for item in metadata["articles"]:
                if isinstance(item, dict):
                    lbl = pick_label(item)
                    if lbl:
                        labels.append(lbl)
        else:
            for value in metadata.values():
                if isinstance(value, dict):
                    lbl = pick_label(value)
                    if lbl:
                        labels.append(lbl)

    if len(labels) < num_docs:
        labels.extend(["unknown"] * (num_docs - len(labels)))
    return labels[:num_docs]


def sample_balanced_sentences(
    documents: list[str],
    topics: list[str],
    total_sentences: int = 500,
    min_per_topic: int = 100,
    min_topics: int = 3,
    seed: int = 42,
) -> list[tuple[str, str]]:
    """Sample a topic-balanced sentence set: [(topic, sentence), ...]."""
    if len(documents) != len(topics):
        raise ValueError("documents and topics must have same length")

    rng = random.Random(seed)
    topic_to_sentences: dict[str, list[str]] = defaultdict(list)

    for doc, topic in zip(documents, topics):
        sentences = sentence_split(doc)
        for sent in sentences:
            if sent:
                topic_to_sentences[topic].append(sent)

    if not topic_to_sentences:
        raise ValueError("No sentences extracted from documents")

    ranked_topics = sorted(topic_to_sentences.keys(), key=lambda t: len(topic_to_sentences[t]), reverse=True)
    selected_topics = ranked_topics[: min_topics]

    sampled: list[tuple[str, str]] = []
    used_by_topic: dict[str, set[int]] = defaultdict(set)

    for topic in selected_topics:
        candidates = topic_to_sentences[topic]
        if not candidates:
            continue
        k = min(min_per_topic, len(candidates))
        idxs = list(range(len(candidates)))
        rng.shuffle(idxs)
        chosen = idxs[:k]
        used_by_topic[topic].update(chosen)
        sampled.extend((topic, candidates[i]) for i in chosen)

    remaining_pool: list[tuple[str, str]] = []
    for topic, sents in topic_to_sentences.items():
        for i, sent in enumerate(sents):
            if i in used_by_topic[topic]:
                continue
            remaining_pool.append((topic, sent))

    rng.shuffle(remaining_pool)
    needed = max(0, total_sentences - len(sampled))
    sampled.extend(remaining_pool[:needed])

    if len(sampled) < total_sentences:
        raise ValueError(
            f"Not enough sentences for requested sample size: got {len(sampled)}, need {total_sentences}"
        )

    rng.shuffle(sampled)
    return sampled[:total_sentences]


def build_rule_lexicon(token_counter: Counter[str], min_per_major_tag: int = 200) -> dict[str, str]:
    """Construct a large rule lexicon by combining seeds and frequency-based heuristics."""
    lexicon: dict[str, str] = {}

    for tok in PRONOUNS:
        lexicon[tok] = "PRON"
    for tok in DETERMINERS:
        lexicon[tok] = "DET"
    for tok in CONJUNCTIONS:
        lexicon[tok] = "CONJ"
    for tok in POSTPOSITIONS:
        lexicon[tok] = "POST"

    major_tags = ["NOUN", "VERB", "ADJ"]
    by_tag: dict[str, set[str]] = {tag: set() for tag in major_tags}

    candidates = [tok for tok, _ in token_counter.most_common() if tok and tok.isalpha()]
    for tok in candidates:
        lower = tok.lower()
        if lower in lexicon:
            continue

        if any(lower.endswith(s) for s in VERB_SUFFIXES):
            by_tag["VERB"].add(lower)
            continue
        if any(lower.endswith(s) for s in ADJ_SUFFIXES):
            by_tag["ADJ"].add(lower)
            continue

        by_tag["NOUN"].add(lower)

    for tag in major_tags:
        picked = 0
        for tok in sorted(by_tag[tag]):
            if tok not in lexicon:
                lexicon[tok] = tag
                picked += 1
            if picked >= min_per_major_tag:
                break

    return lexicon


def _is_punctuation(token: str) -> bool:
    return all(ch in string.punctuation for ch in token)


def _is_numeric(token: str) -> bool:
    return token.replace(",", "").replace(".", "").isdigit()


def tag_pos(tokens: list[str], lexicon: dict[str, str]) -> list[str]:
    """Assign rule-based POS tags."""
    tags: list[str] = []
    for tok in tokens:
        lower = tok.lower()

        if _is_punctuation(tok):
            tags.append("PUNC")
            continue
        if _is_numeric(tok):
            tags.append("NUM")
            continue
        if lower in lexicon:
            tags.append(lexicon[lower])
            continue
        if any(lower.endswith(s) for s in VERB_SUFFIXES):
            tags.append("VERB")
            continue
        if any(lower.endswith(s) for s in ADV_SUFFIXES):
            tags.append("ADV")
            continue
        if any(lower.endswith(s) for s in ADJ_SUFFIXES):
            tags.append("ADJ")
            continue

        tags.append("NOUN")
    return tags


def default_gazetteer() -> dict[str, list[str]]:
    """Seed gazetteer with persons, locations, organisations, misc entities."""
    persons = [
        "imran khan",
        "nawaz sharif",
        "bilawal bhutto",
        "asif zardari",
        "shahbaz sharif",
        "maryam nawaz",
        "altaf hussain",
        "fazlur rehman",
        "siraj ul haq",
        "mohsin naqvi",
        "babar azam",
        "mohammad rizwan",
        "shaheen afridi",
        "haris rauf",
        "fakhar zaman",
        "shadab khan",
        "wasim akram",
        "waqar younis",
        "misbah ul haq",
        "younis khan",
        "abdul sattar",
        "abdus salam",
        "allama iqbal",
        "faiz ahmad faiz",
        "qurat ul ain",
        "saadat hasan manto",
        "ahmad faraz",
        "parveen shakir",
        "javed miandad",
        "inzamam ul haq",
        "sarfaraz ahmad",
        "moin khan",
        "rashid latif",
        "aqib javed",
        "saeed anwar",
        "saqlain mushtaq",
        "abdul qadir",
        "shahid afridi",
        "shoib akhtar",
        "muhammad ali jinnah",
        "liaquat ali khan",
        "zulfiqar ali bhutto",
        "benazir bhutto",
        "pervez musharraf",
        "ashfaq kayani",
        "qamar bajwa",
        "asim munir",
        "abdul hafeez",
        "ishaq dar",
        "khawaja asif",
        "hamza shahbaz",
        "usman buzdar",
        "chaudhry nisar",
        "fawad chaudhry",
    ]

    locations = [
        "islamabad",
        "karachi",
        "lahore",
        "peshawar",
        "quetta",
        "multan",
        "faisalabad",
        "rawalpindi",
        "hyderabad",
        "sialkot",
        "gujranwala",
        "bahawalpur",
        "sukkur",
        "larkana",
        "mardan",
        "swat",
        "kohat",
        "gilgit",
        "skardu",
        "hunza",
        "abadan",
        "chitral",
        "mansehra",
        "abbottabad",
        "naran",
        "kaghan",
        "thar",
        "sindh",
        "punjab",
        "balochistan",
        "khyber pakhtunkhwa",
        "azad kashmir",
        "gilgit baltistan",
        "pakistan",
        "india",
        "afghanistan",
        "iran",
        "china",
        "saudi arabia",
        "turkey",
        "dubai",
        "doha",
        "new york",
        "london",
        "paris",
        "tokyo",
        "beijing",
        "moscow",
        "washington",
        "kabul",
        "tehran",
        "riyadh",
        "ankara",
        "berlin",
    ]

    organisations = [
        "pti",
        "pmln",
        "ppp",
        "mqm",
        "jui",
        "pdm",
        "ecp",
        "supreme court",
        "high court",
        "fbr",
        "sbp",
        "nadra",
        "pemra",
        "wapda",
        "ogdcl",
        "pia",
        "pcb",
        "icc",
        "fifa",
        "un",
        "who",
        "imf",
        "world bank",
        "asian development bank",
        "nab",
        "fia",
        "isi",
        "army",
        "navy",
        "air force",
        "geo news",
        "ary news",
        "dawn",
        "jang",
        "express news",
    ]

    misc = [
        "budget 2026",
        "psl",
        "icc world cup",
        "covid 19",
        "flood relief",
        "climate summit",
    ]

    return {"PER": persons, "LOC": locations, "ORG": organisations, "MISC": misc}


def build_phrase_map(gazetteer: dict[str, list[str]]) -> dict[tuple[str, ...], str]:
    phrase_map: dict[tuple[str, ...], str] = {}
    for ent_type, phrases in gazetteer.items():
        for phrase in phrases:
            toks = tuple(tokenize_text(phrase.lower(), lowercase=True))
            if toks:
                phrase_map[toks] = ent_type
    return phrase_map


def tag_ner_bio(tokens: list[str], phrase_map: dict[tuple[str, ...], str]) -> list[str]:
    lower_tokens = [t.lower() for t in tokens]
    tags = ["O"] * len(tokens)

    max_len = max((len(k) for k in phrase_map.keys()), default=1)
    i = 0
    while i < len(tokens):
        matched = False
        for length in range(min(max_len, len(tokens) - i), 0, -1):
            span = tuple(lower_tokens[i : i + length])
            ent_type = phrase_map.get(span)
            if ent_type is None:
                continue

            tags[i] = f"B-{ent_type}"
            for j in range(i + 1, i + length):
                tags[j] = f"I-{ent_type}"
            i += length
            matched = True
            break

        if not matched:
            i += 1

    return tags


def stratified_split(
    items: list[AnnotatedSentence],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[AnnotatedSentence], list[AnnotatedSentence], list[AnnotatedSentence]]:
    if not (0.0 < train_ratio < 1.0 and 0.0 < val_ratio < 1.0 and train_ratio + val_ratio < 1.0):
        raise ValueError("Invalid split ratios")

    rng = random.Random(seed)
    by_topic: dict[str, list[AnnotatedSentence]] = defaultdict(list)
    for item in items:
        by_topic[item.topic].append(item)

    train: list[AnnotatedSentence] = []
    val: list[AnnotatedSentence] = []
    test: list[AnnotatedSentence] = []

    for topic_items in by_topic.values():
        topic_copy = topic_items[:]
        rng.shuffle(topic_copy)

        n = len(topic_copy)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        n_test = max(0, n - n_train - n_val)

        # Keep at least one sample in each available split where possible.
        if n >= 3:
            n_train = max(1, n_train)
            n_val = max(1, n_val)
            n_test = max(1, n - n_train - n_val)
            while n_train + n_val + n_test > n:
                n_train = max(1, n_train - 1)

        train.extend(topic_copy[:n_train])
        val.extend(topic_copy[n_train : n_train + n_val])
        test.extend(topic_copy[n_train + n_val : n_train + n_val + n_test])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def write_conll(
    sentences: list[AnnotatedSentence],
    output_path: str | Path,
    tag_type: str,
) -> None:
    if tag_type not in {"pos", "ner"}:
        raise ValueError("tag_type must be 'pos' or 'ner'")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for sent in sentences:
            tags = sent.pos_tags if tag_type == "pos" else sent.ner_tags
            for tok, tag in zip(sent.tokens, tags):
                f.write(f"{tok}\t{tag}\n")
            f.write("\n")


def label_distribution(sentences: list[AnnotatedSentence]) -> dict[str, dict[str, int]]:
    pos_counter: Counter[str] = Counter()
    ner_counter: Counter[str] = Counter()

    for sent in sentences:
        pos_counter.update(sent.pos_tags)
        ner_counter.update(sent.ner_tags)

    return {
        "pos": dict(sorted(pos_counter.items())),
        "ner": dict(sorted(ner_counter.items())),
    }
