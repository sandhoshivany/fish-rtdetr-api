from collections import Counter

from app.detector import CONFIDENCE_THRESHOLD


# ============================================================
# INTENT ROUTING
# ============================================================

IMAGE_KEYWORDS = (
    "image",
    "picture",
    "photo",
    "fish",
    "species",
    "object",
    "person",
    "people",
    "animal",
    "detect",
    "detected",
    "count",
    "how many",
    "what objects",
    "which objects",
    "what type",
    "which type",
    "what kind",
    "most common",
    "present",
    "visible",
)


def needs_detector(question: str) -> bool:
    """
    Determine whether the user's question requires
    information from the uploaded image.
    """

    question = question.lower().strip()

    return any(
        keyword in question
        for keyword in IMAGE_KEYWORDS
    )


# ============================================================
# CONFIDENCE GUARDRAIL
# ============================================================

def get_confident_detections(detections: list) -> list:
    """
    Keep only detections that meet the confidence threshold.
    """

    return [
        detection
        for detection in detections
        if detection.get("confidence", 0) >= CONFIDENCE_THRESHOLD
    ]


# ============================================================
# EXTRACT REQUESTED SPECIES
# ============================================================

def find_requested_class(
    question: str,
    detections: list
) -> str | None:
    """
    Check whether the user mentioned one of the
    detected fish species.
    """

    question = question.lower().strip()

    available_classes = {
        detection["class"].lower(): detection["class"]
        for detection in detections
    }

    for class_name, original_name in available_classes.items():
        if class_name in question:
            return original_name

    return None


# ============================================================
# MAIN REASONING FUNCTION
# ============================================================

def answer_question(
    question: str,
    detections: list
) -> str:
    """
    Generate an answer using only confidently
    detected fish information.
    """

    question = question.lower().strip()

    # --------------------------------------------------------
    # CONFIDENCE GUARDRAIL
    # --------------------------------------------------------

    confident_detections = get_confident_detections(detections)

    if not confident_detections:
        return (
            "Insufficient information: no sufficiently "
            "confident fish detections were found."
        )

    # --------------------------------------------------------
    # COUNT FISH
    # --------------------------------------------------------

    if "how many" in question and "fish" in question:

        count = len(confident_detections)

        if count == 1:
            return "There is 1 confidently detected fish in the image."

        return (
            f"There are {count} confidently detected fish "
            f"in the image."
        )

    # --------------------------------------------------------
    # COUNT OBJECTS
    # --------------------------------------------------------

    if "how many objects" in question:

        return (
            f"There are {len(confident_detections)} "
            f"confidently detected fish objects in the image."
        )

    # --------------------------------------------------------
    # MOST COMMON SPECIES
    # --------------------------------------------------------

    if "most common" in question:

        counts = Counter(
            detection["class"]
            for detection in confident_detections
        )

        species, count = counts.most_common(1)[0]

        return (
            f"The most common detected fish species is "
            f"{species} with {count} detection(s)."
        )

    # --------------------------------------------------------
    # PERSON / HUMAN QUESTION
    # --------------------------------------------------------

    if any(
        keyword in question
        for keyword in ("person", "people", "human")
    ):
        return (
            "Insufficient information: this fish-species "
            "detector is not trained to detect people."
        )

    # --------------------------------------------------------
    # SPECIES / TYPE QUESTION
    # --------------------------------------------------------

    if any(
        phrase in question
        for phrase in (
            "what type of fish",
            "which fish",
            "what kind of fish",
            "fish species",
            "what species",
            "which species",
        )
    ):

        species = list(
            dict.fromkeys(
                detection["class"]
                for detection in confident_detections
            )
        )

        return (
            "The confidently detected fish species are: "
            + ", ".join(species)
            + "."
        )

    # --------------------------------------------------------
    # SPECIFIC SPECIES QUESTION
    # --------------------------------------------------------

    requested_class = find_requested_class(
        question,
        confident_detections
    )

    if requested_class:

        count = sum(
            detection["class"].lower()
            == requested_class.lower()
            for detection in confident_detections
        )

        return (
            f"Yes, {count} {requested_class} detection(s) "
            f"were found with sufficient confidence."
        )

    # --------------------------------------------------------
    # GENERAL WHAT / WHICH QUESTION
    # --------------------------------------------------------

    if "what" in question or "which" in question:

        species = list(
            dict.fromkeys(
                detection["class"]
                for detection in confident_detections
            )
        )

        return (
            "The confidently detected fish species are: "
            + ", ".join(species)
            + "."
        )

    # --------------------------------------------------------
    # FALLBACK
    # --------------------------------------------------------

    return (
        "I can answer questions based on the confidently "
        "detected fish species in the image."
    )