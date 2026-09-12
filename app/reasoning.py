from collections import Counter


# ============================================================
# CONFIGURATION
# ============================================================

CONFIDENCE_THRESHOLD = 0.50


# ============================================================
# INTENT ROUTING
# ============================================================

def needs_detector(question):

    question = question.lower().strip()

    image_keywords = [
        "image",
        "picture",
        "photo",
        "fish",
        "fish species",
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
        "visible"
    ]

    for keyword in image_keywords:

        if keyword in question:
            return True

    return False


# ============================================================
# CONFIDENCE GUARDRAIL
# ============================================================

def get_confident_detections(detections):

    if not detections:
        return []

    confident = [
        detection
        for detection in detections
        if detection["confidence"] >= CONFIDENCE_THRESHOLD
    ]

    return confident


def has_sufficient_information(detections):

    confident = get_confident_detections(detections)

    return len(confident) > 0


# ============================================================
# EXTRACT POSSIBLE SPECIES FROM QUESTION
# ============================================================

def find_requested_class(question, detections):

    question = question.lower()

    available_classes = {
        detection["class"].lower(): detection["class"]
        for detection in detections
    }

    for class_lower, original_class in available_classes.items():

        if class_lower in question:
            return original_class

    return None


# ============================================================
# MAIN REASONING FUNCTION
# ============================================================

def answer_question(question, detections):

    question = question.lower().strip()

    # --------------------------------------------------------
    # CONFIDENCE GUARDRAIL
    # --------------------------------------------------------

    confident_detections = get_confident_detections(detections)

    if not confident_detections:

        return (
            "Insufficient information: "
            "no sufficiently confident detections from the trained "
            "classes were found in the image."
        )

    # --------------------------------------------------------
    # COUNT FISH
    # --------------------------------------------------------

    if "how many" in question and "fish" in question:

        fish_detections = [
            d for d in confident_detections
            if "fish" in d["class"].lower()
        ]

        if not fish_detections:

            return (
                "Insufficient information: "
                "no sufficiently confident fish detections were found."
            )

        count = len(fish_detections)

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
            f"confidently detected objects in the image."
        )

    # --------------------------------------------------------
    # MOST COMMON OBJECT
    # --------------------------------------------------------

    if "most common" in question:

        counts = Counter(
            d["class"]
            for d in confident_detections
        )

        most_common_class, count = counts.most_common(1)[0]

        return (
            f"The most common detected object is "
            f"{most_common_class} with {count} detection(s)."
        )

    # --------------------------------------------------------
    # TYPE OF FISH
    # --------------------------------------------------------

    if (
        "what type of fish" in question
        or "which fish" in question
        or "what kind of fish" in question
        or "fish species" in question
    ):

        fish_classes = [
            d["class"]
            for d in confident_detections
            if "fish" in d["class"].lower()
        ]

        if not fish_classes:

            return (
                "Insufficient information: "
                "no sufficiently confident fish species "
                "were detected."
            )

        unique_classes = list(dict.fromkeys(fish_classes))

        return (
            "The confidently detected fish species are: "
            + ", ".join(unique_classes)
            + "."
        )

    # --------------------------------------------------------
    # SPECIFIC CLASS / SPECIES
    # --------------------------------------------------------

    requested_class = find_requested_class(
        question,
        confident_detections
    )

    if requested_class:

        matching = [
            d for d in confident_detections
            if d["class"].lower() == requested_class.lower()
        ]

        if matching:

            return (
                f"Yes, {len(matching)} "
                f"{requested_class} detection(s) "
                f"were found with sufficient confidence."
            )

    # --------------------------------------------------------
    # ZEBRAFISH EXAMPLE
    # --------------------------------------------------------

    if "zebrafish" in question:

        return (
            "Insufficient information: "
            "ZebraFish is not sufficiently represented in the "
            "confident detections for this image."
        )

    # --------------------------------------------------------
    # PERSON DETECTION
    # --------------------------------------------------------

    if (
        "person" in question
        or "people" in question
        or "human" in question
    ):

        people = [
            d for d in confident_detections
            if d["class"].lower()
            in ["person", "people", "human"]
        ]

        if people:

            return (
                f"Yes, {len(people)} person(s) "
                f"were detected."
            )

        return (
            "Insufficient information: "
            "the trained detector does not contain a sufficiently "
            "confident person detection."
        )

    # --------------------------------------------------------
    # GENERAL WHAT / WHICH QUESTION
    # --------------------------------------------------------

    if "what" in question or "which" in question:

        classes = list(
            dict.fromkeys(
                d["class"]
                for d in confident_detections
            )
        )

        return (
            "The confidently detected objects are: "
            + ", ".join(classes)
            + "."
        )

    # --------------------------------------------------------
    # FALLBACK
    # --------------------------------------------------------

    return (
        "I can answer questions based on the "
        "confidently detected objects in the image."
    )