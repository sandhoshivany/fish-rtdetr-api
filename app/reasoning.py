# ============================================================
# REASONING MODULE
# ============================================================

# Keywords that indicate the question needs image detection
DETECTION_KEYWORDS = (
    "fish",
    "species",
    "identify",
    "detected",
    "detection",
    "present",
    "found",
    "how many",
    "count",
)


# ============================================================
# INTENT ROUTING
# ============================================================

def needs_detector(question: str) -> bool:
    """
    Check whether the user's question requires
    information from the uploaded image.
    """

    question = question.lower().strip()

    return any(
        keyword in question
        for keyword in DETECTION_KEYWORDS
    )


# ============================================================
# ANSWER GENERATION
# ============================================================

def answer_question(question: str, detections: list) -> str:
    """
    Generate an answer using only the confident
    RT-DETR detections.
    """

    question = question.lower().strip()

    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    if not detections:
        return (
            "Insufficient information: no sufficiently "
            "confident fish detections were found in the image."
        )

    # --------------------------------------------------------
    # Extract detected species
    # --------------------------------------------------------

    species = [
        detection["class"]
        for detection in detections
    ]

    unique_species = list(dict.fromkeys(species))

    # --------------------------------------------------------
    # Count question
    # --------------------------------------------------------

    if "how many" in question or "count" in question:
        return (
            f"There are {len(detections)} confidently "
            f"detected fish in the image."
        )

    # --------------------------------------------------------
    # Identification / species question
    # --------------------------------------------------------

    if any(
        keyword in question
        for keyword in (
            "identify",
            "species",
            "which fish",
            "what fish",
            "what type",
            "type of fish",
        )
    ):
        if len(unique_species) == 1:
            return (
                f"The detected fish species is "
                f"{unique_species[0]}."
            )

        return (
            "The detected fish species are: "
            f"{', '.join(unique_species)}."
        )

    # --------------------------------------------------------
    # Presence / detection question
    # --------------------------------------------------------

    if any(
        keyword in question
        for keyword in (
            "present",
            "found",
            "detected",
            "in the image",
        )
    ):
        return (
            "The following fish species were confidently "
            "detected: "
            f"{', '.join(unique_species)}."
        )

    # --------------------------------------------------------
    # Default response
    # --------------------------------------------------------

    return (
        "Based on the confident RT-DETR detections, "
        "the image contains: "
        f"{', '.join(unique_species)}."
    )