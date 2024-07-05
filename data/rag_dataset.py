# [(index, llm_prompt, (required, references, ...))]
RAG_QUESTIONS_V10 = [
    (
        0,
        "What is the current release version of FURY and what are the new features?",
        ("https://github.com/fury-gl/fury/blob/v0.10.0/docs/source/release_notes/releasev0.10.0.rst",)
    ),
    (
        1,
        "render 3 spheres with different radii, different colors, different centers, set opacity to 0.5. There should only be 1 sphere actor that does this. Center coordinates are (0, 0, 0), (100, 0, 0), (200, 0, 0). Keep radius 10.",
        ("https://github.com/fury-gl/fury/blob/v0.10.0/fury/actor.py#L1773-L1861",)
    ),
    (
        2,
        "How to convert np color array to vtk color array",
        ("https://github.com/fury-gl/fury/blob/v0.10.0/fury/utils.py#L90-L116",)
    ),
    (
        3,
        "How to install fury",
        ("https://github.com/fury-gl/fury/blob/v0.10.0/docs/source/installation.rst",)
    ),
    (
        4,
        "render octagonalprism and spheres",
        ("https://github.com/fury-gl/fury/blob/v0.10.0/fury/actor.py#L1773-L1861",
         "https://github.com/fury-gl/fury/blob/v0.10.0/fury/actor.py#L2535-L2581",)
    )
]