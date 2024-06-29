LLM_BENCHMARK_PROMPT = """
You are given an image description. You have to check if the image description satisfies the requirement mentioned. Give reasoning before giving your final verdict (YES/NO). Here are some examples:

Criteria: The image must contain 3 spheres of different colours. 
Description: The image presents two circles, one yellow and one purple. The yellow circle is positioned on the left side of the frame, while the purple circle is located on the right side. Both circles are perfectly centered within the frame, creating a sense of balance in the composition.
Reasoning: No, this image do not contain 3 spheres of different colours. It only contains 2 spheres of different colours.
Verdict: NO

Criteria: The image must contain 3 planes of different colours.
Description: The image presents 3 different planes of colours yellow, blue, green. They are all placed adjacent to each other.
Reasoning: Yes, this image satisfies the criteria of 3 planes of different colours.
Verdict: YES

Criteria: {}
Description: {}
Reasoning:
Verdict:
"""


MOONDREAM_PROMPT = "Describe this image"


# [(LLM_PROMPT, CRITERIA), (), (), .....]
BENCHMARK_QUESTIONS = [
    (
        0, "render 3 spheres with different radii, different colors, different centers, set opacity to 0.5. There should only be 1 sphere actor that does this. Center coordinates are (100, 0, 0), (200, 0, 0), (300, 0, 0). Keep radius 10.",
        "The image must contain 3 spheres of different colours"
    ),
    (
        1, "Write 'Hello world' in FURY.",
        "The image must contain 'Hello world' written."
    ),
    (
        2, "Render 2 cones in FURY.",
        "The image must contain 2 cones, it may look like triangles/pyramid due to 3D."
    )

]