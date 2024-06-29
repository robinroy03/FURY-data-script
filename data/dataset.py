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
        0, "render 3 spheres with different radii, different colors, different centers, set opacity to 0.5. There should only be 1 sphere actor that does this. Center coordinates are (0, 0, 0), (100, 0, 0), (200, 0, 0). Keep radius 10.",
        "The image must contain 3 spheres of different colours",
        """
        import numpy.testing as npt
        from fury import window, actor
        import numpy as np

        interactive = True

        xyzr = np.array([[0, 0, 0, 10], [100, 0, 0, 25], [200, 0, 0, 50]])
        colors = np.array([[1, 0, 0, 0.3], [0, 1, 0, 0.4], [0, 0, 1.0, 0.99]])
        opacity = 0.5

        scene = window.Scene()
        sphere_actor = actor.sphere(
            centers=xyzr[:, :3],
            colors=colors[:],
            radii=10,
            opacity=opacity,
            use_primitive=False,
        )
        scene.add(sphere_actor)

        if interactive:
            window.show(scene, order_transparent=True)
        """
    ),
    (
        1, "Write 'Hello world' in FURY.",
        "The image must contain 'Hello world' written.",
        """
        import numpy.testing as npt
        from fury import window, actor
        import numpy as np

        interactive = True

        text_actor = actor.vector_text("Hello world", direction=None)

        scene = window.Scene()
        scene.add(text_actor)
        scene.reset_camera()
        scene.reset_clipping_range()

        assert text_actor.GetCamera() is scene.GetActiveCamera()

        if interactive:
            window.show(scene, reset_camera=False)
        """
    ),
    (
        2, "Render 2 cones in FURY.",
        "The image must contain 2 cones, it may look like triangles/pyramid due to 3D.",
        """
        import numpy.testing as npt
        from fury import window, actor
        import numpy as np

        interactive = True

        scene = window.Scene()
        centers = np.array([[40, 0, 0], [60, 0, 0]])
        directions = np.array([[1, 1, 1], [1, 0, 0], [0, 0, 1], [1, 1, 1]])
        colors = np.array([[1, 0, 0, 0.3], [0, 1, 0, 0.4], [0, 0, 1, 0.99], [1, 1, 1, 0.6]])
        vertices = np.array(
            [[0.0, 0.0, 0.0], [0.0, 10.0, 0.0], [10.0, 0.0, 0.0], [0.0, 0.0, 10.0]]
        )
        faces = np.array([[0, 1, 3], [0, 2, 1]])
        cone_1 = actor.cone(
            centers=centers[:2],
            directions=directions[:2],
            colors=colors[:2],
            vertices=vertices,
            faces=faces,
            use_primitive=False,
        )

        cone_2 = actor.cone(
            centers=centers[2:],
            directions=directions[2:],
            colors=colors[2:],
            heights=10,
            use_primitive=False,
        )
        scene.add(cone_1)
        scene.add(cone_2)

        if interactive:
            window.show(scene, order_transparent=True)
        """
    ),
    (
        3, "Render a plane.",
        "The image must contain a plane.",
        """
        import numpy.testing as npt
        from fury import window, actor
        import numpy as np

        interactive = True

        plane = actor.rectangle(centers=np.array([[0, 0, 0]]))
        scene = window.Scene()
        scene.add(plane)

        window.show(scene)
        """
    ),
    (
        4, "Render 6 ellipsoid actors of different colours.",
        "The image must contain 3 ellipsoid actors. (or something similar to that)",
        """
        import numpy.testing as npt
        from fury import window, actor
        import numpy as np

        interactive = True


        centers = [-1, 1, 0]
        axes = [[[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[1, 2, -2], [2, 1, 2], [2, -2, -1]]]
        lengths = [[1, 1, 1]]
        npt.assert_raises(ValueError, actor.ellipsoid, centers, axes, lengths)

        # number of lengths does not match with number of centers
        lengths = [[1, 1, 1], [1, 1, 0.5]]
        npt.assert_raises(ValueError, actor.ellipsoid, centers, axes, lengths)

        scene = window.Scene()
        scene.background((0, 0, 0))

        axes = np.array(
            [
                [[-0.6, 0.5, -0.6], [-0.8, -0.4, 0.5], [-0.1, -0.7, -0.7]],
                [[0.1, 0.6, -0.8], [0.6, 0.5, 0.5], [-0.8, 0.6, 0.3]],
                [[0.7, 0.5, -0.5], [0, -0.7, -0.7], [-0.7, 0.6, -0.5]],
                [[0.7, -0.3, -0.6], [0.2, -0.8, 0.6], [0.7, 0.6, 0.5]],
                [[1, 2, -2], [2, 1, 2], [2, -2, -1]],
                [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            ]
        )
        lengths = np.array(
            [
                [1, 1, 1],
                [1, 1, 0.5],
                [1, 0.5, 0.5],
                [1, 0.5, 0.25],
                [1, 1, 0.3],
                [1, 0.3, 0.3],
            ]
        )
        centers = np.array(
            [[-1, 1, 0], [0, 1, 0], [1, 1, 0], [-1, 0, 0], [0, 0, 0], [1, 0, 0]]
        )
        colors = np.array(
            [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]]
        )

        ellipsoids = actor.ellipsoid(
            axes=axes, lengths=lengths, centers=centers, scales=1.0, colors=colors
        )
        scene.add(ellipsoids)

        if interactive:
            window.show(scene)
        """
    ),
]