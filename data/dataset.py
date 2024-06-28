BENCHMARK_QUESTIONS = [
    (0, "render 3 spheres with different radii, different colors, different centers, set opacity to 0.5. There should only be 1 sphere actor that does this. The name of that variable should be `sphere_actor`. Code should be inside one python block. Comment `window.show()`. Center coordinates are (100, 0, 0), (200, 0, 0), (300, 0, 0). Keep radius 10.",
     """from code import sphere_actor, scene
from fury import window
import numpy.testing as npt
import numpy as np

arr = window.snapshot(scene)
report = window.analyze_snapshot(arr)
opacity = 0.5
npt.assert_equal(report.objects, 3)
npt.assert_equal(sphere_actor.GetProperty().GetOpacity(), opacity)""")
]