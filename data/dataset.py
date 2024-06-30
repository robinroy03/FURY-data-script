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


# [(INDEX, LLM_PROMPT, CRITERIA, WORKING_CODE), (), (), .....]
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
        "The image must contain 6 ellipsoid actors. (or something similar to that)",
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
    (
        5, "Render streamtubes.",
        "The image must contain streamtubes, it may look like numerous rods or something.",
        """
        import numpy as np
        from fury import actor, window
        scene = window.Scene()
        lines = [np.random.rand(10, 3), np.random.rand(20, 3)]
        colors = np.random.rand(2, 3)
        c = actor.streamtube(lines, colors)
        scene.add(c)
        window.show(scene)
        """
    ),
    (
        6, "Render arrows",
        "The image must contain arrows",
        """
        from fury import window, actor
        import numpy as np

        scene = window.Scene()
        centers = np.random.rand(5, 3)
        directions = np.random.rand(5, 3)
        heights = np.random.rand(5)
        arrow_actor = actor.arrow(centers, directions, (1, 1, 1), heights)
        scene.add(arrow_actor)
        window.show(scene)
        """
    ),
    (
        7, "Render prisms",
        "The image must contain prisms, can also look like triangles due to 3d orientation"
        """
        from fury import window, actor
        import numpy as np

        scene = window.Scene()
        centers = np.random.rand(3, 3)
        dirs = np.random.rand(3, 3)
        colors = np.random.rand(3, 3)
        scales = np.random.rand(3, 1)
        actor = actor.triangularprism(centers, dirs, colors, scales)
        scene.add(actor)
        window.show(scene)
        """
    ),
    (
        8, "Render a rhombicuboctahedron",
        "The image must contain rhombicuboctahedron, it may look different due to 3d orientation. Confirmation of it as some kind of polyhedron is enough.",
        """
        from fury import window, actor
        import numpy as np

        scene = window.Scene()
        centers = np.random.rand(3, 3)
        dirs = np.random.rand(3, 3)
        colors = np.random.rand(3, 3)
        scales = np.random.rand(3, 1)
        actor = actor.rhombicuboctahedron(centers, dirs, colors, scales)
        scene.add(actor)
        window.show(scene)
        """
    ),
    (
        9, "Render markers in fury",
        "The image must contain different markers",
        """
        from fury import actor, window
        import numpy as np

        interactive = True

        scene = window.Scene()
        scene.background((1, 1, 1))
        centers_3do = np.array([[4, 0, 0], [4, 4, 0], [4, 8, 0]])
        markers_2d = ["o", "s", "d", "^", "p", "h", "s6", "x", "+"]
        center_markers_2d = np.array([[0, i * 2, 0] for i in range(len(markers_2d))])
        markers_2d = actor.markers(
            center_markers_2d, colors=(0, 1, 0), scales=1, marker=markers_2d
        )
        scene.add(markers_2d)

        if interactive:
            window.show(scene)
        """
    ),
    (
        10, "Render a duck in FURY, use GLTF",
        "The image must contain a duck",
        """
        from fury import actor, window, utils
        from fury.data import fetch_gltf, read_viz_gltf
        from fury.gltf import glTF
        import numpy.testing as npt


        fetch_gltf("Duck")
        filename = read_viz_gltf("Duck", "glTF")
        importer = glTF(filename)
        polydatas = importer.polydatas

        scene = window.Scene()
        scene.add(utils.get_actor_from_polydata(polydatas[0]))
        """
    ),
    (
        11, "Render a bounding box",
        "The image must contain a box (a bounding box of sorts) in 3D",
        """
        import numpy as np
        from fury import actor, window, molecular as mol
        from fury.data import fetch_viz_icons

        interactive = True

        scene = window.Scene()
        molecule = mol.Molecule()
        mol.add_atom(molecule, 6, 0, 0, 0)
        mol.add_atom(molecule, 6, 1, 1, 1)
        mol.add_bond(molecule, 0, 1, 1)

        molecule_actor = mol.stick(molecule)
        test_box = mol.bounding_box(molecule, colors=(0, 1, 1), linewidth=0.1)
        scene.add(test_box)

        if interactive:
            window.show(scene)
        """
    ),
    (
        12, "render spiky spheres",
        "The image must contain a sphere with arrows pointing outwards"
        """
        import itertools
        import numpy as np
        from fury import actor, primitive, utils, window

        scene = window.Scene()
        vertices, triangles = primitive.prim_sphere(name='symmetric362', gen_faces=False)
        point_actor = actor.point(vertices, point_radius=0.01, colors=(0, 1, 0))
        normals = utils.normals_from_v_f(vertices, triangles)
        arrow_actor = actor.arrow(
            centers=vertices,
            directions=normals,
            colors=(1, 0, 0),
            heights=0.2,
            resolution=10,
            vertices=None,
            faces=None,
        )
        primitive_colors = np.zeros(vertices.shape)
        primitive_colors[:, 2] = 180
        primitive_actor = utils.get_actor_from_primitive(
            vertices=vertices,
            triangles=triangles,
            colors=primitive_colors,
            normals=normals,
            backface_culling=True,
        )
        scene.add(point_actor)
        scene.add(arrow_actor)
        scene.add(primitive_actor)
        scene.add(actor.axes())
        showm = window.ShowManager(
            scene, size=(900, 768), reset_camera=False, order_transparent=True
        )
        counter = itertools.count()
        def timer_callback(_obj, _event):
            cnt = next(counter)
            showm.scene.azimuth(0.05 * cnt)
            primitive_actor.GetProperty().SetOpacity(cnt / 10.0)
            showm.render()
            if cnt == 20:
                showm.exit()

        showm.add_timer_callback(True, 200, timer_callback)
        showm.start()
        window.record(showm.scene, size=(900, 768), out_path='viz_spiky.png')
        """
    ),
    (
        13, "Skeletal Animation in glTF file",
        "There will be a skeleton actor in the image",
        """
        from fury import window
        from fury.data import fetch_gltf, read_viz_gltf
        from fury.gltf import glTF

        fetch_gltf('RiggedFigure', 'glTF')
        filename = read_viz_gltf('RiggedFigure')

        gltf_obj = glTF(filename, apply_normals=False)

        animation = gltf_obj.skin_animation()['anim_0']

        gltf_obj.initialize_skin(animation, bones=False)

        scene = window.Scene()
        showm = window.ShowManager(scene, size=(900, 768), reset_camera=True, order_transparent=True)
        showm.initialize()
        scene.add(animation)

        def timer_callback(_obj, _event):
            gltf_obj.update_skin(animation)
            showm.render()

        showm.add_timer_callback(True, 20, timer_callback)
        scene.reset_camera()

        interactive = True

        if interactive:
            showm.start()

        window.record(scene, out_path='viz_skinning.png', size=(900, 768))
        """
    ),
    (
        14, "Render earth and moon and then animate them",
        "The image must contain earth and moon",
        """
        import itertools

        import numpy as np

        from fury import actor, io, utils, window
        from fury.data import (
            fetch_viz_models,
            fetch_viz_textures,
            read_viz_models,
            read_viz_textures,
        )

        import matplotlib.pyplot as plt

        fetch_viz_textures()
        earth_filename = read_viz_textures('1_earth_8k.jpg')
        earth_image = io.load_image(earth_filename)

        earth_actor = actor.texture_on_sphere(earth_image)

        moon_filename = read_viz_textures('moon-8k.jpg')
        moon_image = io.load_image(moon_filename)

        moon_actor = actor.texture_on_sphere(moon_image)

        scene = window.Scene()
        scene.add(earth_actor)
        scene.add(moon_actor)

        moon_actor.SetPosition(1, 0.1, 0.5)
        moon_actor.SetScale(0.25, 0.25, 0.25)
        utils.rotate(earth_actor, (-90, 1, 0, 0))

        showm = window.ShowManager(
            scene, size=(900, 768), reset_camera=False, order_transparent=True
        )

        counter = itertools.count()

        def timer_callback(_obj, _event):
            cnt = next(counter)
            showm.render()
            if cnt < 450:
                utils.rotate(earth_actor, (1, 0, 1, 0))
            if cnt % 5 == 0 and cnt < 450:
                showm.scene.azimuth(-1)
            if cnt == 300:
                scene.set_camera(
                    position=(-3.679, 0.00, 2.314),
                    focal_point=(0.0, 0.35, 0.00),
                    view_up=(0.00, 1.00, 0.00),
                )
            if cnt > 300 and cnt < 450:
                scene.zoom(1.01)
            if cnt >= 450 and cnt < 1500:
                scene.add(sphere_actor)
                scene.add(text_actor)
            if cnt >= 450 and cnt < 550:
                scene.zoom(1.01)
            if cnt == 575:
                moon_actor.SetPosition(-1, 0.1, 0.5)
                scene.set_camera(
                    position=(-0.5, 0.1, 0.00),
                    focal_point=(-1, 0.1, 0.5),
                    view_up=(0.00, 1.00, 0.00),
                )
                scene.zoom(0.03)
                scene.add(satellite_actor)
                utils.rotate(satellite_actor, (180, 0, 1, 0))
                scene.rm(earth_actor)
            if cnt > 575 and cnt < 750:
                showm.scene.azimuth(-2)
                utils.rotate(moon_actor, (-2, 0, 1, 0))
                satellite_actor.SetPosition(-0.8, 0.1 - cnt / 10000, 0.4)
            if cnt >= 750 and cnt < 1100:
                showm.scene.azimuth(-2)
                utils.rotate(moon_actor, (-2, 0, 1, 0))
                satellite_actor.SetPosition(-0.8, -0.07 + cnt / 10000, 0.4)
            if cnt == 1100:
                showm.exit()

        fetch_viz_models()
        satellite_filename = read_viz_models('satellite_obj.obj')
        satellite = io.load_polydata(satellite_filename)
        satellite_actor = utils.get_actor_from_polydata(satellite)

        satellite_actor.SetPosition(-0.75, 0.1, 0.4)
        satellite_actor.SetScale(0.005, 0.005, 0.005)

        showm.add_timer_callback(True, 35, timer_callback)
        showm.start()
        window.record(showm.scene, size=(900, 768), out_path='viz_earth_animation.png')
        """
    ),
    (
        15, "Render solar system",
        "The image will contain solar system",
        """
\"\"\"
=======================
Solar System Animation
=======================

In this tutorial, we will create an animation of the solar system
using textured spheres. We will also show how to manipulate the
position of these sphere actors in a timer_callback function
to simulate orbital motion.
\"\"\"

import itertools

import numpy as np

from fury import actor, io, ui, utils, window
from fury.data import fetch_viz_textures, read_viz_icons, read_viz_textures

##############################################################################
# Create a scene to start.

scene = window.Scene()

# Create a panel and the start/pause buttons

panel = ui.Panel2D(size=(300, 100), color=(1, 1, 1), align="right")
panel.center = (400, 50)

pause_button = ui.Button2D(icon_fnames=[("square", read_viz_icons(fname="pause2.png"))])
start_button = ui.Button2D(icon_fnames=[("square", read_viz_icons(fname="play3.png"))])

# Add the buttons on the panel

panel.add_element(pause_button, (0.25, 0.33))
panel.add_element(start_button, (0.66, 0.33))


##############################################################################
# Define information relevant for each planet actor including its
# texture name, relative position, and scale.

planets_data = [
    {
        "filename": "8k_mercury.jpg",
        "position": 7,
        "earth_days": 58,
        "scale": (0.4, 0.4, 0.4),
    },
    {
        "filename": "8k_venus_surface.jpg",
        "position": 9,
        "earth_days": 243,
        "scale": (0.6, 0.6, 0.6),
    },
    {
        "filename": "1_earth_8k.jpg",
        "position": 11,
        "earth_days": 1,
        "scale": (0.4, 0.4, 0.4),
    },
    {
        "filename": "8k_mars.jpg",
        "position": 13,
        "earth_days": 1,
        "scale": (0.8, 0.8, 0.8),
    },
    {"filename": "jupiter.jpg", "position": 16, "earth_days": 0.41, "scale": (2, 2, 2)},
    {
        "filename": "8k_saturn.jpg",
        "position": 19,
        "earth_days": 0.45,
        "scale": (2, 2, 2),
    },
    {
        "filename": "8k_saturn_ring_alpha.png",
        "position": 19,
        "earth_days": 0.45,
        "scale": (3, 0.5, 3),
    },
    {
        "filename": "2k_uranus.jpg",
        "position": 22,
        "earth_days": 0.70,
        "scale": (1, 1, 1),
    },
    {
        "filename": "2k_neptune.jpg",
        "position": 25,
        "earth_days": 0.70,
        "scale": (1, 1, 1),
    },
    {"filename": "8k_sun.jpg", "position": 0, "earth_days": 27, "scale": (5, 5, 5)},
]
fetch_viz_textures()

##############################################################################
# To take advantage of the previously defined data structure we are going to
# create an auxiliary function that will load and apply the respective
# texture, set its respective properties (relative position and scale),
# and add the actor to a previously created scene.


def init_planet(planet_data):
    \"\"\"Initialize a planet actor.

    Parameters
    ----------
    planet_data : dict
        The planet_data is a dictionary, and the keys are filename(texture),
        position and scale.

    Returns
    -------
    planet_actor: actor
        The corresponding sphere actor with texture applied.
    \"\"\"
    planet_file = read_viz_textures(planet_data["filename"])
    planet_image = io.load_image(planet_file)
    planet_actor = actor.texture_on_sphere(planet_image)
    planet_actor.SetPosition(planet_data["position"], 0, 0)
    if planet_data["filename"] != "8k_saturn_ring_alpha.png":
        utils.rotate(planet_actor, (90, 1, 0, 0))
    planet_actor.SetScale(planet_data["scale"])
    scene.add(planet_actor)
    return planet_actor


##############################################################################
# Use the ``map`` function to create actors for each of the texture files
# in the ``planet_files`` list. Then, assign each actor to its corresponding
# actor in the list.

planet_actor_list = list(map(init_planet, planets_data))

mercury_actor = planet_actor_list[0]
venus_actor = planet_actor_list[1]
earth_actor = planet_actor_list[2]
mars_actor = planet_actor_list[3]
jupiter_actor = planet_actor_list[4]
saturn_actor = planet_actor_list[5]
saturn_rings_actor = planet_actor_list[6]
uranus_actor = planet_actor_list[7]
neptune_actor = planet_actor_list[8]
sun_actor = planet_actor_list[9]


##############################################################################
# Define the gravitational constant G, the orbital radii of each of the
# planets, and the central mass of the sun. The gravity and mass will be
# used to calculate the orbital position, so multiply these two together to
# create a new constant, which we will call miu.

g_exponent = np.float_power(10, -11)
g_constant = 6.673 * g_exponent

m_exponent = 1073741824  # np.power(10, 30)
m_constant = 1.989 * m_exponent

miu = m_constant * g_constant

##############################################################################
# Let's define two functions that will help us calculate the position of each
# planet as it orbits around the sun: ``get_orbit_period`` and
# ``get_orbital_position``, using the constant miu and the orbital radii
# of each planet.


def get_orbit_period(radius):
    return 2 * np.pi * np.sqrt(np.power(radius, 3) / miu)


def get_orbital_position(radius, time):
    orbit_period = get_orbit_period(radius)
    x = radius * np.cos((-2 * np.pi * time) / orbit_period)
    y = radius * np.sin((-2 * np.pi * time) / orbit_period)
    return x, y


##############################################################################
# Let's define a function to rotate the planet actor axially, we'll be defining
# axis of each planet and angle by which it should be rotated using
# ``rotate_axial`` funtction


def rotate_axial(actor, time, radius):
    axis = (0, radius, 0)
    angle = 50 / time
    utils.rotate(actor, (angle, axis[0], axis[1], axis[2]))
    return angle


##############################################################################
# Let's change the camera position to visualize the planets better.

scene.set_camera(position=(-20, 60, 100))

##############################################################################
# Next, create a ShowManager object. The ShowManager class is the interface
# between the scene, the window and the interactor.

showm = window.ShowManager(
    scene, size=(900, 768), reset_camera=False, order_transparent=True
)
scene.add(panel)

##############################################################################
# Next, let's focus on creating the animation.
# We can determine the duration of animation with using the ``counter``.
# Use itertools to avoid global variables.

counter = itertools.count()

##############################################################################
# Define one new function to use in ``timer_callback`` to update the planet
# positions ``update_planet_position``.


def update_planet_position(r_planet, planet_actor, cnt):
    pos_planet = get_orbital_position(r_planet, cnt)
    planet_actor.SetPosition(pos_planet[0], 0, pos_planet[1])
    return pos_planet


##############################################################################
# ``calculate_path`` function is for calculating the path/orbit
# of every planet.


def calculate_path(r_planet, c):
    planet_track = [
        [get_orbital_position(r_planet, i)[0], 0, get_orbital_position(r_planet, i)[1]]
        for i in range(c)
    ]
    return planet_track


##############################################################################
# First we are making a list that will contain radius from `planets_data`.
# Here we are not taking the radius of orbit/path for sun and saturn ring.
# `planet_actors` will contain all the planet actors.
# `r_times` will contain time taken (in days) by the planets to rotate
# around itself.

r_planets = [
    p_data["position"]
    for p_data in planets_data
    if "sun" not in p_data["filename"]
    if "saturn_ring" not in p_data["filename"]
]

planet_actors = [
    mercury_actor,
    venus_actor,
    earth_actor,
    mars_actor,
    jupiter_actor,
    saturn_actor,
    uranus_actor,
    neptune_actor,
]


sun_data = {
    "actor": sun_actor,
    "position": planets_data[9]["position"],
    "earth_days": planets_data[9]["earth_days"],
}

r_times = [p_data["earth_days"] for p_data in planets_data]

##############################################################################
# Here we are calculating and updating the path/orbit before animation starts.

planet_tracks = [calculate_path(rplanet, rplanet * 85) for rplanet in r_planets]

##############################################################################
# This is for orbit visualization. We are using line actor for orbits.
# After creating an actor we add it to the scene.

orbit_actor = actor.line(planet_tracks, colors=(1, 1, 1), linewidth=0.1)
scene.add(orbit_actor)

##############################################################################
# Define the ``timer_callback`` function, which controls what events happen
# at certain times, using the counter. Update the position of each planet
# actor using ``update_planet_position,`` assigning the x and y values of
# each planet's position with the newly calculated ones.


def timer_callback(_obj, _event):
    cnt = next(counter)
    showm.render()

    # Rotating the sun actor
    rotate_axial(sun_actor, sun_data["earth_days"], 1)

    for r_planet, p_actor, r_time in zip(r_planets, planet_actors, r_times):
        # if the planet is saturn then we also need to update the position
        # of its rings.
        if p_actor == saturn_actor:
            pos_saturn = update_planet_position(19, saturn_actor, cnt)
            saturn_rings_actor.SetPosition(pos_saturn[0], 0, pos_saturn[1])
        else:
            update_planet_position(r_planet, p_actor, cnt)
        rotate_axial(p_actor, r_time, r_planet)

    if cnt == 2000:
        showm.exit()


##############################################################################
# We add a callback to each button to perform some action.


def start_animation(i_ren, _obj, _button):
    showm.add_timer_callback(True, 10, timer_callback)


def pause_animation(i_ren, _obj, _button):
    showm.destroy_timers()


start_button.on_left_mouse_button_clicked = start_animation
pause_button.on_left_mouse_button_clicked = pause_animation


##############################################################################
# Watch the planets orbit the sun in your new animation!


showm.add_timer_callback(True, 10, timer_callback)
showm.start()

window.record(showm.scene, size=(900, 768), out_path="viz_solar_system_animation.png")"""
    ),
    (
        16, "Render the motion of a charged particle in an electric and magnetic field",
        "The image will show spirals, it'll be the motion of charged particles in electric and magnetic fileds",
        """
import itertools

import numpy as np

from fury import actor, ui, utils, window

radius_particle = 0.08
initial_velocity = 0.09
acc = 0.004
time = 0
incre_time = 0.09
angular_frq = 0.1
phase_angle = 0.002

scene = window.Scene()
scene.zoom(1.2)
scene.set_camera(position=(10, 12.5, 19), focal_point=(3.0, 0.0, 0.0), view_up=(0.0, 0.0, 0.0))
showm = window.ShowManager(scene, size=(800, 600), reset_camera=True, order_transparent=True)

color_arrow = window.colors.blue
centers = np.array([[0, 0, 0]])
directions = np.array([[1, 0, 0]])
heights = np.array([8])
arrow_actor = actor.arrow(centers, directions, color_arrow, heights, resolution=20, tip_length=0.06, tip_radius=0.012, shaft_radius=0.005)
scene.add(arrow_actor)

x = initial_velocity * time + 0.5 * acc * (time**2)
y = np.sin(angular_frq * time + phase_angle)
z = np.cos(angular_frq * time + phase_angle)

color_particle = window.colors.red
pts = np.array([[x, y, z]])
charge_actor = actor.point(pts, color_particle, point_radius=radius_particle)
scene.add(charge_actor)

vertices = utils.vertices_from_actor(charge_actor)
vcolors = utils.colors_from_actor(charge_actor, 'colors')
no_vertices_per_point = len(vertices)
initial_vertices = vertices.copy() - np.repeat(pts, no_vertices_per_point, axis=0)

tb = ui.TextBlock2D(bold=True, position=(100, 90))
m1 = 'Motion of a charged particle in a '
m2 = 'combined electric and magnetic field'
tb.message = m1 + m2
scene.add(tb)

counter = itertools.count()

end = 200

coor_1 = np.array([0, 0, 0])

def timer_callback(_obj, _event):
    global pts, time, incre_time, coor_1
    time += incre_time
    cnt = next(counter)

    x = initial_velocity * time + 0.5 * acc * (time**2)
    y = np.sin(10 * angular_frq * time + phase_angle)
    z = np.cos(10 * angular_frq * time + phase_angle)
    pts = np.array([[x, y, z]])

    vertices[:] = initial_vertices + np.repeat(pts, no_vertices_per_point, axis=0)

    utils.update_actor(charge_actor)

    # Plotting the path followed by the particle
    coor_2 = np.array([x, y, z])
    coors = np.array([coor_1, coor_2])
    coors = [coors]
    line_actor = actor.line(coors, window.colors.cyan, linewidth=3)
    scene.add(line_actor)
    coor_1 = coor_2

    showm.render()

    # to end the animation
    if cnt == end:
        showm.exit()

showm.add_timer_callback(True, 15, timer_callback)
showm.start()
window.record(showm.scene, size=(800, 600), out_path='viz_helical_motion.png')"""
    ),
    (
        17, "Render brainfibre odf simulation in fury",
        "the image will contain a brain",
        """
from dipy.data import get_sphere
from dipy.reconst.shm import sh_to_sf_matrix
import nibabel as nib

# First, we import some useful modules and methods.
import numpy as np

from fury import actor, ui, window
from fury.data import fetch_viz_dmri, fetch_viz_icons, read_viz_dmri
from fury.utils import fix_winding_order

fetch_viz_dmri()
fetch_viz_icons()

fodf_img = nib.load(read_viz_dmri("fodf.nii.gz"))
sh = fodf_img.get_fdata()
affine = fodf_img.affine
grid_shape = sh.shape[:-1]

sphere_low = get_sphere("repulsion100")
B_low = sh_to_sf_matrix(sphere_low, 8, return_inv=False)

scale = 0.5
norm = False
colormap = None
radial_scale = True
opacity = 1.0
global_cm = False

odf_actor_z = actor.odf_slicer(
    sh,
    affine=affine,
    sphere=sphere_low,
    scale=scale,
    norm=norm,
    radial_scale=radial_scale,
    opacity=opacity,
    colormap=colormap,
    global_cm=global_cm,
    B_matrix=B_low,
)

odf_actor_y = actor.odf_slicer(
    sh,
    affine=affine,
    sphere=sphere_low,
    scale=scale,
    norm=norm,
    radial_scale=radial_scale,
    opacity=opacity,
    colormap=colormap,
    global_cm=global_cm,
    B_matrix=B_low,
)
odf_actor_y.display_extent(
    0, grid_shape[0] - 1, grid_shape[1] // 2, grid_shape[1] // 2, 0, grid_shape[2] - 1
)

# ODF slicer for sagittal slice
odf_actor_x = actor.odf_slicer(
    sh,
    affine=affine,
    sphere=sphere_low,
    scale=scale,
    norm=norm,
    radial_scale=radial_scale,
    opacity=opacity,
    colormap=colormap,
    global_cm=global_cm,
    B_matrix=B_low,
)
odf_actor_x.display_extent(
    grid_shape[0] // 2, grid_shape[0] // 2, 0, grid_shape[1] - 1, 0, grid_shape[2] - 1
)

scene = window.Scene()
scene.add(odf_actor_z)
scene.add(odf_actor_y)
scene.add(odf_actor_x)

show_m = window.ShowManager(scene, reset_camera=True, size=(1200, 900))


line_slider_z = ui.LineSlider2D(
    min_value=0,
    max_value=grid_shape[2] - 1,
    initial_value=grid_shape[2] / 2,
    text_template="{value:.0f}",
    length=140,
)

line_slider_y = ui.LineSlider2D(
    min_value=0,
    max_value=grid_shape[1] - 1,
    initial_value=grid_shape[1] / 2,
    text_template="{value:.0f}",
    length=140,
)

line_slider_x = ui.LineSlider2D(
    min_value=0,
    max_value=grid_shape[0] - 1,
    initial_value=grid_shape[0] / 2,
    text_template="{value:.0f}",
    length=140,
)

sphere_high = get_sphere("symmetric362")

sphere_high.faces = fix_winding_order(sphere_high.vertices, sphere_high.faces, True)
B_high = sh_to_sf_matrix(sphere_high, 8, return_inv=False)
sphere_dict = {
    "Low resolution": (sphere_low, B_low),
    "High resolution": (sphere_high, B_high),
}
combobox = ui.ComboBox2D(items=list(sphere_dict))
scene.add(combobox)

def change_slice_z(slider):
    i = int(np.round(slider.value))
    odf_actor_z.slice_along_axis(i)


def change_slice_y(slider):
    i = int(np.round(slider.value))
    odf_actor_y.slice_along_axis(i, "yaxis")


def change_slice_x(slider):
    i = int(np.round(slider.value))
    odf_actor_x.slice_along_axis(i, "xaxis")


def change_sphere(combobox):
    sphere, B = sphere_dict[combobox.selected_text]
    odf_actor_x.update_sphere(sphere.vertices, sphere.faces, B)
    odf_actor_y.update_sphere(sphere.vertices, sphere.faces, B)
    odf_actor_z.update_sphere(sphere.vertices, sphere.faces, B)


line_slider_z.on_change = change_slice_z
line_slider_y.on_change = change_slice_y
line_slider_x.on_change = change_slice_x
combobox.on_change = change_sphere

def build_label(text):
    label = ui.TextBlock2D()
    label.message = text
    label.font_size = 18
    label.font_family = "Arial"
    label.justification = "left"
    label.bold = False
    label.italic = False
    label.shadow = False
    label.background_color = (0, 0, 0)
    label.color = (1, 1, 1)

    return label


line_slider_label_z = build_label(text="Z Slice")
line_slider_label_y = build_label(text="Y Slice")
line_slider_label_x = build_label(text="X Slice")

panel = ui.Panel2D(size=(300, 200), color=(1, 1, 1), opacity=0.1, align="right")
panel.center = (1030, 120)

panel.add_element(line_slider_label_x, (0.1, 0.75))
panel.add_element(line_slider_x, (0.38, 0.75))
panel.add_element(line_slider_label_y, (0.1, 0.55))
panel.add_element(line_slider_y, (0.38, 0.55))
panel.add_element(line_slider_label_z, (0.1, 0.35))
panel.add_element(line_slider_z, (0.38, 0.35))

show_m.scene.add(panel)

size = scene.GetSize()


def win_callback(obj, _event):
    global size
    if size != obj.GetSize():
        size_old = size
        size = obj.GetSize()
        size_change = [size[0] - size_old[0], 0]
        panel.re_align(size_change)


interactive = False

if interactive:
    show_m.add_window_callback(win_callback)
    show_m.render()
    show_m.start()
else:
    window.record(
        scene, out_path="odf_slicer_3D.png", size=(1200, 900), reset_camera=False
    )

del show_m
"""
    ),
    (
        18, "Render fractals in fury",
        "the image must contain some form of a fractal (anything resembling it in description is also fine)",
        """
import math

import numpy as np

from fury import primitive, ui, utils, window

###############################################################################
# Before we create our first fractal, let's set some ground rules for us to
# work with.
#
# 1. Instead of creating a new actor to represent each primitive of the
# fractal, we will compute the centers of each primitive and draw them at once
# using ``repeat_primitive()``.
#
# 2. How many primitives do we need? For each fractal, we define a depth which
# will prevent infinite recursion. Assuming we have a depth of :math:`N`, and
# at each level the shape is divided into :math:`k` smaller parts, we will need
# :math:`k^{N}` primitives to represent the fractal.
#
# 3. Ideally, we want to allocate the array of centers upfront. To achieve
# this, we can use the method of representing a binary tree in an array, and
# extend it to work with k-ary trees (formulas for the same can be found
# `here`_). In this scheme of representation, we represent every primitive as a
# node, and each sub-primitive as a child node. We can also skip storing the
# first :math:`\frac{k^{N} - 1}{k - 1} + 1` entries as we only need to render
# the leaf nodes. This allows us to create an array of exactly the required
# size at the start, without any additional overhead.
#
# .. _here: https://book.huihoo.com/data-structures-and-algorithms-with-object-oriented-design-patterns-in-c++/html/page356.html # noqa
#
# -----------------------------------------------------------------------------

###############################################################################
# The tetrix is a classic 3d fractal, a natural three-dimensional extension of
# the Sierpinski Triangle. At each level, we need to calculate the new centers
# for the next level. We can use the vertices of a tetrahedron as the offsets
# for the new centers, provided that the tetrahedron is centered at the origin
# (which is the case here).


def tetrix(N):
    centers = np.zeros((4**N, 3))

    # skipping non-leaf nodes (see above)
    offset = (4**N - 1) // 3 + 1

    # just need the vertices
    U, _ = primitive.prim_tetrahedron()

    def gen_centers(depth, pos, center, dist):
        if depth == N:
            centers[pos - offset] = center
        else:
            idx = 4 * (pos - 1) + 2
            for i in range(4):
                # distance gets halved at each level
                gen_centers(depth + 1, idx + i, center + dist * U[i], dist / 2)

    # the division by sqrt(6) is to ensure correct scale
    gen_centers(0, 1, np.zeros(3), 2 / (6**0.5))

    vertices, faces = primitive.prim_tetrahedron()

    # primitive is scaled down depending on level
    vertices /= 2 ** (N - 1)

    # compute some pretty colors
    bounds_min, bounds_max = np.min(centers, axis=0), np.max(centers, axis=0)
    colors = (centers - bounds_min) / (bounds_max - bounds_min)

    vertices, triangles, colors, _ = primitive.repeat_primitive(
        centers=centers, colors=colors, vertices=vertices, faces=faces
    )
    return utils.get_actor_from_primitive(vertices, triangles, colors)


###############################################################################
# For a Menger Sponge, each cube is divided into 27 smaller cubes, and we skip
# some of them (face centers, and the center of the cube). This means that on
# every level we get 20 new cubes.
#
# Here, to compute the points of each new center, we start at a corner cube's
# center and add the offsets to each smaller cube, scaled according to the
# level.


def sponge(N):
    centers = np.zeros((20**N, 3))
    offset = (20**N - 1) // 19 + 1

    # these are the offsets of the new centers at the next level of recursion
    # each cube is divided into 20 smaller cubes for a snowflake
    V = np.array(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 0, 2],
            [0, 1, 0],
            [0, 1, 2],
            [0, 2, 0],
            [0, 2, 1],
            [0, 2, 2],
            [1, 0, 0],
            [1, 0, 2],
            [1, 2, 0],
            [1, 2, 2],
            [2, 0, 0],
            [2, 0, 1],
            [2, 0, 2],
            [2, 1, 0],
            [2, 1, 2],
            [2, 2, 0],
            [2, 2, 1],
            [2, 2, 2],
        ]
    )

    def gen_centers(depth, pos, center, dist):
        if depth == N:
            centers[pos - offset] = center
        else:
            # we consider a corner cube as our starting point
            start = center - np.array([1, 1, 1]) * dist**0.5
            idx = 20 * (pos - 1) + 2

            # this moves from the corner cube to each new cube's center
            for i in range(20):
                # each cube is divided into 27 cubes so side gets divided by 3
                gen_centers(depth + 1, idx + i, start + V[i] * dist, dist / 3)

    gen_centers(0, 1, np.zeros(3), 1 / 3)

    vertices, faces = primitive.prim_box()
    vertices /= 3**N

    bounds_min, bounds_max = np.min(centers, axis=0), np.max(centers, axis=0)
    colors = (centers - bounds_min) / (bounds_max - bounds_min)

    vertices, triangles, colors, _ = primitive.repeat_primitive(
        centers=centers, colors=colors, vertices=vertices, faces=faces
    )
    return utils.get_actor_from_primitive(vertices, triangles, colors)


###############################################################################
# A snowflake is exactly the same as above, but we skip different cubes
# (corners and center). I think this looks quite interesting, and it is
# possible to see the Koch snowflake if you position the camera just right.


def snowflake(N):
    centers = np.zeros((18**N, 3))
    offset = (18**N - 1) // 17 + 1
    V = np.array(
        [
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [0, 1, 2],
            [0, 2, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 0, 2],
            [1, 1, 0],
            [1, 1, 2],
            [1, 2, 0],
            [1, 2, 1],
            [1, 2, 2],
            [2, 0, 1],
            [2, 1, 0],
            [2, 1, 1],
            [2, 1, 2],
            [2, 2, 1],
        ]
    )

    def gen_centers(depth, pos, center, side):
        if depth == N:
            centers[pos - offset] = center
        else:
            start = center - np.array([1, 1, 1]) * side**0.5
            idx = 18 * (pos - 1) + 2
            for i in range(18):
                gen_centers(depth + 1, idx + i, start + V[i] * side, side / 3)

    gen_centers(0, 1, np.zeros(3), 1 / 3)

    vertices, faces = primitive.prim_box()
    vertices /= 3**N

    bounds_min, bounds_max = np.min(centers, axis=0), np.max(centers, axis=0)
    colors = (centers - bounds_min) / (bounds_max - bounds_min)

    vertices, triangles, colors, _ = primitive.repeat_primitive(
        centers=centers, colors=colors, vertices=vertices, faces=faces
    )
    return utils.get_actor_from_primitive(vertices, triangles, colors)


###############################################################################
# Now that we have the functions to generate fractals, we can start setting up
# the Scene and ShowManager.

scene = window.Scene()
showmgr = window.ShowManager(scene, "Fractals", (800, 800), reset_camera=True)

###############################################################################
# These values are what work nicely on my machine without lagging. If you have
# a powerful machine, you could bump these up by around 2-3.

fractals = [tetrix(6), sponge(3), snowflake(3)]

###############################################################################
# We want to be able to switch between the three fractals. To achieve this
# we'll create a RadioButton and register a callback which will remove existing
# fractals and add the selected one. This also resets the camera.

options = {
    "Tetrix": 0,
    "Sponge": 1,
    "Snowflake": 2,
}

shape_chooser = ui.RadioButton(
    options.keys(),
    padding=10,
    font_size=16,
    checked_labels=["Tetrix"],
    position=(10, 10),
)


def choose_shape(radio):
    showmgr.scene.rm(*fractals)
    showmgr.scene.add(fractals[options[radio.checked_labels[0]]])
    showmgr.scene.reset_camera()


shape_chooser.on_change = choose_shape

# selected at start
showmgr.scene.add(fractals[0])
showmgr.scene.add(shape_chooser)

###############################################################################
# Let's add some basic camera movement to make it look a little interesting.
# We can use a callback here to update a counter and calculate the camera
# positions using the counter. ``sin`` and ``cos`` are used here to make smooth
# looping movements.

counter = 0


def timer_callback(_obj, _event):
    global counter
    counter += 1
    showmgr.scene.azimuth(math.sin(counter * 0.01))
    showmgr.scene.elevation(math.cos(counter * 0.01) / 4)
    showmgr.render()


showmgr.add_timer_callback(True, 20, timer_callback)

###############################################################################
# Finally, show the window if running in interactive mode or render to an image
# otherwise. This is needed for generating the documentation that you are
# reading.

interactive = True
if interactive:
    showmgr.start()
else:
    window.record(showmgr.scene, out_path="fractals.png", size=(800, 800))
"""
    ),
    (
        19, "Render a card with the body written 'FURY' and use image from `\"https://raw.githubusercontent.com/fury-gl/fury-communication-assets/main/fury-logo.png\"`",
        "The image must contain a card with the word FURY written on it",
        """
from fury import ui, window
from fury.data import fetch_viz_icons

##############################################################################
# First we need to fetch some icons that are included in FURY.

fetch_viz_icons()

###############################################################################
# Let's create a card and add it to the show manager

img_url = (
    "https://raw.githubusercontent.com/fury-gl"
    "/fury-communication-assets/main/fury-logo.png"
)

title = "FURY"
body = (
    "FURY - Free Unified Rendering in pYthon."
    "A software library for scientific visualization in Python."
)

card = ui.elements.Card2D(
    image_path=img_url,
    title_text=title,
    body_text=body,
    image_scale=0.55,
    size=(300, 300),
    bg_color=(1, 0.294, 0.180),
    bg_opacity=0.8,
    border_width=5,
    border_color=(0.1, 0.4, 0.4),
)

###############################################################################
# Now that the card has been initialised, we add it to the show
# manager.

current_size = (1000, 1000)
show_manager = window.ShowManager(size=current_size, title="FURY Card Example")

show_manager.scene.add(card)
# To interact with the UI, set interactive = True
interactive = False

if interactive:
    show_manager.start()

window.record(show_manager.scene, out_path="card_ui.png", size=(1000, 1000))
"""
    ),
    (
        20, "make a robot arm in fury and animate it",
        "the image will contain arrows and boxes etc",
        """
import numpy as np

from fury import actor, window
from fury.animation import Animation, Timeline
from fury.utils import set_actor_origin

scene = window.Scene()

showm = window.ShowManager(
    scene, size=(900, 768), reset_camera=False, order_transparent=True
)
showm.initialize()


###############################################################################
# Creating robot arm components

base = actor.cylinder(
    np.array([[0, 0, 0]]), np.array([[0, 1, 0]]), colors=(0, 1, 0), radius=1
)
main_arm = actor.box(np.array([[0, 0, 0]]), colors=(1, 0.5, 0), scales=(12, 1, 1))

sub_arm = actor.box(np.array([[0, 0, 0]]), colors=(0, 0.5, 0.8), scales=(8, 0.7, 0.7))
joint_1 = actor.sphere(np.array([[0, 0, 0]]), colors=np.array([1, 0, 1]), radii=1.2)
joint_2 = actor.sphere(np.array([[0, 0, 0]]), colors=np.array([1, 0, 1]))

end = actor.cone(
    np.array([[0, 0, 0]]),
    np.array([[1, 0, 0]]),
    np.array([[1, 0, 0]]),
    heights=2.2,
    resolution=6,
)

###############################################################################
# Setting the center of both shafts to the beginning.
set_actor_origin(main_arm, np.array([-6, 0, 0]))
set_actor_origin(sub_arm, np.array([-4, 0, 0]))

###############################################################################
# Creating a timeline
timeline = Timeline(playback_panel=True)

###############################################################################
# Creating animations
main_arm_animation = Animation([main_arm, joint_1], length=2 * np.pi)
child_arm_animation = Animation([sub_arm, joint_2])
drill_animation = Animation(end)


###############################################################################
# Adding other Animations in hierarchical order
main_arm_animation.add_child_animation(child_arm_animation)
child_arm_animation.add_child_animation(drill_animation)


###############################################################################
# Creating Arm joints time dependent animation functions.


def rot_main_arm(t):
    return np.array([np.sin(t / 2) * 180, np.cos(t / 2) * 180, 0])


def rot_sub_arm(t):
    return np.array([np.sin(t) * 180, np.cos(t) * 70, np.cos(t) * 40])


def rot_drill(t):
    return np.array([t * 1000, 0, 0])


###############################################################################
# Setting timelines (joints) relative position
# 1- Placing the main arm on the cube static base.
main_arm_animation.set_position(0, np.array([0, 1.3, 0]))

###############################################################################
# 2- Translating the timeline containing the sub arm to the end of the first
# arm.
child_arm_animation.set_position(0, np.array([12, 0, 0]))

###############################################################################
# 3- Translating the timeline containing the drill to the end of the sub arm.
drill_animation.set_position(0, np.array([8, 0, 0]))

###############################################################################
# Setting rotation time-based evaluators
main_arm_animation.set_rotation_interpolator(rot_main_arm, is_evaluator=True)
child_arm_animation.set_rotation_interpolator(rot_sub_arm, is_evaluator=True)
drill_animation.set_rotation_interpolator(rot_drill, is_evaluator=True)

###############################################################################
# Setting camera position to observe the robot arm.
scene.camera().SetPosition(0, 0, 90)

###############################################################################
# Adding the base actor to the scene
scene.add(base)

###############################################################################
# Adding the main parent animation to the Timeline.
timeline.add_animation(main_arm_animation)

###############################################################################
# Now we add the timeline to the ShowManager
showm.add_animation(timeline)

interactive = True

if interactive:
    showm.start()

window.record(scene, out_path="viz_robot_arm.png", size=(900, 768))
"""
    ),
    (
        21, "Render an animation where the ball collides with a brick wall",
        "the image will contain a wall and a ball moving against it", 
        """
import itertools

import numpy as np
import pybullet as p

from fury import actor, ui, utils, window

###############################################################################
# Next, we initialize a pybullet client to render the physics. We use `DIRECT`
# mode to initialize pybullet without a GUI.

p.connect(p.DIRECT)

###############################################################################
# Apply gravity to the scene. In pybullet all values are in SI units.
p.setGravity(0, 0, -10)

###############################################################################
# We define some global parameters so that its easier for us to tweak the
# tweak the simulation.

# Ball Parameters
ball_radius = 0.3
ball_color = np.array([1, 0, 0])
ball_mass = 3
ball_position = np.array([2, 0, 1.5])
ball_orientation = np.array([0, 0, 0, 1])

# Base Plane Parameters
base_size = np.array([5, 5, 0.2])
base_color = np.array([1, 1, 1])
base_position = np.array([0, 0, -0.1])
base_orientation = np.array([0, 0, 0, 1])

# Wall Parameters
wall_height = 10
wall_width = 10
brick_mass = 0.5
brick_size = np.array([0.2, 0.4, 0.2])

###############################################################################
# Now we define the required parameters to render the Ball.

# Ball actor
ball_actor = actor.sphere(
    centers=np.array([[0, 0, 0]]), colors=ball_color, radii=ball_radius
)

# Collision shape for the ball.
ball_coll = p.createCollisionShape(p.GEOM_SPHERE, radius=ball_radius)

# Creating a multi-body which will be tracked by pybullet.
ball = p.createMultiBody(
    baseMass=3,
    baseCollisionShapeIndex=ball_coll,
    basePosition=ball_position,
    baseOrientation=ball_orientation,
)

# Change the dynamics of the ball by adding friction and restitution.
p.changeDynamics(ball, -1, lateralFriction=0.3, restitution=0.5)

###############################################################################
# Render a base plane to support the bricks.

base_actor = actor.box(
    centers=np.array([[0, 0, 0]]),
    directions=[0, 0, 0],
    scales=base_size,
    colors=base_color,
)

base_coll = p.createCollisionShape(p.GEOM_BOX, halfExtents=base_size / 2)
# half of the actual size.

base = p.createMultiBody(
    baseCollisionShapeIndex=base_coll,
    basePosition=base_position,
    baseOrientation=base_orientation,
)

p.changeDynamics(base, -1, lateralFriction=0.3, restitution=0.5)

###############################################################################
# Now we render the bricks. All the bricks are rendered by a single actor for
# better performance.

nb_bricks = wall_height * wall_width

brick_centers = np.zeros((nb_bricks, 3))

brick_directions = np.zeros((nb_bricks, 3))
brick_directions[:] = np.array([1.57, 0, 0])

brick_orns = np.zeros((nb_bricks, 4))

brick_sizes = np.zeros((nb_bricks, 3))
brick_sizes[:] = brick_size

brick_colors = np.random.rand(nb_bricks, 3)

brick_coll = p.createCollisionShape(p.GEOM_BOX, halfExtents=brick_size / 2)

# We use this array to store the reference of brick objects in pybullet world.
bricks = np.zeros(nb_bricks, dtype=np.int8)

# Logic to position the bricks appropriately to form a wall.
i = 0
for k in range(wall_height):
    for j in range(wall_width):
        center_pos = np.array([-1, (j * 0.4) - 1.8, (0.2 * k) + 0.1])
        brick_centers[i] = center_pos
        brick_orns[i] = np.array([0, 0, 0, 1])
        bricks[i] = p.createMultiBody(
            baseMass=brick_mass,
            baseCollisionShapeIndex=brick_coll,
            basePosition=center_pos,
            baseOrientation=brick_orns[i],
        )
        p.changeDynamics(bricks[i], -1, lateralFriction=0.1, restitution=0.1)
        i += 1

brick_actor = actor.box(
    centers=brick_centers,
    directions=brick_directions,
    scales=brick_sizes,
    colors=brick_colors,
)

###############################################################################
# Now, we define a scene and add actors to it.

scene = window.Scene()
scene.add(actor.axes())
scene.add(ball_actor)
scene.add(base_actor)
scene.add(brick_actor)

# Create show manager.
showm = window.ShowManager(
    scene, size=(900, 768), reset_camera=False, order_transparent=True
)


# Counter iterator for tracking simulation steps.
counter = itertools.count()

# Variable for tracking applied force.
apply_force = True

###############################################################################
# Now, we define methods to sync objects between fury and Pybullet.

# Get the position of base and set it.
base_pos, _ = p.getBasePositionAndOrientation(base)
base_actor.SetPosition(*base_pos)

# Do the same for ball.
ball_pos, _ = p.getBasePositionAndOrientation(ball)
ball_actor.SetPosition(*ball_pos)

# Calculate the vertices of the bricks.
vertices = utils.vertices_from_actor(brick_actor)
num_vertices = vertices.shape[0]
num_objects = brick_centers.shape[0]
sec = int(num_vertices / num_objects)


###############################################################################
# ==============
# Syncing Bricks
# ==============
#
# Here, we perform three major steps to sync bricks accurately.
# * Get the position and orientation of the bricks from pybullet.
# * Calculate the Rotation Matrix.
#
#   - Get the difference in orientations (Quaternion).
#   - Generate the corresponding rotation matrix according to that difference.
#   - Reshape it in a 3x3 matrix.
#
# * Perform calculations to get the required position and orientation.
# * Update the position and orientation.


def sync_brick(object_index, multibody):
    pos, orn = p.getBasePositionAndOrientation(multibody)

    rot_mat = np.reshape(
        p.getMatrixFromQuaternion(
            p.getDifferenceQuaternion(orn, brick_orns[object_index])
        ),
        (3, 3),
    )

    vertices[object_index * sec : object_index * sec + sec] = (
        vertices[object_index * sec : object_index * sec + sec]
        - brick_centers[object_index]
    ) @ rot_mat + pos

    brick_centers[object_index] = pos
    brick_orns[object_index] = orn


###############################################################################
# A simpler but inaccurate approach is used here to update the position and
# orientation.


def sync_actor(actor, multibody):
    pos, orn = p.getBasePositionAndOrientation(multibody)
    actor.SetPosition(*pos)
    orn_deg = np.degrees(p.getEulerFromQuaternion(orn))
    actor.SetOrientation(*orn_deg)


###############################################################################
# Here, we define a textblock to display the Avg. FPS and simulation steps.

fpss = np.array([])
tb = ui.TextBlock2D(
    text="Avg. FPS: \nSim Steps: ", position=(0, 680), font_size=30, color=(1, 0.5, 0)
)
scene.add(tb)

###############################################################################
# Set the camera for better visualization.

scene.set_camera(
    position=(10.46, -8.13, 6.18),
    focal_point=(0.0, 0.0, 0.79),
    view_up=(-0.27, 0.26, 0.90),
)


###############################################################################
# Timer callback is created which is responsible for calling the sync and
# simulation methods.


# Create timer callback which will execute at each step of simulation.
def timer_callback(_obj, _event):
    global apply_force, fpss
    cnt = next(counter)
    showm.render()

    if cnt % 1 == 0:
        fps = showm.frame_rate
        fpss = np.append(fpss, fps)
        tb.message = (
            "Avg. FPS: " + str(np.round(np.mean(fpss), 0)) + "\nSim Steps: " + str(cnt)
        )

    # Get the position and orientation of the ball.
    ball_pos, ball_orn = p.getBasePositionAndOrientation(ball)

    # Apply force for 5 times for the first step of simulation.
    if apply_force:
        # Apply the force.
        p.applyExternalForce(
            ball, -1, forceObj=[-10000, 0, 0], posObj=ball_pos, flags=p.WORLD_FRAME
        )
        apply_force = False

    # Set position and orientation of the ball.
    sync_actor(ball_actor, ball)

    # Updating the position and orientation of each individual brick.
    for idx, brick in enumerate(bricks):
        sync_brick(idx, brick)
    utils.update_actor(brick_actor)

    # Simulate a step.
    p.stepSimulation()

    # Exit after 2000 steps of simulation.
    if cnt == 130:
        showm.exit()


# Add the timer callback to showmanager.
# Increasing the duration value will slow down the simulation.
showm.add_timer_callback(True, 1, timer_callback)

interactive = True

# start simulation
if interactive:
    showm.start()

window.record(scene, out_path="viz_brick_wall.png", size=(900, 768))
"""
    ),
    (
        22, "Render a scene where 2 balls are coming to collide in fury.",
        "The image can contain 2 balls coming at each other to collide",
        """
import itertools

import numpy as np
import pybullet as p

from fury import actor, ui, window

client = p.connect(p.DIRECT)

###############################################################################
# Parameters and definition of red and blue balls.

red_radius = 0.5
blue_radius = 0.5
duration = 50

# Red Ball
red_ball_actor = actor.sphere(
    centers=np.array([[0, 0, 0]]), colors=np.array([[1, 0, 0]]), radii=red_radius
)

red_ball_coll = p.createCollisionShape(p.GEOM_SPHERE, radius=red_radius)

red_ball = p.createMultiBody(
    baseMass=0.5,
    baseCollisionShapeIndex=red_ball_coll,
    basePosition=[10, 0, 0],
    baseOrientation=[0, 0, 0, 1],
)

# Blue ball
blue_ball_actor = actor.sphere(
    centers=np.array([[0, 0, 0]]), colors=np.array([[0, 0, 1]]), radii=blue_radius
)

blue_ball_coll = p.createCollisionShape(p.GEOM_SPHERE, radius=blue_radius)

blue_ball = p.createMultiBody(
    baseMass=0.5,
    baseCollisionShapeIndex=blue_ball_coll,
    basePosition=[-10, 0, 0],
    baseOrientation=[0, 0, 0, 1],
)

###############################################################################
# We set the coefficient of restitution of both the balls to `0.6`.

p.changeDynamics(red_ball, -1, restitution=0.6)
p.changeDynamics(blue_ball, -1, restitution=0.6)

###############################################################################
# We add all the actors to the scene.

scene = window.Scene()
scene.add(actor.axes())
scene.add(red_ball_actor)
scene.add(blue_ball_actor)

showm = window.ShowManager(
    scene, size=(900, 700), reset_camera=False, order_transparent=True
)


counter = itertools.count()

###############################################################################
# Method to sync objects.


def sync_actor(actor, multibody):
    pos, orn = p.getBasePositionAndOrientation(multibody)
    actor.SetPosition(*pos)
    orn_deg = np.degrees(p.getEulerFromQuaternion(orn))
    actor.SetOrientation(*orn_deg)


apply_force = True
tb = ui.TextBlock2D(position=(0, 600), font_size=30, color=(1, 0.5, 0), text="")
scene.add(tb)
scene.set_camera(
    position=(0.30, -18.78, 0.89), focal_point=(0.15, 0.25, 0.40), view_up=(0, 0, 1.00)
)


###############################################################################
# Timer callback to sync and step simulation every second.


def timer_callback(_obj, _event):
    global apply_force
    cnt = next(counter)
    showm.render()
    red_pos, red_orn = p.getBasePositionAndOrientation(red_ball)
    blue_pos, blue_orn = p.getBasePositionAndOrientation(blue_ball)

    # Apply force for the first step of the simulation.
    if apply_force:
        p.applyExternalForce(
            red_ball, -1, forceObj=[-40000, 0, 0], posObj=red_pos, flags=p.WORLD_FRAME
        )

        p.applyExternalForce(
            blue_ball, -1, forceObj=[40000, 0, 0], posObj=blue_pos, flags=p.WORLD_FRAME
        )

        apply_force = 0

    sync_actor(blue_ball_actor, blue_ball)
    sync_actor(red_ball_actor, red_ball)

    # Get various collision information using `p.getContactPoints`.
    contact = p.getContactPoints(red_ball, blue_ball, -1, -1)
    if len(contact) != 0:
        tb.message = "Collision!!"

    p.stepSimulation()

    if cnt == 50:
        showm.exit()


showm.add_timer_callback(True, duration, timer_callback)

interactive = True

if interactive:
    showm.start()

window.record(scene, size=(900, 700), out_path="viz_ball_collide.png")
"""
    ),
]