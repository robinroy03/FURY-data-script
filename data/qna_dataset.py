"""
The dataset for QnA questions.

Questions will be answered using normal FURY Engine prompt templates. We'll use another LLM to judge the relevancy and other factors wrt to the base ground_truth answer.
"""


# [(INDEX, LLM_PROMPT, GROUND_TRUTH)]
QNA_QUESTIONS_DISCORD = [
    (
        0, "Hi folks. I am wondering if someone knows an easy way to add (anatomical) orientation information in the form of three orthogonal sticks with the corresponding colors and legend (see image) to a FURY scene so that it always appears in the lower right/left corner of the scene, it is scaled to the size of the scene, and ideally rotates as we rotate the scene. Thanks.",
        "The resultant output must mention fury actor.axes()"
    ),
    (
        1, """Is there an easy way to change the opacity of an imported actor from polydata? 

My case is as follows: 
allen_brain = io.load_polydata(mesh_path)
brain_actor = utils.get_actor_from_polydata(allen_brain)
scene.add(brain_actor)""",
    "The answer should reference brain_actor.GetProperty().SetOpacity(0.5)."
    ),
    (
        2, """I was playing around with fury, and also downloaded the examples from official website
After downloading and executing once now whenever I execute the program it just create a PNG img in the same folder... 
I am not getting why it's happening can anyone help me??""",
    "You need to edit the file and set the interactive variable to True inorder to interact with them."
    ),
    (
        3, "can someone help me as when i move the cylinder to the hollow part it darkens.How to fix this?",
        "set capped=True"
    ),
    (
        4, """How to run single tests??
Suppose I have to just run test_ui then how can I run it??""",
        """You can test a specific test by
pytest -svv path/to/file -k "name of test"""
    ),
    (
        5, """can you explain to me a bit how window.analyze_snapshot works.
How it finds objects that are present in the snapshot?""",
        """It does 2 things:
- look for every colors in the image and count them. A bit like a histogram
- convert the image in greyscale, small erosion in case 2 objects are adjacent and search connected components greyscale different from background. A classic labellisation algorithm"""
    ),
    (
        6, "Is it possible to use the Fury Showm Manager object inside of a environment with no X-server (or wayland) available ?",
        "You will have to use xvfb with offscreen=True.  You can check FURY GitHub action or azure pipeline"
    ),
    (
        7, """Hi folks. I'd like to display a list of vectors with different colors according to a given feature. The feature could be a list of numbers e.g. in the [0, 1] range, one for each vector. Unlike the line actor where this is a built-in possibility by providing the colors parameter such a list, the arrow actor requires colors to be already an RGB(A) array. How can this be done? Thanks.""",
        "from fury.colormap import create_colormap"
    ),
    (
        8, """Hi. Does anyone know how to use vtkmarchingcubes to create a mesh? I have the integer coordinates of the points composing the isosurface.""",
        """Marching cubes takes a volume as input: https://vtk.org/doc/nightly/html/classvtkMarchingCubes.html#details
The isosurface is computed on the values of such input data:
mc = vtk.vtkMarchingCubes()
mc.SetInputData(volume)
mc.ComputeNormalsOn()
mc.ComputeGradientsOn()
mc.SetValue(0, threshold)
mc.Update()

A self-contained example can be found here: https://kitware.github.io/vtk-examples/site/Python/Modelling/MarchingCubes/"""
    ),
    (
        9, """Hi, I was going through the fury tutorials and I'm not sure why the code is not working (https://fury.gl/latest/auto_examples/01_introductory/viz_texture.html#sphx-glr-auto-examples-01-introductory-viz-texture-py)

I tried downloading the viz_texture.py file and running it, but the output is not generated.

For me, the output was (modified the path)
Dataset is already in place. If you want to fetch it again please first remove the folder C:\Users\......\.fury\\textures

Once I comment out fetch_viz_textures() that message is also gone. 
I can find the earth textures in the path location""",
    "Set interactive=True"
    ),
    (
        10, """I tried sending a mail to the fury mailing list, it's shown as being held. Is that normal? 


The message is being held because:

    The message comes from a moderated member""",
    "This is normal to avoid spam"
    ),
    (
        11, """I had a doubt regarding the Scrollbar refactoring. You suggested not to use ListBox2D to connect with the ScrollBar class, but the update function is required in Scrolling. Without an object of ListBox2D class in Scrollbar how can obtain the functioning of update on the list?""",
        """You can implement the update function anonymously so that others can use it accordingly. You can see the on_change function of LineSlider2D for reference
"""
    ),
    (
        12, """I was wondering how to use the text_template argument in ui.LineSlider2D.

To be more specific, I want it to show only integer values, instead of the default .1f and ratio. Does anyone know how or any documentation I should be looking at?""",
        """it should work if you just do text_template="{value:.1f} "
is it what you need ?
for only integer: text_template="{value:d} "  I suppose
"""
    ),
    (
        13, """I am trying to implement a custom event listener, so far I know that custom events can be registered as User events with some offset and some callback can be binded with the event the thing that I can't figure out is how to actually define what triggers these custom events . Is there a way to do so in vtk? 
""",
        """Inside vtk c++ code , there is  an method  InvokeEvent which it is associated with any vtkObject
https://vtk.org/doc/nightly/html/vtkObject_8h_source.html 
https://vtk.org/doc/nightly/html/vtkCommand_8h_source.html
Now take as example the UpdateShaderEvent , https://gitlab.kitware.com/vtk/vtk/-/blob/65fda83261598490bd643ec39865acad9ce62c03/Rendering/OpenGL2/vtkOpenGLPolyDataMapper.cxx#L1810  this->InvokeEvent from vtkObject it's called using the int related with UpdateShaderEvent.
Therefore, any  call of vtkOpenGLPolyDataMapper::UpdateShaders wil ltrigger the UpdateShaderEvent associated with the vtkObject 
https://gitlab.kitware.com/vtk/vtk/-/blob/65fda83261598490bd643ec39865acad9ce62c03/Rendering/OpenGL2/vtkOpenGLPolyDataMapper.cxx#L1754"""
    ),
    (
        14, """can we add additional arguments  for the callback in AddObserver""",
        """you can use the python functional tools to achieve this
For example, 

def callback(
        _caller, _event, calldata=None,
            uniform_type='f', uniform_name=None, value=None):
        program = calldata
        if program is not None:
            program.__getattribute__(f'SetUniform{uniform_type}')(
                uniform_name, value)

    add_shader_callback(
            sq_actor, partial(
                callback, uniform_type='f', uniform_name='edgeWidth',
                value=edge_width))
you should import 
from functools import partial
In this example, uniform_type, uniform_name and value are the additional arguments to be used inside the observer event.
"""
    ),
    (
        15, """Do all UI elements that are present in the scene share the same instance of the window or do they have seperate instances?
""",
        """They share the same instance"""
    ),
    (
        16, """While working with fury.actor.line, the lod_points_size argument is used to manipulate the thickness of the line to be rendered is what I could infer from the API Reference. But changing the values of lod_points_size doesn't seem to change the thickness of the line. Is it used for something else?
The thickness of both the lines ought to be different I guess as lod_points_size argument for both the lines has different values yet the lines seem to have the same thickness.""",
        """To change the line thickness use the linewidth parameter. lod_points_size is only in effect when the scene complexity requires downgrading the rendering from lines to points. This is a technique called Level Of Detail"""
    ),
    (
        17, "Hi how to contribute to FURY for GSoC",
        """Hi, first learn the FURY repo through the examples provided on website. Then go through the open issues on github and try tackling them. When it is time we'll let you know to send in a proposal. Make a proposal and submit. A good blog is here: https://levelup.gitconnected.com/my-journey-guide-for-google-summer-of-code-7b03e1007c07"""
    )
]



QNA_QUESTIONS_GITHUB = []