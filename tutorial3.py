from random import gauss, uniform
from time import time, sleep
from math import pi, cos, sin
from OpenGL.GL import glViewport, glClearColor, glClear,\
     glEnable, GL_DEPTH_TEST, glDisable, GL_LIGHTING,\
     GL_NORMALIZE, glShadeModel, GL_FLAT, glMatrixMode,\
     GL_PROJECTION, glLoadIdentity, GL_MODELVIEW, glLightfv,\
     GL_LIGHT0, GL_POSITION, GL_DIFFUSE, GL_SPECULAR,\
     glPushMatrix, glMultMatrixf, glScalef, glPopMatrix
from OpenGL.GLU import gluPerspective, gluLookAt
from OpenGL.GLUT import glutInit, glutInitDisplayMode,\
     GLUT_RGB, glutInitWindowPosition, glutInitWindowSize,\
     glutCreateWindow, glutKeyboardFunc, glutDisplayFunc,\
     glutIdleFunc, glutMainLoop, glutSwapBuffers,\
     glutPostRedisplay, glutSolidCube
from ode import World, Body, Mass, GeomBox, collide,\
     ContactJoint, Space, GeomPlane, JointGroup
from numpy import array
from numpy.linalg import norm

# geometric utility functions
def scalp (vec, scal):
    vec = array(vec)
    
    vec *= scal
    
def length (vec):
    vec = array(vec)

    return norm(vec)

# prepare_GL
def prepare_GL():
    """Prepare drawing."""
    # Viewport
    glViewport(0, 0, w, h)

    # Initialize
    glClearColor(0.8, 0.8, 0.9, 0)
    glClear(16640)
    glEnable(GL_DEPTH_TEST)
    glDisable(GL_LIGHTING)
    glEnable(GL_LIGHTING)
    glEnable(GL_NORMALIZE)
    glShadeModel(GL_FLAT)

    # Projection
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective (45, w / h, 0.2, 20)

    # Initialize ModelView matrix
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    # Light source
    glLightfv(GL_LIGHT0, GL_POSITION, (0,0,1,0))
    glLightfv(GL_LIGHT0, GL_DIFFUSE, (1,1,1,1))
    glLightfv(GL_LIGHT0, GL_SPECULAR, (1,1,1,1))
    glEnable(GL_LIGHT0)

    # View transformation
    gluLookAt (2.4, 3.6, 4.8, 0.5, 0.5, 0, 0, 1, 0)

# draw_body
def draw_body(body):
    """Draw an ODE body."""
    x, y, z = body.getPosition()
    R = body.getRotation()
    rot = (R[0], R[3], R[6], 0,
           R[1], R[4], R[7], 0,
           R[2], R[5], R[8], 0,
           x, y, z, 1)

    glPushMatrix()

    glMultMatrixf(rot)

    if body.shape == "box":
        sx, sy, sz = body.boxsize
        glScalef(sx, sy, sz)
        glutSolidCube(1)

    glPopMatrix()

# create_box
def create_box(world, space, density, lx, ly, lz):
    """Create a box body and its corresponding geom."""
    # Create body
    body = Body(world)
    M = Mass()
    M.setBox(density, lx, ly, lz)
    body.setMass(M)

    # Set parameters for drawing the body
    body.shape = "box"
    body.boxsize = (lx, ly, lz)

    # Create a box geom for collision detection
    geom = GeomBox(space, lengths = body.boxsize)
    geom.setBody(body)

    return body, geom

# drop_object
def drop_object():
    """Drop an object into the scene."""
    global bodies, geom, counter, objcount

    body, geom = create_box(world, space, 1000, 1, 0.2, 0.2)
    body.setPosition((gauss(0, 0.1), 3.0, gauss(0, 0.1)))
    theta = uniform(0, 2 * pi)
    ct = cos(theta)
    st = sin(theta)
    body.setRotation((ct, 0, -st, 0, 1, 0, st, 0, ct))
    bodies.append(body)
    geoms.append(geom)
    counter = 0
    objcount += 1

# explosion
def explosion():
    """Simulate an explosion.
    Every object is pushed away from the origin.
    The force is dependent on the objects distance from the origin."""
    global bodies

    for b in bodies:
        l = b.getPosition()
        d = length(l)
        a = max(0, 4e4 * (1 - d ** 2 / 5))
        l = array(l)
        l /= (4, 1, 4)
        scalp(l, a / length(l))
        b.addForce(l)

# pull
def pull():
    """Pull the objects back to the origin.
    Every object will be pulled back to the origin.
    Every couple of frames there'll be a thrust upwards so that
    the objects won't stick to the ground all the time."""
    global bodies, counter

    for b in bodies:
        l = list(b.getPosition())
        scalp(l, -1e3 / length(l))
        b.addForce(l)

        if counter % 60 == 0:
            b.addForce((0, 1e4, 0))

# Collision callback
def near_callback(args, geom1, geom2):
    """Callback function for the collide() method.

    This function checks if the given geoms do collide and
    creates contact joints if they do."""
    # Check if the objects do collide
    contacts = collide(geom1, geom2)

    # Create contact joints
    world,contactgroup = args
    
    for c in contacts:
        c.setBounce(0.2)
        c.setMu(5000)
        j = ContactJoint(world, contactgroup, c)
        j.attach(geom1.getBody(), geom2.getBody())

######################################################################

# Initialize Glut
glutInit()

# Open a window
glutInitDisplayMode(GLUT_RGB)

x, y, w, h = 0, 0, 1280, 720
glutInitWindowPosition(x, y)
glutInitWindowSize(w, h)
glutCreateWindow("")

# Create a world object
world = World()
world.setGravity(10 * array((0, -1, 0)))
world.setERP(0.8)
world.setCFM(1e-5)

# Create a space object
space = Space()

# Create a plane geom which prevent the objects from falling forever
floor = GeomPlane(space, (0, 1, 0), 0)

# A list with ODE bodies
bodies = []

# The geoms for each of the bodies
geoms = []

# A joint group for the contact joints that are generated whenever
# two bodies collide
contactgroup = JointGroup()

# Some variables used inside the simulation loop
fps, running, state, counter, objcount, lasttime = 100, True, 0,\
                                                0, 0, time()
dt = 1 / fps

# keyboard callback
def _keyfunc (c, x, y):
    exit(0)

glutKeyboardFunc(_keyfunc)

# draw callback
def _drawfunc ():
    # Draw the scene
    prepare_GL()

    for b in bodies:
        draw_body(b)

    glutSwapBuffers()

glutDisplayFunc(_drawfunc)

# idle callback
def _idlefunc ():
    global counter, state, lasttime

    t = dt - (time() - lasttime) #?
    print(t)

    if t > 0:
        sleep(t)

    counter += 1

    if state == 0:
        if counter == 20:
            drop_object()

        if objcount == 30:
            state = 1
            counter = 0
            
    # State 1: Explosion and pulling back the objects
    elif state == 1:
        if counter == 100:
            explosion()

        if counter > 300:
            pull()

        if counter == 500:
            counter = 20

    glutPostRedisplay ()

    # Simulate
    n = 2

    for i in range(n):
        # Detect collisions and create contact joints
        space.collide((world, contactgroup), near_callback)

        # Simulation step
        world.step(dt / n)

        # Remove all contact joints
        contactgroup.empty()

    lasttime = time()

glutIdleFunc(_idlefunc)

glutMainLoop()
