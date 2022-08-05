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
from numpy import array, ones
from numpy.linalg import norm
    
# prepare_GL
def prepare_GL():
    """Prepare drawing."""
    # Viewport
    glViewport(0, 0, *(i for i in wh))

    # Initialize
    glClearColor(0.8, 0.8, 0.9, 0)
    glClear(16640)
    tuple(map(glEnable, (GL_DEPTH_TEST, GL_LIGHTING, GL_NORMALIZE)))
    glShadeModel(GL_FLAT)

    w, h = wh

    # Projection
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective (45, w / h, 0.2, 20)

    # Initialize ModelView matrix
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    # Light source
    unos = ones(3)
    (glLightfv(a) for a in ((GL_LIGHT0, GL_POSITION, (0,0,1)),\
                            (GL_LIGHT0, GL_DIFFUSE, unos),\
                            (GL_LIGHT0, GL_SPECULAR, unos)))
        
    glEnable(GL_LIGHT0)

    # View transformation
    gluLookAt(2.4, 3.6, 4.8, *(i for i in array((1,1,0)) / 2),\
              *(i for i in arriba))

# draw_body
def draw_body(body):
    """Draw an ODE body."""
    rot = list(array(body.getRotation()).reshape(3, 3).\
               transpose().reshape(9))
    
    rot.insert(3, 0)
    rot.insert(7, 0) #no mape
    rot.extend((0, *(i for i in body.getPosition()), 1))

    glPushMatrix()

    glMultMatrixf(tuple(rot))

    if body.shape == "box":
        glScalef(*(i for i in body.boxsize))
        glutSolidCube(1)

    glPopMatrix()

# create_box
def create_box(world, space, density, lx, ly, lz):
    """Create a box body and its corresponding geom."""
    # Create body
    body, M = Body(world), Mass()
    M.setBox(density, lx, ly, lz)
    body.setMass(M)

    # Set parameters for drawing the body
    body.shape, body.boxsize = "box", (lx, ly, lz)

    # Create a box geom for collision detection
    geom = GeomBox(space, lengths = body.boxsize)
    geom.setBody(body)

    return body, geom

arriba = (0, 1, 0)

# drop_object
def drop_object():
    """Drop an object into the scene."""
    global geom, counter, objcount

    body, geom = create_box(world, space, 1000, 1, 0.2, 0.2)
    body.setPosition((gauss(0, 0.1), 3, gauss(0, 0.1)))
    theta = uniform(0, 2 * pi)
    ct, st = cos(theta), sin(theta)
    body.setRotation((ct, 0, -st,  *(i for i in arriba),\
                      st, 0, ct))
    bodies.append(body)
    geoms.append(geom)
    counter = 0
    objcount += 1

# explosion
def explosion():
    """Simulate an explosion.
    Every object is pushed away from the origin.
    The force is dependent on the objects distance from the origin."""
    for b in bodies:
        l = array(b.getPosition())
        a = max(0, 4e4 * (1 - norm(l) ** 2 / 5))
        l /= (4, 1, 4) 
        l = l / norm(l) * a  
        b.addForce(l)

# pull
def pull():
    """Pull the objects back to the origin.
    Every object will be pulled back to the origin.
    Every couple of frames there'll be a thrust upwards so that
    the objects won't stick to the ground all the time."""
    for b in bodies:
        l = array(b.getPosition())
        b.addForce(-array(l) / norm(l) * 1e3)

        if counter % 60 == 0:
            b.addForce(1e4 * array(arriba))

# Collision callback
def near_callback(args, geom1, geom2):
    """Callback function for the collide() method.
    This function checks if the given geoms do collide and
    creates contact joints if they do."""
    # Check if the objects do collide
    # Create contact joints    
    for c in collide(geom1, geom2):
        c.setBounce(0.2)
        c.setMu(5000)
        ContactJoint(*(i for i in args), c).\
                         attach(geom1.getBody(), geom2.getBody())

######################################################################

# Initialize Glut
glutInit()

# Open a window
glutInitDisplayMode(GLUT_RGB)

xy = (0, 0)
wh = (1280, 720)
glutInitWindowPosition(*(i for i in xy))
glutInitWindowSize(*(i for i in wh))
glutCreateWindow("")

# Create a world object
world = World()
world.setGravity(-10 * array(arriba))
world.setERP(0.8)
world.setCFM(1e-5)

# Create a space object
space = Space()

# Create a plane geom which prevent the objects from falling forever
# A list with ODE bodies
# The geoms for each of the bodies
# A joint group for the contact joints that are generated whenever
# two bodies collide
# Some variables used inside the simulation loop
floor, bodies, geoms, contactgroup, fps, running, state, counter,\
       objcount, lasttime = GeomPlane(space, arriba, 0), [], [],\
                            JointGroup(), 100, True, 0, 0, 0,\
                            time()
dt = 1 / fps

# keyboard callback
def _keyfunc (c, x, y):
    exit()

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

    t = dt - time() + lasttime 

    if t > 0:
        sleep(t)

    counter += 1

    if state == 0:
        if counter == 20:
            drop_object()

        if objcount == 30:
            state, counter = 1, 0
            
    # State 1: Explosion and pulling back the objects
    elif state == 1:
        if counter == 100:
            explosion()

        if counter > 300:
            pull()

        if counter == 500:
            counter = 20

    glutPostRedisplay()

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
