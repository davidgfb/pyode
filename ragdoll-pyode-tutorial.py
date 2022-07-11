from time import time, sleep
from random import uniform

from math import pi, acos, cos, sin
from OpenGL.GL import glClearColor, glClear, glEnable,\
     GL_DEPTH_TEST, GL_LIGHTING, GL_NORMALIZE, glShadeModel,\
     GL_SMOOTH, glMatrixMode, GL_PROJECTION, glLoadIdentity,\
     glViewport, GL_MODELVIEW, glLightfv, GL_LIGHT0,\
     GL_POSITION, GL_DIFFUSE, GL_SPECULAR, GL_COLOR_MATERIAL,\
     glColor3f, glPushMatrix, glMultMatrixf, glTranslatef,\
     glPopMatrix
from OpenGL.GLU import gluPerspective, gluLookAt, gluNewQuadric,\
     gluQuadricNormals, GLU_SMOOTH, gluQuadricTexture,\
     gluCylinder
from OpenGL.GLUT import glutInit, glutInitDisplayMode,\
     glutInitWindowPosition, glutInitWindowSize,\
     glutCreateWindow, glutKeyboardFunc,\
     glutDisplayFunc, glutIdleFunc, glutMainLoop,\
     glutSolidSphere, glutSwapBuffers, glutSetWindowTitle,\
     glutPostRedisplay
from ode import Infinity, Body, Mass, GeomCCylinder, FixedJoint,\
     HingeJoint, ParamLoStop, ParamHiStop, UniversalJoint,\
     ParamLoStop2, ParamHiStop2, BallJoint, areConnected,\
     collide, ContactJoint, World, Space, GeomPlane, JointGroup

from numpy import array, cross, zeros, matmul
from numpy.linalg import norm
from sympy import Matrix, symbols, Poly

def a_Array(a, b):
    return (array(a), array(b))

def add3(a, b):
    """Returns the sum of 3-vectors a and b."""
    a, b = a_Array(a, b)

    return a + b

def sub3(a, b):
    """Returns the difference between 3-vectors a and b."""
    a, b = a_Array(a, b)

    return a - b 

def mul3(v, s):
    """Returns 3-vector v multiplied by scalar s."""
    v = array(v)

    return s * v

def dist3(a, b):
    """Returns the distance between point 3-vectors a and b."""
    a, b = a_Array(a, b)

    return norm(a - b)

def norm3(v):
    """Returns the unit length 3-vector parallel to 3-vector v."""
    v = array(v)
    l = norm(v)
    normalizado = zeros(3)
    
    try:
        normalizado = v / l 

    except:    
        print('e: division entre 0')

    return normalizado

def dot3(a, b):
    """Returns the dot product of 3-vectors a and b."""
    a, b = a_Array(a, b)

    return a @ b

def cross(a, b):
    """Returns the cross product of 3-vectors a and b.
    a = array(a)
    b = array(b)
    return cross(a, b) == a * b? pasa a iterativo"""
    x, y, z = symbols('x y z')
    c = Matrix(((x, y, z), a, b)).det()

    x1, y1, z1 = 0, 0, 0

    pos = 0

    args = c.args

    try:
        if float(args[0]):
            args = args[0] * args[1]
        
    except:
        pass
        #print('e: no es float')

    try:
        len(args)

    except:
        args = (args,)
        #print('e: es un solo elemento')  

    for mono in args:
        literal = mono.free_symbols
        coef = float(Poly(mono).coeffs()[0])
        
        if x in literal:
            x1 = coef

        if y in literal:
            y1 = coef

        if z in literal:
            z1 = coef

        pos += 1

    return (x1, y1, z1)

def project3(v, d):
    """Returns projection of 3-vector v onto unit 3-vector d."""
    v = array(v)
    l = norm(v)
    proyectado = zeros(3)

    try:
        normalizado = v / l
        proyectado = normalizado @ d * array(v)

    except:   
        print('e: division entre 0')
            
    return proyectado

def acosdot3(a, b):
    """Returns the angle between unit 3-vectors a and b."""
    a, b = a_Array(a, b)

    x = a @ b

    if x < -1:
        angulo = pi

    elif x > 1:
        angulo = 0

    else: # -1 < x < 1
        angulo = acos(x)

    return angulo

def rotate3(m, v):
    """Returns the rotation of 3-vector v by 3x3 (row major) matrix m."""
    return matmul(array(v), array(m).reshape(3, 3).transpose())

def invert3x3(m):
    """Returns the inversion (transpose) of 3x3 rotation matrix m."""
    return array(m).reshape(3,3).transpose().reshape(9)

def zaxis(m):
    """Returns the z-axis vector from 3x3 (row major) rotation matrix m."""
    return array(m).reshape(3, 3).transpose()[2]

def calcRotMatrix(axis, angle):
    """Returns the row-major 3x3 rotation matrix defining a rotation around axis by
    angle. Esto conviene q lo haga opengl""" 
    cosTheta = cos(angle)
    sinTheta = sin(angle)
    t = 1 - cosTheta
    return (t * axis[0] ** 2 + cosTheta,
            t * axis[0] * axis[1] - sinTheta * axis[2],
            t * axis[0] * axis[2] + sinTheta * axis[1],

            t * axis[0] * axis[1] + sinTheta * axis[2],
            t * axis[1] ** 2 + cosTheta,
            t * axis[1] * axis[2] - sinTheta * axis[0],

            t * axis[0] * axis[2] - sinTheta * axis[1],
            t * axis[1] * axis[2] + sinTheta * axis[0],
            t * axis[2] ** 2 + cosTheta)

def makeOpenGLMatrix(r, p):
    """Returns an OpenGL compatible (column-major, 4x4 homogeneous) transformation
    matrix from ODE compatible (row-major, 3x3) rotation matrix r and position
    vector p."""
    r = array(r).reshape(3, 3).transpose().tolist() 

    return (r[0] + [0] + r[1] + [0] + r[2] + [0] + list(p) + [1])

def getBodyRelVec(b, v):
    """Returns the 3-vector v transformed into the local coordinate system of ODE
    body b."""
    return rotate3(invert3x3(b.getRotation()), v)

'''rotation directions are named by the third (z-axis) row of the 3x3 matrix,
because ODE capsules are oriented along the z-axis'''
rightRot = array((0, 0, -1, 0, 1, 0, 1, 0, 0))
leftRot = array((0, 0, 1, 0, 1, 0, -1, 0, 0))
upRot = array((1, 0, 0, 0, 0, -1, 0, 1, 0))
downRot = array((1, 0, 0, 0, 0, -1, 0, 1, 0))
bkwdRot = array((1, 0, 0, 0, 1, 0, 0, 0, 1))

# axes used to determine constrained joint rotations
rightAxis, upAxis, bkwdAxis = array((1, 0, 0)), array((0, 1, 0)),\
                              array((0, 0, 1))
leftAxis, downAxis, fwdAxis = -rightAxis, -upAxis, -bkwdAxis 

UPPER_ARM_LEN, FORE_ARM_LEN, HAND_LEN, FOOT_LEN, HEEL_LEN = 0.3,\
                                        0.25, 0.13, 0.18, 0.05
'''HAND_LEN wrist to mid-fingers only
FOOT_LEN ankles to base of ball of foot only'''

BROW_H, MOUTH_H, NECK_H, SHOULDER_H, CHEST_H, HIP_H, KNEE_H,\
        ANKLE_H, SHOULDER_W, CHEST_W, LEG_W, PELVIS_W = 1.68,\
        1.53, 1.5, 1.37, 1.35, 0.86, 0.48, 0.08, 0.41, 0.36,\
        0.28, 0.25
'''CHEST_W actually wider, but we want narrower than shoulders (esp. with large radius)
LEG_W between middles of upper legs
PELVIS_W actually wider, but we want smaller than hip width'''

j = array((-1, 1, 1))
L_SHOULDER_POS = array((SHOULDER_W / 2, SHOULDER_H, 0))
R_SHOULDER_POS = L_SHOULDER_POS * j

k = UPPER_ARM_LEN * rightAxis
R_ELBOW_POS = R_SHOULDER_POS - k
L_ELBOW_POS = L_SHOULDER_POS + k

k = FORE_ARM_LEN * rightAxis
R_WRIST_POS = R_ELBOW_POS - k
L_WRIST_POS = L_ELBOW_POS + k

k = HAND_LEN * rightAxis
R_FINGERS_POS = R_WRIST_POS - k
L_FINGERS_POS = L_WRIST_POS + k

L_HIP_POS = array((LEG_W / 2, HIP_H, 0))
R_HIP_POS = L_HIP_POS * j

L_KNEE_POS = array((LEG_W / 2, KNEE_H, 0))
R_KNEE_POS = L_KNEE_POS * j

L_ANKLE_POS = array((LEG_W / 2, ANKLE_H, 0))
R_ANKLE_POS = L_ANKLE_POS * j

k = HEEL_LEN * bkwdAxis
R_HEEL_POS = R_ANKLE_POS - k
L_HEEL_POS = L_ANKLE_POS - k

k = FOOT_LEN * bkwdAxis
R_TOES_POS = R_ANKLE_POS + k
L_TOES_POS = L_ANKLE_POS + k

class RagDoll():
    def __init__(self, world, space, density, offset = zeros(3)):
        """Creates a ragdoll of standard size at the given offset."""
        self.world = world
        self.space = space
        self.density = density
        self.bodies = []
        self.geoms = []
        self.joints = []
        self.totalMass = 0

        self.offset = offset

        k = (CHEST_W / 2, CHEST_H, 0)
        self.chest = self.addBody(k * j, k, 0.13)
        self.belly = self.addBody((0, CHEST_H - 0.1, 0),
                                  (0, HIP_H + 0.1, 0), 0.125)
        self.midSpine = self.addFixedJoint(self.chest, self.belly)
        self.pelvis = self.addBody((-PELVIS_W / 2, HIP_H, 0),
                                  (PELVIS_W / 2, HIP_H, 0), 0.125)
        self.lowSpine = self.addFixedJoint(self.belly,\
                                           self.pelvis)

        self.head = self.addBody((0, BROW_H, 0), (0, MOUTH_H, 0),\
                                 0.11)

        k = pi / 4
        self.neck = self.addBallJoint(self.chest, self.head,\
            (0, NECK_H, 0), -upAxis, bkwdAxis, k, k, 80, 40)

        self.rightUpperLeg = self.addBody(R_HIP_POS, R_KNEE_POS,\
                                          0.11)

        k = -pi / 10
        l = -0.15 * pi
        m = 0.75 * pi
        n = 0.3 * pi
        self.rightHip = self.addUniversalJoint(self.pelvis,\
                                        self.rightUpperLeg,\
                                            R_HIP_POS, bkwdAxis,\
                                rightAxis, k, n,\
                                        l, m)
        self.leftUpperLeg = self.addBody(L_HIP_POS, L_KNEE_POS,\
                                         0.11)
        self.leftHip = self.addUniversalJoint(self.pelvis,\
                                self.leftUpperLeg, L_HIP_POS,\
                                fwdAxis, rightAxis, k, n,\
                                              l, m)

        self.rightLowerLeg = self.addBody(R_KNEE_POS, R_ANKLE_POS, 0.09)
        self.rightKnee = self.addHingeJoint(self.rightUpperLeg,
            self.rightLowerLeg, R_KNEE_POS, leftAxis, 0.0, pi * 0.75)
        self.leftLowerLeg = self.addBody(L_KNEE_POS, L_ANKLE_POS, 0.09)
        self.leftKnee = self.addHingeJoint(self.leftUpperLeg,
            self.leftLowerLeg, L_KNEE_POS, leftAxis, 0.0, pi * 0.75)

        self.rightFoot = self.addBody(R_HEEL_POS, R_TOES_POS, 0.09)
        self.rightAnkle = self.addHingeJoint(self.rightLowerLeg,
            self.rightFoot, R_ANKLE_POS, rightAxis, -0.1 * pi, 0.05 * pi)
        self.leftFoot = self.addBody(L_HEEL_POS, L_TOES_POS, 0.09)
        self.leftAnkle = self.addHingeJoint(self.leftLowerLeg,
            self.leftFoot, L_ANKLE_POS, rightAxis, -0.1 * pi, 0.05 * pi)

        self.rightUpperArm = self.addBody(R_SHOULDER_POS, R_ELBOW_POS, 0.08)
        self.rightShoulder = self.addBallJoint(self.chest, self.rightUpperArm,
            R_SHOULDER_POS, norm3((-1.0, -1.0, 4.0)), (0.0, 0.0, 1.0), pi * 0.5,
            pi * 0.25, 150.0, 100.0)
        self.leftUpperArm = self.addBody(L_SHOULDER_POS, L_ELBOW_POS, 0.08)
        self.leftShoulder = self.addBallJoint(self.chest, self.leftUpperArm,
            L_SHOULDER_POS, norm3((1.0, -1.0, 4.0)), (0.0, 0.0, 1.0), pi * 0.5,
            pi * 0.25, 150.0, 100.0)

        self.rightForeArm = self.addBody(R_ELBOW_POS, R_WRIST_POS, 0.075)
        self.rightElbow = self.addHingeJoint(self.rightUpperArm,
            self.rightForeArm, R_ELBOW_POS, downAxis, 0.0, 0.6 * pi)
        self.leftForeArm = self.addBody(L_ELBOW_POS, L_WRIST_POS, 0.075)
        self.leftElbow = self.addHingeJoint(self.leftUpperArm,
            self.leftForeArm, L_ELBOW_POS, upAxis, 0.0, 0.6 * pi)

        self.rightHand = self.addBody(R_WRIST_POS, R_FINGERS_POS, 0.075)
        self.rightWrist = self.addHingeJoint(self.rightForeArm,
            self.rightHand, R_WRIST_POS, fwdAxis, -0.1 * pi, 0.2 * pi)
        self.leftHand = self.addBody(L_WRIST_POS, L_FINGERS_POS, 0.075)
        self.leftWrist = self.addHingeJoint(self.leftForeArm,
            self.leftHand, L_WRIST_POS, bkwdAxis, -0.1 * pi, 0.2 * pi)

    def addBody(self, p1, p2, radius):
        """Adds a capsule body between joint positions p1 and p2 and with given
        radius to the ragdoll."""
        p1 = array(p1)
        p2 = array(p2)
        
        p1 = p1 + self.offset
        p2 = p2 + self.offset

        # cylinder length not including endcaps, make capsules overlap by half
        #   radius at joints
        cyllen = norm(p1 - p2) - radius

        body = Body(self.world)
        m = Mass()
        m.setCapsule(self.density, 3, radius, cyllen)
        body.setMass(m)

        # set parameters for drawing the body
        body.shape = "capsule"
        body.length = cyllen
        body.radius = radius

        # create a capsule geom for collision detection
        geom = GeomCCylinder(self.space, radius, cyllen)
        geom.setBody(body)

        # define body rotation automatically from body axis
        za = norm3(p2 - p1) #!

        if abs(dot3(za, (1, 0, 0))) < 0.7:
            xa = (1, 0, 0)

        else:
            xa = (0, 1, 0)
            
        ya = cross(za, xa)
        xa = norm3(cross(ya, za))
        ya = cross(za, xa)
        rot = (xa[0], ya[0], za[0],\
               xa[1], ya[1], za[1],\
               xa[2], ya[2], za[2])

        body.setPosition((p1 + p2) / 2)
        body.setRotation(rot)

        self.bodies.append(body)
        self.geoms.append(geom)
        
        self.totalMass += body.getMass().mass

        return body

    def addFixedJoint(self, body1, body2):
        joint = FixedJoint(self.world)
        joint.attach(body1, body2)
        joint.setFixed()

        joint.style = "fixed"
        self.joints.append(joint)

        return joint

    def addHingeJoint(self, body1, body2, anchor, axis,\
                      loStop = -Infinity,\
                      hiStop = Infinity):

        anchor += array(self.offset)

        joint = HingeJoint(self.world)
        joint.attach(body1, body2)
        joint.setAnchor(anchor)
        joint.setAxis(axis)
        joint.setParam(ParamLoStop, loStop)
        joint.setParam(ParamHiStop, hiStop)

        joint.style = "hinge"
        self.joints.append(joint)

        return joint

    def addUniversalJoint(self, body1, body2, anchor, axis1,\
                          axis2, loStop1 = -Infinity,\
                          hiStop1 = Infinity,\
                          loStop2 = -Infinity,\
                          hiStop2 = Infinity):

        anchor += array(self.offset)

        joint = UniversalJoint(self.world)
        joint.attach(body1, body2)
        joint.setAnchor(anchor)
        joint.setAxis1(axis1)
        joint.setAxis2(axis2)
        joint.setParam(ParamLoStop, loStop1)
        joint.setParam(ParamHiStop, hiStop1)
        joint.setParam(ParamLoStop2, loStop2)
        joint.setParam(ParamHiStop2, hiStop2)

        joint.style = "univ"
        self.joints.append(joint)

        return joint

    def addBallJoint(self, body1, body2, anchor, baseAxis,\
                     baseTwistUp, flexLimit = pi,\
                     twistLimit = pi, flexForce = 0,\
                     twistForce = 0):

        anchor += array(self.offset)

        # create the joint
        joint = BallJoint(self.world)
        joint.attach(body1, body2)
        joint.setAnchor(anchor)

        '''store the base orientation of the joint in the local coordinate system
        of the primary body (because baseAxis and baseTwistUp may not be
        orthogonal, the nearest vector to baseTwistUp but orthogonal to
        baseAxis is calculated and stored with the joint)'''
        joint.baseAxis = getBodyRelVec(body1, baseAxis)
        tempTwistUp = getBodyRelVec(body1, baseTwistUp)
        baseSide = norm3(cross(tempTwistUp, joint.baseAxis))
        joint.baseTwistUp = norm3(cross(joint.baseAxis, baseSide))

        '''store the base twist up vector (original version) in the local
        coordinate system of the secondary body'''
        joint.baseTwistUp2 = getBodyRelVec(body2, baseTwistUp)

        # store joint rotation limits and resistive force factors
        joint.flexLimit = flexLimit
        joint.twistLimit = twistLimit
        joint.flexForce = flexForce
        joint.twistForce = twistForce

        joint.style = "ball"
        self.joints.append(joint)

        return joint

    def update(self):
        for j in self.joints:
            if j.style == "ball":
                # determine base and current attached body axes
                baseAxis = rotate3(j.getBody(0).getRotation(),\
                                   j.baseAxis)
                currAxis = zaxis(j.getBody(1).getRotation())

                # get angular velocity of attached body relative to fixed body
                relAngVel = sub3(j.getBody(1).getAngularVel(),\
                                 j.getBody(0).getAngularVel())
                twistAngVel = project3(relAngVel, currAxis)
                flexAngVel = sub3(relAngVel, twistAngVel)

                # restrict limbs rotating too far from base axis
                angle = acosdot3(currAxis, baseAxis)

                if angle > j.flexLimit:
                    # add torque to push body back towards base axis
                    j.getBody(1).addTorque(\
                        mul3(norm3(cross(currAxis, baseAxis)),\
                        (angle - j.flexLimit) * j.flexForce))

                    # dampen flex to prevent bounceback
                    j.getBody(1).addTorque(\
                            mul3(flexAngVel, -j.flexForce / 100))

                '''determine the base twist up vector for the current attached
                body by applying the current joint flex to the fixed body's
                base twist up vector'''
                baseTwistUp = rotate3(j.getBody(0).getRotation(),\
                                      j.baseTwistUp)
                base2current = calcRotMatrix(\
                                norm3(cross(baseAxis, currAxis)),\
                                acosdot3(baseAxis, currAxis))
                projBaseTwistUp = rotate3(base2current,\
                                          baseTwistUp)

                # determine the current twist up vector from the attached body
                actualTwistUp = rotate3(\
                                j.getBody(1).getRotation(),\
                                j.baseTwistUp2)

                # restrict limbs twisting
                angle = acosdot3(actualTwistUp, projBaseTwistUp)

                if angle > j.twistLimit:
                    # add torque to rotate body back towards base angle
                    j.getBody(1).addTorque(\
                            mul3(norm3(\
                        cross(actualTwistUp, projBaseTwistUp)),
                        (angle - j.twistLimit) * j.twistForce))

                    # dampen twisting
                    j.getBody(1).addTorque(\
                        mul3(twistAngVel, -j.twistForce / 100))


def createCapsule(world, space, density, length, radius):
    """Creates a capsule body and corresponding geom.

    create capsule body (aligned along the z-axis so that it matches the
    GeomCCylinder created below, which is aligned along the z-axis by
    default)"""
    body = Body(world)
    M = Mass()  
    M.setCapsule(density, 3, radius, length)
    body.setMass(M)

    # set parameters for drawing the body
    body.shape = "capsule"
    body.length = length
    body.radius = radius

    # create a capsule geom for collision detection
    geom = GeomCCylinder(space, radius, length)
    geom.setBody(body)

    return body, geom

def near_callback(args, geom1, geom2):
    """Callback function for the collide() method.

    This function checks if the given geoms do collide and creates contact
    joints if they do."""

    if not areConnected(geom1.getBody(), geom2.getBody()):
        # check if the objects collide
        contacts = collide(geom1, geom2)

        # create contact joints
        world, contactgroup = args

        for c in contacts:
            c.setBounce(0.2)
            c.setMu(500) # 0-5 = very slippery, 50-500 = normal, 5000 = very sticky
            j = ContactJoint(world, contactgroup, c)
            j.attach(geom1.getBody(), geom2.getBody())

def prepare_GL():
    """Setup basic OpenGL rendering with smooth shading and a single light."""
    glClearColor(0.8, 0.8, 0.9, 0)
    glClear(16640)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    glShadeModel(GL_SMOOTH)

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective (45, width / height, 0.2, 20)

    glViewport(0, 0, width, height)

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    glLightfv(GL_LIGHT0, GL_POSITION, (0, 0, 1, 0))
    glLightfv(GL_LIGHT0, GL_DIFFUSE, (1, 1, 1, 1))
    glLightfv(GL_LIGHT0, GL_SPECULAR, (1, 1, 1, 1))
    glEnable(GL_LIGHT0)

    glEnable(GL_COLOR_MATERIAL)
    glColor3f(0.8, 0.8, 0.8)

    gluLookAt(1.5, 4, 3, 0.5, 1, 0, 0, 1, 0)

# polygon resolution for capsule bodies
CAPSULE_SLICES, CAPSULE_STACKS = 16, 12

def draw_body(body):
    """Draw an ODE body."""
    rot = makeOpenGLMatrix(body.getRotation(), body.getPosition())

    glPushMatrix()

    glMultMatrixf(rot)
    
    if body.shape == "capsule":
        cylHalfHeight = body.length / 2
        quadric = gluNewQuadric()
        glTranslatef(0, 0, -cylHalfHeight)
        glutSolidSphere(body.radius, CAPSULE_SLICES,\
                        CAPSULE_STACKS)
        gluCylinder(quadric, body.radius, body.radius,\
                    body.length, CAPSULE_SLICES, CAPSULE_STACKS)     
        glTranslatef(0, 0, body.length)
        glutSolidSphere(body.radius, CAPSULE_SLICES,\
                        CAPSULE_STACKS)

    glPopMatrix()

def onKey(c, x, y):
    """GLUT keyboard callback."""
    global SloMo, Paused
    
    try:
        # set simulation speed
        c = int(c)
        
        if -1 < c < 10: 
            SloMo = 4 * c + 1

    except:
        print('e: c no es un numero')

    try:
        # pause/unpause simulation
        c = c.decode("utf-8").lower()

        if c == 'p':
            Paused = not Paused
            
        # quit
        elif c == 'q':
            exit(0)

    except:
        print('e: c no es un string')



def onDraw():
    """GLUT render callback."""
    global t
    
    prepare_GL()

    for b in bodies + ragdoll.bodies:
        draw_body(b)

    glutSwapBuffers()

    #contador fps
    t1 = time()
    glutSetWindowTitle(str(round(1 / (t1 - t))))
    t = t1

def onIdle():
    """GLUT idle processing callback, performs ODE simulation step."""
    global Paused, lasttime, numiter, t1

    if not Paused:
        t = dt - time() + lasttime

        if t > 0:
            sleep(t)
        
        glutPostRedisplay()

        for i in range(stepsPerFrame):
            # Detect collisions and create contact joints
            space.collide((world, contactgroup), near_callback)

            # Simulation step (with slo motion)
            world.step(dt / stepsPerFrame / SloMo)

            numiter += 1

            # apply internal ragdoll forces
            ragdoll.update()

            # Remove all contact joints
            contactgroup.empty()

        lasttime = time()

t = 0

# initialize GLUT
glutInit()
glutInitDisplayMode(16) #18 o GLUT_DOUBLE = vsync

# create the program window
x, y, width, height = 0, 0, 1280, 720
glutInitWindowPosition(x, y)
glutInitWindowSize(width, height)
glutCreateWindow("")

# create an ODE world object
world = World()
world.setGravity((0, -10, 0))
world.setERP(0.1)
world.setCFM(1e-4)

# create an ODE space object
space = Space()

# create a plane geom to simulate a floor
floor = GeomPlane(space, (0, 1, 0), 0)

'''create a list to store any ODE bodies which are not part of the ragdoll (this
is needed to avoid Python garbage collecting these bodies)'''
bodies = []

'''create a joint group for the contact joints generated during collisions
between two bodies collide'''
contactgroup = JointGroup()

# set the initial simulation loop parameters
fps, stepsPerFrame, SloMo, Paused, lasttime, numiter = 100, 2, 1,\
                                                False, time(), 0
dt = 1 / fps
 
# create the ragdoll
ragdoll = RagDoll(world, space, 500, (0, 0.9, 0))
print("total mass is %.1f kg (%.1f lbs)" % (ragdoll.totalMass,\
    ragdoll.totalMass * 2.2))

# create an obstacle
obstacle, obsgeom = createCapsule(world, space, 1000, 0.05, 0.15)
pos = (uniform(-0.3, 0.3), 0.2, uniform(-0.15, 0.2))
#pos = (0.27396178783269359, 0.20000000000000001, 0.17531818795388002)
obstacle.setPosition(pos)
obstacle.setRotation(rightRot)
bodies.append(obstacle)
print("obstacle created at %s" % (str(pos)))

# set GLUT callbacks
glutKeyboardFunc(onKey)
glutDisplayFunc(onDraw)
glutIdleFunc(onIdle)

# enter the GLUT event loop
glutMainLoop()




