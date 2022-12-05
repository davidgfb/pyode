from time import time, sleep
from random import uniform
from math import pi, acos, cos, sin
from numpy import array, cross, zeros, ones
from numpy.linalg import norm

from OpenGL.GL import glClearColor, glClear, glEnable,\
     GL_DEPTH_TEST, GL_LIGHTING, GL_NORMALIZE, glShadeModel,\
     GL_SMOOTH, glMatrixMode, GL_PROJECTION, glLoadIdentity,\
     glViewport, GL_MODELVIEW, glLightfv, GL_LIGHT0,\
     GL_POSITION, GL_DIFFUSE, GL_SPECULAR, GL_COLOR_MATERIAL,\
     glColor3f, glPushMatrix, glMultMatrixf, glTranslatef,\
     glPopMatrix, glBegin, GL_POLYGON, glVertex3f, glEnd
from OpenGL.GLU import gluPerspective, gluLookAt, gluNewQuadric,\
     gluQuadricNormals, GLU_SMOOTH, gluQuadricTexture,\
     gluCylinder
from OpenGL.GLUT import glutInit, glutInitDisplayMode,\
     glutInitWindowPosition, glutInitWindowSize,\
     glutCreateWindow, glutKeyboardFunc,\
     glutDisplayFunc, glutIdleFunc, glutMainLoop,\
     glutSolidSphere, glutSwapBuffers, glutSetWindowTitle,\
     glutPostRedisplay
from ode import Body, Mass, GeomCCylinder, FixedJoint,\
     HingeJoint, ParamLoStop, ParamHiStop, UniversalJoint,\
     ParamLoStop2, ParamHiStop2, BallJoint, areConnected,\
     collide, ContactJoint, World, Space, GeomPlane, JointGroup

es_Primera_Vez, t_Cam_Lenta = True, 0 
# axes used to determine constrained joint rotations
rightAxis, upAxis, bkwdAxis = array((1, 0, 0)), array((0, 1, 0)),\
                              array((0, 0, 1))
leftAxis, downAxis, fwdAxis = -rightAxis, -upAxis, -bkwdAxis 
'''rotation directions are named by the third (z-axis) row of the 3x3 matrix,
because ODE capsules are oriented along the z-axis'''
rightRot = array((*-bkwdAxis,) + (*upAxis,) + (*rightAxis,))
UPPER_ARM_LEN, FORE_ARM_LEN, HAND_LEN, FOOT_LEN, HEEL_LEN = 3/10,\
                                        1/4, 13/100, 9/50, 1/20
'''HAND_LEN wrist to mid-fingers only
FOOT_LEN ankles to base of ball of foot only'''
BROW_H, MOUTH_H, NECK_H, SHOULDER_H, CHEST_H, HIP_H, KNEE_H,\
        ANKLE_H, SHOULDER_W, CHEST_W, LEG_W, PELVIS_W = 42/25,\
        153/100, 3/2, 137/100, 27/20, 43/50, 12/25, 2/25, 41/100,\
        9/25, 7/25, 1/4
'''CHEST_W actually wider, but we want narrower than shoulders (esp. with large radius)
LEG_W between middles of upper legs
PELVIS_W actually wider, but we want smaller than hip width'''

j = array((-1, 1, 1))
L_SHOULDER_POS = array((SHOULDER_W / 2, SHOULDER_H, 0))
R_SHOULDER_POS = L_SHOULDER_POS * j

k = UPPER_ARM_LEN * rightAxis
R_ELBOW_POS, L_ELBOW_POS = R_SHOULDER_POS - k, L_SHOULDER_POS + k

k = FORE_ARM_LEN * rightAxis
R_WRIST_POS, L_WRIST_POS = R_ELBOW_POS - k, L_ELBOW_POS + k

k = HAND_LEN * rightAxis
R_FINGERS_POS, L_FINGERS_POS = R_WRIST_POS - k, L_WRIST_POS + k

a = LEG_W / 2
L_HIP_POS = array((a, HIP_H, 0))
R_HIP_POS = L_HIP_POS * j

L_KNEE_POS = array((a, KNEE_H, 0))
R_KNEE_POS = L_KNEE_POS * j

L_ANKLE_POS = array((a, ANKLE_H, 0))
R_ANKLE_POS = L_ANKLE_POS * j

k = HEEL_LEN * bkwdAxis
R_HEEL_POS, L_HEEL_POS = R_ANKLE_POS - k, L_ANKLE_POS - k

k = FOOT_LEN * bkwdAxis
R_TOES_POS, L_TOES_POS = R_ANKLE_POS + k, L_ANKLE_POS + k

# polygon resolution for capsule bodies
cuerpos, nCuerpo, CAPSULE_SLICES, CAPSULE_STACKS, t = {}, 0, 16,\
                                                      12, 0

def a_Array(a, b):
    return (*map(array, (a, b)),)

def norm3(v):
    """Returns the unit length 3-vector parallel to 3-vector v."""
    v, l, normalizado = array(v), norm(v), zeros(3) 
    
    if not l == 0:
        normalizado = v / l 

    return normalizado

def project3(v, d):
    """Returns projection of 3-vector v onto unit 3-vector d."""    
    proyectado = zeros(3)
            
    return norm3(v) @ d * array(v)

def acosdot3(a, b):
    """Returns the angle between unit 3-vectors a and b."""
    (a, b), x = a_Array(a, b), a @ b 
    angulo = acos(x)
    
    if x < -1:
        angulo = pi

    elif x > 1:
        angulo = 0        

    return angulo

def traspuesta(v):
    return array(v).reshape(3, 3).transpose()

def rotate3(m, v):
    """Returns the rotation of 3-vector v by 3x3 (row major) matrix m."""
    return array(v) @ traspuesta(m)       

def getBodyRelVec(b, v):
    """Returns the 3-vector v transformed into the local coordinate system of ODE
    body b."""
    return rotate3(traspuesta(b.getRotation()).reshape(9), v)

class Ragdoll():
    def __init__(self, world, space, density, offset = zeros(3)):
        """Creates a ragdoll of standard size at the given offset."""
        self.world, self.space, self.density, self.bodies,\
                    self.geoms, self.joints, self.totalMass,\
                    self.offset, k = world, space, density, [],\
                    [], [], 0, offset, (CHEST_W / 2, CHEST_H, 0)

        '''self.chest, self.belly, self.pelvis, self.head,\
                    self.rightUpperLeg, self.leftUpperLeg,\
                    self.rightLowerLeg, self.leftLowerLeg,\
                    self.rightFoot, self.leftFoot,\
                    self.rightUpperArm, self.leftUpperArm,\
                    self.rightForeArm, self.leftForeArm,\
                    self.rightHand, self.leftHand =\
        (*(self.addBody(*a) for a in\
        ((k * j, k, 13/100, 'chest'),\
        ((CHEST_H - 1 / 10) * upAxis, (HIP_H + 1 / 10) * upAxis,\
        1 / 8, 'belly'), ((-PELVIS_W / 2, HIP_H, 0),\
        (PELVIS_W / 2, HIP_H, 0), 1 / 8, 'pelvis'),\
        (BROW_H * upAxis, MOUTH_H * upAxis, 0.11, 'head'),\
        (R_HIP_POS, R_KNEE_POS, 11/100, 'rightUpperLeg'),\
        (L_HIP_POS, L_KNEE_POS, 11/100, 'leftUpperLeg'),\
        (R_KNEE_POS, R_ANKLE_POS, 9/100, 'rightLowerLeg'),\
        (L_KNEE_POS, L_ANKLE_POS, 9/100, 'leftLowerLeg'),\
        (R_HEEL_POS, R_TOES_POS, 9/100, 'rightFoot'),
        (L_HEEL_POS, L_TOES_POS, 9/100, 'leftFoot'),\
        (R_SHOULDER_POS, R_ELBOW_POS, 2/25, 'rightUpperArm'),\
        (L_SHOULDER_POS, L_ELBOW_POS, 2/25, 'leftUpperArm'),\
        (R_ELBOW_POS, R_WRIST_POS, 3/40, 'rightForeArm'),\
        (L_ELBOW_POS, L_WRIST_POS, 3/40, 'leftForeArm'),\
        (R_WRIST_POS, R_FINGERS_POS, 3/40, 'rightHand'),\
        (L_WRIST_POS, L_FINGERS_POS, 3/40, 'leftHand'))),)'''

        Ragdoll = {}
        miembros = {'Chest' : (k * j, k, 13/100, 'chest'),
                    'Belly' : ((CHEST_H - 1 / 10) * upAxis,\
                               (HIP_H + 1 / 10) * upAxis, 1 / 8,\
                               'belly'),
                    'Pelvis' : ((-PELVIS_W / 2, HIP_H, 0),\
                                (PELVIS_W / 2, HIP_H, 0), 1 / 8,\
                                'pelvis'),
                    'Head' : (BROW_H * upAxis, MOUTH_H * upAxis,\
                              0.11, 'head')}

        for nom_M in miembros:
            Ragdoll[nom_M] = self.addBody(*miembros[nom_M])

        miembros = {'UpperLegs' : {False : (R_HIP_POS, R_KNEE_POS, 11/100, 'rightUpperLeg'),
                                   True : (L_HIP_POS, L_KNEE_POS, 11/100, 'leftUpperLeg')},
                    'LowerLegs' : {False : (R_KNEE_POS, R_ANKLE_POS, 9/100, 'rightLowerLeg'),
                                   True : (L_KNEE_POS, L_ANKLE_POS, 9/100, 'leftLowerLeg')},
                    'Feet' : {False : (R_HEEL_POS, R_TOES_POS, 9/100, 'rightFoot'),
                              True : (L_HEEL_POS, L_TOES_POS, 9/100, 'leftFoot')},
                    'UpperArms' : {False : (R_SHOULDER_POS, R_ELBOW_POS, 2/25, 'rightUpperArm'),
                                   True : (L_SHOULDER_POS, L_ELBOW_POS, 2/25, 'leftUpperArm')},
                    'ForeArms' : {False : (R_ELBOW_POS, R_WRIST_POS, 3/40, 'rightForeArm'),
                                  True : (L_ELBOW_POS, L_WRIST_POS, 3/40, 'leftForeArm')},
                    'Hands' : {False : (R_WRIST_POS, R_FINGERS_POS, 3/40, 'rightHand'),
                               True : (L_WRIST_POS, L_FINGERS_POS, 3/40, 'leftHand')}}

        for nom_Ms in miembros:
            for es_Izdo in miembros[nom_Ms]:
                lado = 'Right'
                
                if es_Izdo:
                    lado = 'Left'

                Ragdoll[nom_Ms + lado] = self.addBody(*miembros[nom_Ms][es_Izdo])

        self.chest, self.belly, self.pelvis, self.head,\
                    self.rightUpperLeg, self.leftUpperLeg,\
                    self.rightLowerLeg, self.leftLowerLeg,\
                    self.rightFoot, self.leftFoot,\
                    self.rightUpperArm, self.leftUpperArm,\
                    self.rightForeArm, self.leftForeArm,\
                    self.rightHand, self.leftHand =\
                    (*(Ragdoll[nom_M] for nom_M in Ragdoll),)

        self.midSpine, self.lowSpine =\
        (*(self.addFixedJoint(*a) for a in ((self.chest,
                                             self.belly),\
                                            (self.belly,
                                             self.pelvis))),)                                   
        k, l, m, n, o, p, q, r, s = -pi / 10, -3/20 * pi,\
                                    3/4 * pi, 3/10 * pi,\
                                    pi / 20, pi / 2, pi / 4,\
                                    3/50 * pi, pi / 5
        self.neck = self.addBallJoint(self.chest, self.head,\
                    NECK_H * upAxis, -upAxis, bkwdAxis, q, q, 80,\
                    40)      
        self.rightHip, self.leftHip =\
        (self.addUniversalJoint(*a) for a in ((self.pelvis,\
        self.rightUpperLeg, R_HIP_POS, bkwdAxis, rightAxis, k,\
        n, l, m), (self.pelvis, self.leftUpperLeg, L_HIP_POS,\
        fwdAxis, rightAxis, k, n, l, m)))  

        self.rightKnee, self.leftKnee, self.rightAnkle,\
        self.leftAnkle =\
        (*(self.addHingeJoint(*a) for a in ((self.rightUpperLeg,\
        self.rightLowerLeg, R_KNEE_POS, leftAxis, 0, m),\
        (self.leftUpperLeg, self.leftLowerLeg, L_KNEE_POS,\
        leftAxis, 0, m), (self.rightLowerLeg, self.rightFoot,\
        R_ANKLE_POS, rightAxis, k, o), (self.leftLowerLeg,\
        self.leftFoot, L_ANKLE_POS, rightAxis, k, o))),)
        self.rightShoulder, self.leftShoulder =\
        (*(self.addBallJoint(*a) for a in ((self.chest,\
        self.rightUpperArm, R_SHOULDER_POS, norm3((-1, -1, 4)),\
        bkwdAxis, p, q, 150, 100), (self.chest,\
        self.leftUpperArm, L_SHOULDER_POS, norm3((1, -1, 4)),\
        bkwdAxis, p, q, 150, 100))),)                    
        self.rightElbow, self.leftElbow, self.rightWrist,\
        self.leftWrist =\
        (*(self.addHingeJoint(*a) for a in ((self.rightUpperArm,\
        self.rightForeArm, R_ELBOW_POS, downAxis, 0, r),\
        (self.leftUpperArm, self.leftForeArm, L_ELBOW_POS, upAxis,\
        0, r), (self.rightForeArm, self.rightHand, R_WRIST_POS,\
        fwdAxis, k, s), (self.leftForeArm, self.leftHand,\
        L_WRIST_POS, bkwdAxis, k, s))),)
    
    def addBody(self, p1, p2, radius, name):
        """Adds a capsule body between joint positions p1 and p2 and with given
        radius to the ragdoll."""
        global cuerpos, nCuerpo

        (p1, p2), body = (p1, p2) + self.offset, Body(self.world)

        # cylinder length not including endcaps, make capsules overlap by half
        #   radius at joints
        cyllen, cuerpos[name], m = norm(p1 - p2) - radius, body,\
                                   Mass()  
        
        m.setCapsule(self.density, 3, radius, cyllen)
        body.setMass(m)
        # set parameters for drawing the body
        # create a capsule geom for collision detection
        # define body rotation automatically from body axis
        body.shape, body.length, body.radius, geom, za =\
        "capsule", cyllen, radius, GeomCCylinder(self.space,\
        radius, cyllen), norm3(p2 - p1)

        geom.setBody(body)

        if abs(za @ rightAxis) < 0.7:
            xa = rightAxis

        else:
            xa = upAxis
            
        ya = cross(za, xa) 
        rot = array((norm3(cross(ya, za)), ya, za)). transpose().\
              reshape(9)
        
        body.setPosition((p1 + p2) / 2)
        body.setRotation(rot)
        (*(a.append(b) for a, b in ((self.bodies, body),\
                                    (self.geoms, geom))),) #?     
        self.totalMass += body.getMass().mass

        return body

    def get_Junta(self, j_Style, joint):
        joint.style = j_Style
        self.joints.append(joint)

        return joint

    def addFixedJoint(self, body1, body2):
        joint = FixedJoint(self.world)

        joint.attach(body1, body2)
        joint.setFixed()

        return self.get_Junta("fixed", joint)

    inf = float('inf')

    def junta(self, junta, cuerpos, ancla):
        junta.attach(*cuerpos) # joint
        junta.setAnchor(ancla)

    def addHingeJoint(self, body1, body2, anchor, axis,\
                      loStop = -inf, hiStop = inf):
        anchor += array(self.offset)
        joint = HingeJoint(self.world)

        self.junta(joint, (body1, body2), anchor)
        joint.setAxis(axis)
        (*(joint.setParam(a, b) for a, b in\
              ((ParamLoStop, loStop), (ParamHiStop, hiStop))),)
        
        return self.get_Junta("hinge", joint)

    def addUniversalJoint(self, body1, body2, anchor, axis1,\
                          axis2, loStop1 = -inf, hiStop1 = inf,\
                          loStop2 = -inf, hiStop2 = inf):
        anchor += array(self.offset)

        joint = UniversalJoint(self.world)

        self.junta(joint, (body1, body2), anchor)
        joint.setAxis1(axis1)
        joint.setAxis2(axis2)
        (*(joint.setParam(a, b) for a, b in\
            ((ParamLoStop, loStop1), (ParamHiStop, hiStop1),\
             (ParamLoStop2, loStop2), (ParamHiStop2, hiStop2))),)

        return self.get_Junta("univ", joint)

    def addBallJoint(self, body1, body2, anchor, baseAxis,\
                     baseTwistUp, flexLimit = pi,\
                     twistLimit = pi, flexForce = 0,\
                     twistForce = 0):
        anchor += array(self.offset)

        # create the joint
        joint = BallJoint(self.world)

        self.junta(joint, (body1, body2), anchor)
        '''store the base orientation of the joint in the local coordinate system
        of the primary body (because baseAxis and baseTwistUp may not be
        orthogonal, the nearest vector to baseTwistUp but orthogonal to
        baseAxis is calculated and stored with the joint)'''
        joint.baseAxis, tempTwistUp =\
                        tuple(getBodyRelVec(body1, a) for a in\
                             (baseAxis, baseTwistUp))
                        
        baseSide = norm3(cross(tempTwistUp, joint.baseAxis))
        joint.baseTwistUp = norm3(cross(joint.baseAxis, baseSide)) #interdep

        '''store the base twist up vector (original version) in the local
        coordinate system of the secondary body'''
        joint.baseTwistUp2 = getBodyRelVec(body2, baseTwistUp)

        # store joint rotation limits and resistive force factors
        joint.flexLimit, joint.twistLimit, joint.flexForce, joint.twistForce =\
                         flexLimit, twistLimit, flexForce, twistForce

        return self.get_Junta("ball", joint)

    def update(self):
        for j in self.joints:
            if j.style == "ball":
                j_B0, j_B1 = j.getBody(0), j.getBody(1)
                # determine base and current attached body axes
                baseAxis = rotate3(j_B0.getRotation(),\
                                   j.baseAxis)
                currAxis = traspuesta(j_B1.getRotation())[2]

                # get angular velocity of attached body relative to fixed body
                relAngVel = array(j_B1.getAngularVel()) -\
                            array(j_B0.getAngularVel())
                twistAngVel = project3(relAngVel, currAxis)
                flexAngVel = relAngVel - twistAngVel

                # restrict limbs rotating too far from base axis
                angle = acosdot3(currAxis, baseAxis)

                if angle > j.flexLimit:
                    # add torque to push body back towards base axis
                    (*map(j_B1.addTorque,\
                        ((norm3(cross(currAxis, baseAxis)) *\
                         (angle - j.flexLimit) * j.flexForce),\
                         (-flexAngVel / 100 * j.flexForce))),) #dampen flex to prevent bounceback

                '''determine the base twist up vector for the current attached
                body by applying the current joint flex to the fixed body's
                base twist up vector'''
                baseTwistUp = rotate3(j_B0.getRotation(),\
                                      j.baseTwistUp)

                """Returns the row-major 3x3 rotation matrix defining a rotation around axis by
                angle. Esto conviene q lo haga opengl"""
                axis, angle = norm3(cross(baseAxis, currAxis)),\
                              acosdot3(baseAxis, currAxis)
                cosTheta, sinTheta = cos(angle), sin(angle)
                t = 1 - cosTheta 
                a0, a1, a2 = axis
                sT_A0, sT_A1, sT_A2, t_A0_A1, t_A0_A2, t_A1_A2 =\
                sinTheta * a0, sinTheta * a1, sinTheta * a2,\
                t * a0 * a1, t * a0 * a2, t * a1 * a2

                base2current = (t * a0 ** 2 + cosTheta,
                                t_A0_A1 - sT_A2, t_A0_A2 + sT_A1,

                                t_A0_A1 + sT_A2,
                                t * a1 ** 2 + cosTheta,
                                t_A1_A2 - sT_A0,

                                t_A0_A2 - sT_A1, t_A1_A2 + sT_A0,
                                t * a2 ** 2 + cosTheta)

                projBaseTwistUp = rotate3(base2current,\
                                          baseTwistUp)

                # determine the current twist up vector from the attached body
                actualTwistUp = rotate3(j_B1.getRotation(),\
                                j.baseTwistUp2)

                # restrict limbs twisting
                angle = acosdot3(actualTwistUp, projBaseTwistUp)

                if angle > j.twistLimit:
                    # add torque to rotate body back towards base angle
                    (*(map(j_B1.addTorque,\
                    ((norm3(cross(actualTwistUp, projBaseTwistUp)) *\
                     (angle - j.twistLimit) * j.twistForce),\
                     (-twistAngVel / 100 * j.twistForce)))),) # dampen twisting

def createCapsule(world, space, density, length, radius):
    """Creates a capsule body and corresponding geom.
    create capsule body (aligned along the z-axis so that it matches the
    GeomCCylinder created below, which is aligned along the z-axis by
    default)"""
    body = Body(world)
    M = Mass()  
    M.setCapsule(density, 3, radius, length)
    body.setMass(M)

    # set parameters for drawing the body # create a capsule geom for collision detection
    body.shape, body.length, body.radius, geom =\
    "capsule", length, radius,GeomCCylinder(space, radius, length)
    geom.setBody(body)

    return body, geom

def near_callback(args, geom1, geom2):
    """Callback function for the collide() method.
    This function checks if the given geoms do collide and creates contact
    joints if they do."""
    global SloMo, es_Primera_Vez, t_Cam_Lenta

    if not es_Primera_Vez and time() - t_Cam_Lenta > 2:
        SloMo = 1

    if not areConnected(geom1.getBody(), geom2.getBody()):
        # check if the objects collide
        contacts = collide(geom1, geom2)

        # create contact joints
        world, contactgroup = args

        for c in contacts:
            if es_Primera_Vez and type(geom1) == type(geom2):# else False
                SloMo, t_Cam_Lenta, es_Primera_Vez = 5, time(),\
                                                     False
                          
            c.setBounce(1 / 5)
            c.setMu(500) # 0-5 = very slippery, 50-500 = normal, 5000 = very sticky
            ContactJoint(world, contactgroup, c).\
                         attach(geom1.getBody(), geom2.getBody()) #uso global?

def prepare_GL():
    """Setup basic OpenGL rendering with smooth shading and a single light.""" 
    #glClearColor(*(0.8, 0.8, 0.9, 0)) 
    glClear(16640)
    (*map(glEnable, (GL_DEPTH_TEST, GL_LIGHTING)),)
    glShadeModel(GL_SMOOTH)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, width / height, 0.2, 20)
    glViewport(0, 0, width, height)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()   
    (*map(glEnable, (GL_LIGHT0, GL_COLOR_MATERIAL)),)
    glColor3f(*(0.8 * ones(3)))
    gluLookAt(*((2, 4, 3) + cuerpos['pelvis'].getPosition() +\
             (*(upAxis),)))

pos_Pies = {}

def draw_body(body):
    """Draw an ODE body."""
    global pos_Pies
    
    glPushMatrix()

    """Returns an OpenGL compatible (column-major, 4x4 homogeneous) transformation
    matrix from ODE compatible (row-major, 3x3) rotation matrix r and position
    vector p."""
    r = traspuesta(body.getRotation()).tolist() 

    for e_R in r:
        e_R += (0,)    
    
    glMultMatrixf(((*(array(r).reshape(12).tolist()),) +\
                   (*(body.getPosition()) + (1,),)))
    
    if body.shape == "capsule": 
        cylHalfHeight = body.length / 2
        quadric = gluNewQuadric()
        
        glTranslatef(*(-cylHalfHeight * bkwdAxis))
        glutSolidSphere(body.radius, CAPSULE_SLICES,\
                        CAPSULE_STACKS)
        gluCylinder(quadric, body.radius, body.radius,\
                    body.length, CAPSULE_SLICES, CAPSULE_STACKS)     
        glTranslatef(*(body.length * bkwdAxis)) #no need pp
        glutSolidSphere(body.radius, CAPSULE_SLICES,\
                        CAPSULE_STACKS)

        '''es_Pie_Izdo, es_Pie_Dcho = body == cuerpos['leftFoot'],\
                                   body == cuerpos['rightFoot']
        if es_Pie_Izdo or es_Pie_Dcho:
            if es_Pie_Izdo:
                pos_Pies['pie_Izdo'] = 'a'
    
            if es_Pie_Dcho:
                pos_Pies['pie_Dcho'] = 'b'

        if len(pos_Pies) > 1:
            pos_Pies = {}'''
                
            #glPushMatrix()

        '''if n_Pies > 0:
                #guarda pos
                glBegin(GL_POLYGON)
                glVertex3f(1,0,0)
                glVertex3f(0,1,0)
                glVertex3f(0,0,0)
                glEnd()
                n_Pies = 0

            else:
                n_Pies += 1'''

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
            exit()

    except:
        print('e: c no es un string')

def onDraw():
    """GLUT render callback."""
    global t
    
    prepare_GL()
    (*(draw_body(b) for b in bodies + ragdoll.bodies),)
    glutSwapBuffers()

    #contador fps
    t1 = time()
    glutSetWindowTitle(str(round(1 / (t1 - t))))
    t = t1

def onIdle():
    """GLUT idle processing callback, performs ODE simulation step."""
    global Paused, lasttime, t1

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
            # apply internal ragdoll forces
            ragdoll.update()
            # Remove all contact joints
            contactgroup.empty()

        lasttime = time()

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

world.setGravity(-10 * upAxis)
world.setERP(0.1)
world.setCFM(1e-4)

# create an ODE space object
space = Space()
# create a plane geom to simulate a floor
'''create a list to store any ODE bodies which are not part of the ragdoll (this
is needed to avoid Python garbage collecting these bodies)'''
'''create a joint group for the contact joints generated during collisions
between two bodies collide'''
# set the initial simulation loop parameters
floor, bodies, contactgroup, fps, stepsPerFrame, SloMo, Paused,\
lasttime = GeomPlane(space, upAxis, 0), [],\
JointGroup(), 100, 2, 1, False, time()
# create the ragdoll
dt, ragdoll, pos = 1 / fps, Ragdoll(world, space, 500,\
9/10 * upAxis), (uniform(-3/10, 3/10), 1 / 5, uniform(-3/20, 1 / 5))
# create an obstacle
obstacle, obsgeom = createCapsule(world, space, 1000, 1 / 20,\
                                  3/20)
#pos = (0.27396178783269359, 0.20000000000000001, 0.17531818795388002)

obstacle.setPosition(pos)
obstacle.setRotation(rightRot)
bodies.append(obstacle)
# set GLUT callbacks
glutKeyboardFunc(onKey)
glutDisplayFunc(onDraw)
glutIdleFunc(onIdle)
print("obstacle created at", str(pos), "\ntotal mass is %.1f kg (%.1f lbs)" %\
      (ragdoll.totalMass, ragdoll.totalMass * 2.2))
# enter the GLUT event loop
glutMainLoop()






