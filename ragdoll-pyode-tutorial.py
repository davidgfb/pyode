# Copyright (c) 2007, Matt Heinzen
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY MATT HEINZEN ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL MATT HEINZEN BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import sys, os, random, time
from math import *
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import ode

def sign(x):
	"""Returns 1.0 if x is positive, -1.0 if x is negative or zero."""
	if x > 0.0: return 1.0
	else: return -1.0

def len3(v):
	"""Returns the length of 3-vector v."""
	return sqrt(v[0]**2 + v[1]**2 + v[2]**2)

def neg3(v):
	"""Returns the negation of 3-vector v."""
	return (-v[0], -v[1], -v[2])

def add3(a, b):
	"""Returns the sum of 3-vectors a and b."""
	return (a[0] + b[0], a[1] + b[1], a[2] + b[2])

def sub3(a, b):
	"""Returns the difference between 3-vectors a and b."""
	return (a[0] - b[0], a[1] - b[1], a[2] - b[2])

def mul3(v, s):
	"""Returns 3-vector v multiplied by scalar s."""
	return (v[0] * s, v[1] * s, v[2] * s)

def div3(v, s):
	"""Returns 3-vector v divided by scalar s."""
	return (v[0] / s, v[1] / s, v[2] / s)

def dist3(a, b):
	"""Returns the distance between point 3-vectors a and b."""
	return len3(sub3(a, b))

def norm3(v):
	"""Returns the unit length 3-vector parallel to 3-vector v."""
	l = len3(v)
	if (l > 0.0): return (v[0] / l, v[1] / l, v[2] / l)
	else: return (0.0, 0.0, 0.0)

def dot3(a, b):
	"""Returns the dot product of 3-vectors a and b."""
	return (a[0] * b[0] + a[1] * b[1] + a[2] * b[2])

def cross(a, b):
	"""Returns the cross product of 3-vectors a and b."""
	return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
		a[0] * b[1] - a[1] * b[0])

def project3(v, d):
	"""Returns projection of 3-vector v onto unit 3-vector d."""
	return mul3(v, dot3(norm3(v), d))

def acosdot3(a, b):
	"""Returns the angle between unit 3-vectors a and b."""
	x = dot3(a, b)
	if x < -1.0: return pi
	elif x > 1.0: return 0.0
	else: return acos(x)

def rotate3(m, v):
	"""Returns the rotation of 3-vector v by 3x3 (row major) matrix m."""
	return (v[0] * m[0] + v[1] * m[1] + v[2] * m[2],
		v[0] * m[3] + v[1] * m[4] + v[2] * m[5],
		v[0] * m[6] + v[1] * m[7] + v[2] * m[8])

def invert3x3(m):
	"""Returns the inversion (transpose) of 3x3 rotation matrix m."""
	return (m[0], m[3], m[6], m[1], m[4], m[7], m[2], m[5], m[8])

def zaxis(m):
	"""Returns the z-axis vector from 3x3 (row major) rotation matrix m."""
	return (m[2], m[5], m[8])

def calcRotMatrix(axis, angle):
	"""
	Returns the row-major 3x3 rotation matrix defining a rotation around axis by
	angle.
	"""
	cosTheta = cos(angle)
	sinTheta = sin(angle)
	t = 1.0 - cosTheta
	return (
		t * axis[0]**2 + cosTheta,
		t * axis[0] * axis[1] - sinTheta * axis[2],
		t * axis[0] * axis[2] + sinTheta * axis[1],
		t * axis[0] * axis[1] + sinTheta * axis[2],
		t * axis[1]**2 + cosTheta,
		t * axis[1] * axis[2] - sinTheta * axis[0],
		t * axis[0] * axis[2] - sinTheta * axis[1],
		t * axis[1] * axis[2] + sinTheta * axis[0],
		t * axis[2]**2 + cosTheta)

def makeOpenGLMatrix(r, p):
	"""
	Returns an OpenGL compatible (column-major, 4x4 homogeneous) transformation
	matrix from ODE compatible (row-major, 3x3) rotation matrix r and position
	vector p.
	"""
	return (
		r[0], r[3], r[6], 0.0,
		r[1], r[4], r[7], 0.0,
		r[2], r[5], r[8], 0.0,
		p[0], p[1], p[2], 1.0)

def getBodyRelVec(b, v):
	"""
	Returns the 3-vector v transformed into the local coordinate system of ODE
	body b.
	"""
	return rotate3(invert3x3(b.getRotation()), v)


# rotation directions are named by the third (z-axis) row of the 3x3 matrix,
#   because ODE capsules are oriented along the z-axis
rightRot = (0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0)
leftRot = (0.0, 0.0, 1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0)
upRot = (1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0)
downRot = (1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0)
bkwdRot = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

# axes used to determine constrained joint rotations
rightAxis = (1.0, 0.0, 0.0)
leftAxis = (-1.0, 0.0, 0.0)
upAxis = (0.0, 1.0, 0.0)
downAxis = (0.0, -1.0, 0.0)
bkwdAxis = (0.0, 0.0, 1.0)
fwdAxis = (0.0, 0.0, -1.0)

UPPER_ARM_LEN = 0.30
FORE_ARM_LEN = 0.25
HAND_LEN = 0.13 # wrist to mid-fingers only
FOOT_LEN = 0.18 # ankles to base of ball of foot only
HEEL_LEN = 0.05

BROW_H = 1.68
MOUTH_H = 1.53
NECK_H = 1.50
SHOULDER_H = 1.37
CHEST_H = 1.35
HIP_H = 0.86
KNEE_H = 0.48
ANKLE_H = 0.08

SHOULDER_W = 0.41
CHEST_W = 0.36 # actually wider, but we want narrower than shoulders (esp. with large radius)
LEG_W = 0.28 # between middles of upper legs
PELVIS_W = 0.25 # actually wider, but we want smaller than hip width

R_SHOULDER_POS = (-SHOULDER_W * 0.5, SHOULDER_H, 0.0)
L_SHOULDER_POS = (SHOULDER_W * 0.5, SHOULDER_H, 0.0)
R_ELBOW_POS = sub3(R_SHOULDER_POS, (UPPER_ARM_LEN, 0.0, 0.0))
L_ELBOW_POS = add3(L_SHOULDER_POS, (UPPER_ARM_LEN, 0.0, 0.0))
R_WRIST_POS = sub3(R_ELBOW_POS, (FORE_ARM_LEN, 0.0, 0.0))
L_WRIST_POS = add3(L_ELBOW_POS, (FORE_ARM_LEN, 0.0, 0.0))
R_FINGERS_POS = sub3(R_WRIST_POS, (HAND_LEN, 0.0, 0.0))
L_FINGERS_POS = add3(L_WRIST_POS, (HAND_LEN, 0.0, 0.0))

R_HIP_POS = (-LEG_W * 0.5, HIP_H, 0.0)
L_HIP_POS = (LEG_W * 0.5, HIP_H, 0.0)
R_KNEE_POS = (-LEG_W * 0.5, KNEE_H, 0.0)
L_KNEE_POS = (LEG_W * 0.5, KNEE_H, 0.0)
R_ANKLE_POS = (-LEG_W * 0.5, ANKLE_H, 0.0)
L_ANKLE_POS = (LEG_W * 0.5, ANKLE_H, 0.0)
R_HEEL_POS = sub3(R_ANKLE_POS, (0.0, 0.0, HEEL_LEN))
L_HEEL_POS = sub3(L_ANKLE_POS, (0.0, 0.0, HEEL_LEN))
R_TOES_POS = add3(R_ANKLE_POS, (0.0, 0.0, FOOT_LEN))
L_TOES_POS = add3(L_ANKLE_POS, (0.0, 0.0, FOOT_LEN))


class RagDoll():
	def __init__(self, world, space, density, offset = (0.0, 0.0, 0.0)):
		"""Creates a ragdoll of standard size at the given offset."""

		self.world = world
		self.space = space
		self.density = density
		self.bodies = []
		self.geoms = []
		self.joints = []
		self.totalMass = 0.0

		self.offset = offset

		self.chest = self.addBody((-CHEST_W * 0.5, CHEST_H, 0.0),
			(CHEST_W * 0.5, CHEST_H, 0.0), 0.13)
		self.belly = self.addBody((0.0, CHEST_H - 0.1, 0.0),
			(0.0, HIP_H + 0.1, 0.0), 0.125)
		self.midSpine = self.addFixedJoint(self.chest, self.belly)
		self.pelvis = self.addBody((-PELVIS_W * 0.5, HIP_H, 0.0),
			(PELVIS_W * 0.5, HIP_H, 0.0), 0.125)
		self.lowSpine = self.addFixedJoint(self.belly, self.pelvis)

		self.head = self.addBody((0.0, BROW_H, 0.0), (0.0, MOUTH_H, 0.0), 0.11)
		self.neck = self.addBallJoint(self.chest, self.head,
			(0.0, NECK_H, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0), pi * 0.25,
			pi * 0.25, 80.0, 40.0)

		self.rightUpperLeg = self.addBody(R_HIP_POS, R_KNEE_POS, 0.11)
		self.rightHip = self.addUniversalJoint(self.pelvis, self.rightUpperLeg,
			R_HIP_POS, bkwdAxis, rightAxis, -0.1 * pi, 0.3 * pi, -0.15 * pi,
			0.75 * pi)
		self.leftUpperLeg = self.addBody(L_HIP_POS, L_KNEE_POS, 0.11)
		self.leftHip = self.addUniversalJoint(self.pelvis, self.leftUpperLeg,
			L_HIP_POS, fwdAxis, rightAxis, -0.1 * pi, 0.3 * pi, -0.15 * pi,
			0.75 * pi)

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
		"""
		Adds a capsule body between joint positions p1 and p2 and with given
		radius to the ragdoll.
		"""

		p1 = add3(p1, self.offset)
		p2 = add3(p2, self.offset)

		# cylinder length not including endcaps, make capsules overlap by half
		#   radius at joints
		cyllen = dist3(p1, p2) - radius

		body = ode.Body(self.world)
		m = ode.Mass()
		m.setCapsule(self.density, 3, radius, cyllen)
		body.setMass(m)

		# set parameters for drawing the body
		body.shape = "capsule"
		body.length = cyllen
		body.radius = radius

		# create a capsule geom for collision detection
		geom = ode.GeomCCylinder(self.space, radius, cyllen)
		geom.setBody(body)

		# define body rotation automatically from body axis
		za = norm3(sub3(p2, p1))
		if (abs(dot3(za, (1.0, 0.0, 0.0))) < 0.7): xa = (1.0, 0.0, 0.0)
		else: xa = (0.0, 1.0, 0.0)
		ya = cross(za, xa)
		xa = norm3(cross(ya, za))
		ya = cross(za, xa)
		rot = (xa[0], ya[0], za[0], xa[1], ya[1], za[1], xa[2], ya[2], za[2])

		body.setPosition(mul3(add3(p1, p2), 0.5))
		body.setRotation(rot)

		self.bodies.append(body)
		self.geoms.append(geom)
		
		self.totalMass += body.getMass().mass

		return body

	def addFixedJoint(self, body1, body2):
		joint = ode.FixedJoint(self.world)
		joint.attach(body1, body2)
		joint.setFixed()

		joint.style = "fixed"
		self.joints.append(joint)

		return joint

	def addHingeJoint(self, body1, body2, anchor, axis, loStop = -ode.Infinity,
		hiStop = ode.Infinity):

		anchor = add3(anchor, self.offset)

		joint = ode.HingeJoint(self.world)
		joint.attach(body1, body2)
		joint.setAnchor(anchor)
		joint.setAxis(axis)
		joint.setParam(ode.ParamLoStop, loStop)
		joint.setParam(ode.ParamHiStop, hiStop)

		joint.style = "hinge"
		self.joints.append(joint)

		return joint

	def addUniversalJoint(self, body1, body2, anchor, axis1, axis2,
		loStop1 = -ode.Infinity, hiStop1 = ode.Infinity,
		loStop2 = -ode.Infinity, hiStop2 = ode.Infinity):

		anchor = add3(anchor, self.offset)

		joint = ode.UniversalJoint(self.world)
		joint.attach(body1, body2)
		joint.setAnchor(anchor)
		joint.setAxis1(axis1)
		joint.setAxis2(axis2)
		joint.setParam(ode.ParamLoStop, loStop1)
		joint.setParam(ode.ParamHiStop, hiStop1)
		joint.setParam(ode.ParamLoStop2, loStop2)
		joint.setParam(ode.ParamHiStop2, hiStop2)

		joint.style = "univ"
		self.joints.append(joint)

		return joint

	def addBallJoint(self, body1, body2, anchor, baseAxis, baseTwistUp,
		flexLimit = pi, twistLimit = pi, flexForce = 0.0, twistForce = 0.0):

		anchor = add3(anchor, self.offset)

		# create the joint
		joint = ode.BallJoint(self.world)
		joint.attach(body1, body2)
		joint.setAnchor(anchor)

		# store the base orientation of the joint in the local coordinate system
		#   of the primary body (because baseAxis and baseTwistUp may not be
		#   orthogonal, the nearest vector to baseTwistUp but orthogonal to
		#   baseAxis is calculated and stored with the joint)
		joint.baseAxis = getBodyRelVec(body1, baseAxis)
		tempTwistUp = getBodyRelVec(body1, baseTwistUp)
		baseSide = norm3(cross(tempTwistUp, joint.baseAxis))
		joint.baseTwistUp = norm3(cross(joint.baseAxis, baseSide))

		# store the base twist up vector (original version) in the local
		#   coordinate system of the secondary body
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
				baseAxis = rotate3(j.getBody(0).getRotation(), j.baseAxis)
				currAxis = zaxis(j.getBody(1).getRotation())

				# get angular velocity of attached body relative to fixed body
				relAngVel = sub3(j.getBody(1).getAngularVel(),
					j.getBody(0).getAngularVel())
				twistAngVel = project3(relAngVel, currAxis)
				flexAngVel = sub3(relAngVel, twistAngVel)

				# restrict limbs rotating too far from base axis
				angle = acosdot3(currAxis, baseAxis)
				if angle > j.flexLimit:
					# add torque to push body back towards base axis
					j.getBody(1).addTorque(mul3(
						norm3(cross(currAxis, baseAxis)),
						(angle - j.flexLimit) * j.flexForce))

					# dampen flex to prevent bounceback
					j.getBody(1).addTorque(mul3(flexAngVel,
						-0.01 * j.flexForce))

				# determine the base twist up vector for the current attached
				#   body by applying the current joint flex to the fixed body's
				#   base twist up vector
				baseTwistUp = rotate3(j.getBody(0).getRotation(), j.baseTwistUp)
				base2current = calcRotMatrix(norm3(cross(baseAxis, currAxis)),
					acosdot3(baseAxis, currAxis))
				projBaseTwistUp = rotate3(base2current, baseTwistUp)

				# determine the current twist up vector from the attached body
				actualTwistUp = rotate3(j.getBody(1).getRotation(),
					j.baseTwistUp2)

				# restrict limbs twisting
				angle = acosdot3(actualTwistUp, projBaseTwistUp)
				if angle > j.twistLimit:
					# add torque to rotate body back towards base angle
					j.getBody(1).addTorque(mul3(
						norm3(cross(actualTwistUp, projBaseTwistUp)),
						(angle - j.twistLimit) * j.twistForce))

					# dampen twisting
					j.getBody(1).addTorque(mul3(twistAngVel,
						-0.01 * j.twistForce))


def createCapsule(world, space, density, length, radius):
	"""Creates a capsule body and corresponding geom."""

	# create capsule body (aligned along the z-axis so that it matches the
	#   GeomCCylinder created below, which is aligned along the z-axis by
	#   default)
	body = ode.Body(world)
	M = ode.Mass()	
	M.setCapsule(density, 3, radius, length)
	body.setMass(M)

	# set parameters for drawing the body
	body.shape = "capsule"
	body.length = length
	body.radius = radius

	# create a capsule geom for collision detection
	geom = ode.GeomCCylinder(space, radius, length)
	geom.setBody(body)

	return body, geom

def near_callback(args, geom1, geom2):
	"""
	Callback function for the collide() method.

	This function checks if the given geoms do collide and creates contact
	joints if they do.
	"""

	if (ode.areConnected(geom1.getBody(), geom2.getBody())):
		return

	# check if the objects collide
	contacts = ode.collide(geom1, geom2)

	# create contact joints
	world, contactgroup = args
	for c in contacts:
		c.setBounce(0.2)
		c.setMu(500) # 0-5 = very slippery, 50-500 = normal, 5000 = very sticky
		j = ode.ContactJoint(world, contactgroup, c)
		j.attach(geom1.getBody(), geom2.getBody())

def prepare_GL():
	"""Setup basic OpenGL rendering with smooth shading and a single light."""

	glClearColor(0.8, 0.8, 0.9, 0.0)
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glEnable(GL_DEPTH_TEST)
	glEnable(GL_LIGHTING)
	glEnable(GL_NORMALIZE)
	glShadeModel(GL_SMOOTH)

	glMatrixMode(GL_PROJECTION)
	glLoadIdentity()
	gluPerspective (45.0, 1.3333, 0.2, 20.0)

	glViewport(0, 0, 640, 480)

	glMatrixMode(GL_MODELVIEW)
	glLoadIdentity()

	glLightfv(GL_LIGHT0,GL_POSITION,[0, 0, 1, 0])
	glLightfv(GL_LIGHT0,GL_DIFFUSE,[1, 1, 1, 1])
	glLightfv(GL_LIGHT0,GL_SPECULAR,[1, 1, 1, 1])
	glEnable(GL_LIGHT0)

	glEnable(GL_COLOR_MATERIAL)
	glColor3f(0.8, 0.8, 0.8)

	gluLookAt(1.5, 4.0, 3.0, 0.5, 1.0, 0.0, 0.0, 1.0, 0.0)


# polygon resolution for capsule bodies
CAPSULE_SLICES = 16
CAPSULE_STACKS = 12

def draw_body(body):
	"""Draw an ODE body."""

	rot = makeOpenGLMatrix(body.getRotation(), body.getPosition())
	glPushMatrix()
	glMultMatrixd(rot)
	if body.shape == "capsule":
		cylHalfHeight = body.length / 2.0
		glBegin(GL_QUAD_STRIP)
		for i in range(0, CAPSULE_SLICES + 1):
			angle = i / float(CAPSULE_SLICES) * 2.0 * pi
			ca = cos(angle)
			sa = sin(angle)
			glNormal3f(ca, sa, 0)
			glVertex3f(body.radius * ca, body.radius * sa, cylHalfHeight)
			glVertex3f(body.radius * ca, body.radius * sa, -cylHalfHeight)
		glEnd()
		glTranslated(0, 0, cylHalfHeight)
		glutSolidSphere(body.radius, CAPSULE_SLICES, CAPSULE_STACKS)
		glTranslated(0, 0, -2.0 * cylHalfHeight)
		glutSolidSphere(body.radius, CAPSULE_SLICES, CAPSULE_STACKS)
	glPopMatrix()


def onKey(c, x, y):
	"""GLUT keyboard callback."""

	global SloMo, Paused

	# set simulation speed
	if c >= '0' and c <= '9':
		SloMo = 4 * int(c) + 1
	# pause/unpause simulation
	elif c == 'p' or c == 'P':
		Paused = not Paused
	# quit
	elif c == 'q' or c == 'Q':
		sys.exit(0)


def onDraw():
	"""GLUT render callback."""

	prepare_GL()

	for b in bodies:
		draw_body(b)
	for b in ragdoll.bodies:
		draw_body(b)

	glutSwapBuffers()


def onIdle():
	"""GLUT idle processing callback, performs ODE simulation step."""

	global Paused, lasttime, numiter

	if Paused:
		return

	t = dt - (time.time() - lasttime)
	if (t > 0):
		time.sleep(t)

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

	lasttime = time.time()

# initialize GLUT
glutInit([])
glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE)

# create the program window
x = 0
y = 0
width = 640
height = 480
glutInitWindowPosition(x, y);
glutInitWindowSize(width, height);
glutCreateWindow("PyODE Ragdoll Simulation")

# create an ODE world object
world = ode.World()
world.setGravity((0.0, -9.81, 0.0))
world.setERP(0.1)
world.setCFM(1E-4)

# create an ODE space object
space = ode.Space()

# create a plane geom to simulate a floor
floor = ode.GeomPlane(space, (0, 1, 0), 0)

# create a list to store any ODE bodies which are not part of the ragdoll (this
#   is needed to avoid Python garbage collecting these bodies)
bodies = []

# create a joint group for the contact joints generated during collisions
#   between two bodies collide
contactgroup = ode.JointGroup()

# set the initial simulation loop parameters
fps = 60
dt = 1.0 / fps
stepsPerFrame = 2
SloMo = 1
Paused = False
lasttime = time.time()
numiter = 0

# create the ragdoll
ragdoll = RagDoll(world, space, 500, (0.0, 0.9, 0.0))
print("total mass is %.1f kg (%.1f lbs)" % (ragdoll.totalMass,
	ragdoll.totalMass * 2.2))

# create an obstacle
obstacle, obsgeom = createCapsule(world, space, 1000, 0.05, 0.15)
pos = (random.uniform(-0.3, 0.3), 0.2, random.uniform(-0.15, 0.2))
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

