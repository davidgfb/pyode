from ode import World, Space, Body, BallJoint, JointGroup, Mass, GeomBox,\
     areConnected, Infinity

NUM = 10
SIDE = 0.2
MASS = 1
RADIUS = 0.1732

world = World()
space = Space()
body = Body(world)
joint = BallJoint(world)
contactgroup = JointGroup()

def nearCallback(args, geom1, geom2):
    b1, b2 = geom1.getBody(), geom2.getBody()

    if not (b1 and b2 and areConnected(b1,b2)):
        contacts = collide(geom1, geom2) #?
        contact.surface.mode = 0
        contact.surface.mu = Infinity

        '''for c in contacts:
            c.setMu(500)'''
    
        #if collide():
            
        
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
    geom = GeomBox(space, lengths=body.boxsize)
    geom.setBody(body)

    return body
