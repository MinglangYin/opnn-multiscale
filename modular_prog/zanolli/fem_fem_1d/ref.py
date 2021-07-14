from fenics import *

def bvp_05 ( ):
#*****************************************************************************80
#
## bvp_05 solves a boundary value problem.
#
#  Discussion:
#
#    -u'' + u = x, 0 < x < 1
#    u(0) = 0, 
#    u'(1) = 1-cosh(1)/sinh(1)
#
#    Exact solution is u(x) = x - sinh(x)/sinh(1)

    import matplotlib.pyplot as plt
    mesh = UnitIntervalMesh ( 10 )
    f = Expression ( "x[0]", degree = 10 )
    ue = Expression ( "x[0] - sinh ( x[0] ) / sinh ( 1.0 )", degree = 10 ) 
    g = Expression ( "1.0 - cosh(x[0]) / sinh(1.0)", degree = 10 )
    V = FunctionSpace ( mesh, "CG", 1 )

    def left_boundary ( x, on_boundary ):
        return on_boundary and ( abs ( x[0] - 0.0 ) < DOLFIN_EPS )

    u_left  = ue ( 0.0 )
    bc = DirichletBC ( V, u_left, left_boundary )

    psii = TrialFunction ( V )
    psij = TestFunction ( V )

    A   = inner ( grad ( psii ), grad ( psij ) ) * dx + psii * psij * dx 
    RHS = f * psij * dx + g * psij * ds

    u = Function ( V )
    solve ( A == RHS, u, bc )

    plot ( u, title = 'bvp_05 solution' )
    filename = 'bvp_05_solution.png'
    plt.savefig ( filename )
    print ( '  Graphics saved as "%s"' % ( filename ) )
    plt.close ( )

    return

def bvp_05_test():
    import time
    print ( time.ctime ( time.time() ) )
    level = 30
    set_log_level ( level )

    print ( '' )
    print ( 'bvp_05_test:' )
    print ( '  FENICS/Python version' )
    print ( '  Solve  -u" + u = x, 0 < x < 1' )
    print ( '  u(0) = 0' )
    print ( '  u''(1) = 1-cosh(1)/sinh(1)' )
    print ( '  Exact solution: u(x) = x - sinh(x)/sinh(1)' )
    print ( '  Notice Neumann condition on the right.' )

    bvp_05 ( )

    print ( '' )
    print ( 'bvp_05_test:' )
    print ( '  Normal end of execution.' )
    print ( '' )
    print ( time.ctime ( time.time() ) )
    return

if ( __name__ == '__main__' ):
    bvp_05_test ( )

