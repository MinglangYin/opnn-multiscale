from dolfin import *
import ufl
from numpy import zeros
import numpy as np
from numpy import linalg as LA
from numpy.linalg import norm as npNorm
import math

# Number of FE/peridynamic nodes across the width of the domain

N_ELEM = 10
N_NODES = N_ELEM+1
Tf = 1.0
Rbeta = 3;



cpp_code = r"""
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include<iostream>
#include<vector>
#include<cmath>
#include<cstdlib>
//#include"vvector.h"
#include <iomanip>
#include <fstream>

namespace py = pybind11;

template<class Type>
class vvector : public std::vector<Type>
{
  private:
    typedef std::vector<Type> Base;
  public:
    vvector() { }

    vvector(int size, Type val = 0) : Base(size, val) { }

    template<class C>
      vvector& operator=( vvector<C>& vv){
        if((*this).size() != vv.size()){
          std::cerr<<"Different dim"<<std::endl;
          return *this;
        }
        for(size_t i=0;i!=(*this).size();++i){
          (*this)[i]=vv[i];
        }
        return *this;
      }


    vvector& operator+=(const vvector& vv) {
      for(size_t i=0;i!=(*this).size();++i){
        (*this)[i]+=vv[i];
      }
      return *this;
    }

    vvector& operator-=(const vvector& vv) {
      for(size_t i=0;i!=(*this).size();++i){
        (*this)[i]-=vv[i];
      }
      return *this;
    }

    template<class C> vvector& operator*=(const C& CC) {
      for(size_t i=0;i!=(*this).size();++i){
        (*this)[i]*=CC;
      }
      return *this;
    }

    template<class C> vvector& operator/=(const C& CC) {
      for(size_t i=0;i!=(*this).size();++i){
        (*this)[i]/=CC;
      }
      return *this;
    }

    Type Norm1(){
      Type a(0);
      for(size_t i=0;i!=(*this).size();++i){
        a+=std::abs((*this)[i]);
      }
      return a;
    }

    Type Norm2(){
      Type a(0);
      for(size_t i=0;i!=(*this).size();++i){
        a+=pow((*this)[i],2);
      }
      return sqrt(a);
    }

    Type NormInfinite(){
      Type a(0);
      for(size_t i=0;i!=(*this).size();++i){
        if(std::abs((*this)[i])>a){
          a=std::abs((*this)[i]);
        }
      }
      return a;
    }

    template<class C>
      vvector<int> sign(){
        vvector<int> s((*this).size());
        for(size_t i=0;i!=(*this).size();++i){
          if((*this)[i]>0) {
            s[i]=1;
          }
          else s[i]=-1;
        }
        return s;
      }

    Type Householder(vvector<Type>& v){
      int n=(*this).size();
      Type eta=NormInfinite();
      if(eta==0){
        return 0;
      }
      (*this)/=eta;
      Type sigma(0);
      for(int i=1;i!=n;++i){
        sigma+=pow((*this)[i],2);
      }
      v=(*this);
      if(std::abs(sigma)>pow(10,-14)){
        Type alpha(sqrt(sigma+pow((*this)[0],2)));
        if((*this)[0] <= 0) { v[0]=(*this)[0]-alpha; }
        else v[0]=-sigma/((*this)[0]+alpha);
        Type beta=2*pow(v[0],2)/(sigma+pow(v[0],2));
        Type temp=v[0];
        v/=temp;
        return beta;
      }
      else{
        return 0;
      }

    }

    template<class T> friend std::ostream& operator<<(std::ostream&, const vvector&);

    template<class T> friend std::istream& operator>>(std::istream&, vvector&);
};

template<class T>
std::istream& operator>>(std::istream& in, vvector<T>& vv){
  std::cout<<"Please input numbers"<<std::endl;
  T value;
  while(in>>value){
    vv.push_back(value);
  }
  return in;
}

template<class T>
std::ostream& operator<<(std::ostream& os, const vvector<T>& vv){
  os.precision(4);
  os<<std::showpos;
  os.setf(std::ios::scientific);
  for(size_t i=0;i!=vv.size();++i){
    os<<std::setw(15)<<vv[i];
  }

  return os;
}

  template<class T>
vvector<T> operator+(const vvector<T>& v1, const vvector<T>& v2)
{
  if(v1.size() != v2.size()){
    std::cerr<<"diferent dim"<<std::endl;
    return v1;
  }
  vvector<T> v3(v1);
  v3 += v2;
  return v3;
}

  template<class T>
vvector<T> operator-(const vvector<T>& v1, const vvector<T>& v2)
{
  if(v1.size() != v2.size()){
    std::cerr<<"diferent dim"<<std::endl;
    return v1;
  }
  vvector<T> v3(v1);
  v3 -= v2;
  return v3;
}

  template<class T>
vvector<T> operator*(const vvector<T>& v1, const vvector<T>& v2)
{
  if(v1.size() != 3 || v2.size() != 3){
    std::cerr<<"Size don't match"<<std::endl;
    return v1;
  }
  vvector<T> v3(3);
  v3[0] = v1[1]*v2[2] - v1[2]*v2[1];
  v3[1] = v1[2]*v2[0] - v1[0]*v2[2];
  v3[2] = v1[0]*v2[1] - v1[1]*v2[0];
  return v3;
}


  template<class T,class T1>
vvector<T> operator*(const vvector<T>& v1, const T1& a)
{
  vvector<T> v3(v1);
  v3 *= a;
  return v3;
}

  template<class T,class T1>
vvector<T> operator*(const T1& a, const vvector<T>& v1)
{
  vvector<T> v3(v1);
  v3 *= a;
  return v3;
}

  template<class T,class T1>
vvector<T> operator/(const vvector<T>& v1, const T1& a)
{
  if(a==0){
    std::cerr<<"Cannot be divided by 0."<<std::endl;
    return v1;
  }
  vvector<T> v3(v1);
  v3 /= a;
  return v3;
}

typedef py::array_t<double, py::array::c_style | py::array::forcecast> 
    npArray;

extern "C" void dgetrf_(int*, int*, double*, int*, int*, int*);
extern "C" void dgetri_(int* , double* , int* , int* , double* , int* , int* );
extern "C" void dgetrs_(char*, int*, int*, double*, int*, int*, double*, int*, int*);
extern "C" void dgesvd_(char*, char*, int*, int*, double*, int*, double*, double*, int*, double*, int*, double*, int*, int*);


using namespace std;
const double pi = 3.1415926535897932384626;
const double beta = """+str(Rbeta)+""";

double weight(double r,double epsilon){
    if(r > epsilon) return 0;
    else return pow((1-r/epsilon),4);
}

void inverse(double* A, int N)
{
    int *IPIV = new int[N+1];
    int LWORK = N*N;
    double *WORK = new double[LWORK];
    int INFO;
    dgetrf_(&N,&N,A,&N,IPIV,&INFO);
    dgetri_(&N,A,&N,IPIV,WORK,&LWORK,&INFO);
    delete[] IPIV;
    delete[] WORK;
}
//basis funtions
//
/////////////////////////////////////////////////////////////////////////
//case 1: quadratic in space, static in time
double u_exact(double x, double y,double t){
  //return x*x;
  return t*sin(x)*cos(y);//x*x+y*y+t*t;//cos(myB*pi*x)*sin(myB*pi*y);//t*cos(myB*pi*x)*sin(myB*pi*y);
  //return t*(x*x+y*y);
}



double Ffun(double x,double y,double t){
  //return -2;
  return sin(x)*cos(y) + 2*t*sin(x)*cos(y);//2*t-4;//4*Bulk*myB*pi*myB*pi*cos(myB*pi*x)*sin(myB*pi*y)/5;//4*Bulk*myB*pi*myB*pi*t*cos(myB*pi*x)*sin(myB*pi*y)/5;
  //return x*x+y*y-4*t;
}

double Gfun(double x, double y,double t){
  return 2*x+beta*(x*x+y*y+t);//-4.*Bulk*myB*pi*sin(myB*pi*x)*sin(myB*pi*y)/5.;//-4.*Bulk*myB*pi*t*sin(myB*pi*x)*sin(myB*pi*y)/5.;
}


////////////////////////////////////////////////////////////////////////////


//////////////////////////////////////////////////////////////////////////////





int N="""+str(N_ELEM)+""";
const double delta=3.9;
double h = 1./N;            // lattice constant, all the nodes are evenly distributed
int k = floor(delta);
int dim = (N+1+2*k)*(N+1+k);
//int basedim = 10;      //linear basis funtions
double* x_0 = new double [dim];    //initial configuration
double* y_0 = new double [dim];
double* u = new double [dim];   //displacement in x direction and y direction
double* u0 = new double [dim];
double* du0 = new double [dim];
double* rhs = new double [dim];
double* stiffK = new double [dim*dim];
double* stiffKT = new double[dim*dim];
double *gin = new double [N-1];
double *uout = new double [N-1];

vector< vector<int> > nei((N+1)*(N+1)); //neighborlist

double diff = 100;
const double totalT="""+str(Tf)+""";
const int nstep = N*N/10;//time-step
const double Dt = totalT/nstep;//0.0001; // time-stepsize

int* IPIV = new int[dim];
int info;
int nrhs = 1;
char C='T';
char D = 'A';

void Preprocess()
{
  int cenx,cenx1;
  vvector<double> xi(2),zeta(2),gamma(2), eta(2);

  //set initial configuration
  for(int i=0;i<N+1+2*k;i++){
    for(int j=0;j<N+1+k;j++){
      x_0[i*(N+1+k)+j] = (j-k)*h;
      y_0[i*(N+1+k)+j] = (i-k)*h;
    }
  }


  for(int i=0;i<N+1+2*k;i++){
    for(int j=0;j<N+1+k;j++){
      u0[i*(N+1+k)+j]=u_exact(x_0[i*(N+1+k)+j],y_0[i*(N+1+k)+j],0.);
    }
  }


  for(int i=0;i<dim*dim;i++){
    stiffKT[i] = 0;
  }

  //set neighborhood list for every particle
  for(int i=0;i<N+1;i++){
    for(int j=0;j<N+1;j++){
      for(int s1=-k;s1<k+1;s1++){
        for(int s2=-k;s2<k+1;s2++){
          if((i+k+s1)*(N+1+k)+j+k+s2 < (N+1+2*k)*(N+1+k) && (i+k+s1)*(N+1+k)+j+k+s2 > -1){
            xi[0] = x_0[(i+k+s1)*(N+1+k)+j+k+s2] - x_0[(i+k)*(N+1+k)+j+k];
            xi[1] = y_0[(i+s1+k)*(N+1+k)+j+k+s2] - y_0[(i+k)*(N+1+k)+j+k];
            if(xi.Norm2() < delta*h){
              nei[i*(N+1)+j].push_back((i+k+s1)*(N+1+k)+j+k+s2);
            }
          }
        }
      }
    }
  }

    for(int i=1;i<N;i++){
        for(int j=1;j<N+1;j++){
            cenx = (i+k)*(N+1+k)+j+k;
            double* P = new double[nei[i*(N+1)+j].size()*6];
            double* M = new double[6*6];
            double* D = new double[nei[i*(N+1)+j].size()];
            double* Beta = new double[nei[i*(N+1)+j].size()*6];
            for(int s=0;s<nei[i*(N+1)+j].size();s++){
                P[s*6] = 1;
                P[s*6+1] = x_0[nei[i*(N+1)+j][s]]-x_0[cenx];
                P[s*6+2] = y_0[nei[i*(N+1)+j][s]]-y_0[cenx];
                P[s*6+3] = pow(x_0[nei[i*(N+1)+j][s]]-x_0[cenx],2);
                P[s*6+4] = pow(y_0[nei[i*(N+1)+j][s]]-y_0[cenx],2);
                P[s*6+5] = (x_0[nei[i*(N+1)+j][s]]-x_0[cenx])*(y_0[nei[i*(N+1)+j][s]]-y_0[cenx]);
                D[s] =  weight(sqrt(pow(P[s*6+1],2)+pow(P[s*6+2],2)),delta*h);
            }

            for(int m=0;m<6;m++){
                for(int n=0;n<6;n++){
                    M[m*6+n] = 0;
                    for(int s=0;s<nei[i*(N+1)+j].size();s++){
                       M[m*6+n] += P[s*6+m]*D[s]*P[s*6+n];
                    }
                }
            }

            inverse(M,6);

            for(int m=0;m<nei[i*(N+1)+j].size();m++){
                for(int n=0;n<6;n++){
                    Beta[m*6+n] = 0;
                    for(int s=0;s<6;s++){
                        Beta[m*6+n] += D[m]*P[m*6+s]*M[s*6+n];
                    }
                }
            }

            if(j < N-k ){
                for(int s=0;s<nei[i*(N+1)+j].size();s++){
                    stiffKT[cenx*dim+nei[i*(N+1)+j][s]] -= Beta[s*6+3]*2 + Beta[s*6+4]*2;
                }
                stiffKT[cenx*dim+cenx] += 1./Dt;
            }

            else if(j <= N){
                double distance = 1 - x_0[(i+k)*(N+1+k)+j+k];
                double c1 = -16*(pow(sqrt(pow(delta*h,2)-pow(distance,2)),3))/3/pi/pow(delta*h,4);
                double c2 = (pi+2*distance*(2*distance*distance-delta*delta*h*h)*sqrt(delta*delta*h*h-distance*distance)/pow(delta*h,4)
                          +2*atan(distance/sqrt(delta*delta*h*h-distance*distance)))/pi;
              for(int s=0;s<nei[i*(N+1)+j].size();s++){
                    stiffKT[cenx*dim+nei[i*(N+1)+j][s]] -= Beta[s*6+1]*(-16*(pow(sqrt(pow(delta*h,2)-pow(distance,2)),3))/3/pi/pow(delta*h,4))
                      +Beta[s*6+3]*(pi+2*distance*(2*distance*distance-delta*delta*h*h)*sqrt(delta*delta*h*h-distance*distance)/pow(delta*h,4)
                          +2*atan(distance/sqrt(delta*delta*h*h-distance*distance)))/pi
                      +Beta[s*6+4]*(pi+2*distance*(5*distance*distance-2*delta*delta*h*h)*sqrt(delta*delta*h*h-distance*distance)/3/pow(delta*h,4)
                          +2*atan(distance/sqrt(delta*delta*h*h-distance*distance)))/pi
                      +Beta[s*6+4]*(-c1*distance+(-distance*pow(sqrt(delta*delta*h*h-distance*distance),3))/3/pi/pow(delta*h,4))*2;
              }

                stiffKT[cenx*dim+cenx] += (0.5*c2-distance*c1)/Dt;
                stiffKT[cenx*dim+(i+k)*(N+1+k)+N+k] += -1*c1*beta;
                    
            }


            delete[] P;
            delete[] M;
            delete[] D;
            delete[] Beta;
        }
    }



  for(int i=0;i<N+1+2*k;i++){
    if(i <=k || i >= N+k){
      for(int j=0;j<N+1+k;j++){
        cenx = i*(N+1+k)+j;
        stiffKT[cenx*dim+cenx] = 1;
      }
    }
    else{
      for(int j=0;j<=k;j++){
        cenx = i*(N+1+k)+j;
        stiffKT[cenx*dim+cenx] = 1;
      }
    }
  }
}

//void BE(int s,npArray g, npArray v, int UPDATE)
//void BE(int s,double *g_ptr, double *v_ptr, int UPDATE)
void BE(int s,npArray g, npArray v, int UPDATE)
{
  int cenx,cenx1;

  auto g_buffer = g.request();
  double *g_ptr = (double *)g_buffer.ptr;
  auto v_buffer = v.request();
  double *v_ptr = (double *)v_buffer.ptr;

  //apply boundary condition
  for(int i=1;i<N;i++){
    for(int j=1;j<N+1;j++){
      cenx = (i+k)*(N+1+k)+j+k;
      if(j <= N && j>= N-k){
        double distance = 1- x_0[cenx];
        double c1 = -16*(pow(sqrt(pow(delta*h,2)-pow(distance,2)),3))/3/pi/pow(delta*h,4);
        double c2 = (pi+2*distance*(2*distance*distance-delta*delta*h*h)*sqrt(delta*delta*h*h-distance*distance)/pow(delta*h,4)
                          +2*atan(distance/sqrt(delta*delta*h*h-distance*distance)))/pi;

        rhs[cenx] = (0.5*c2-distance*c1)*(Ffun(x_0[cenx],y_0[cenx],s*Dt)+u0[cenx]/Dt)-c1*g_ptr[i-1];

      }
      else{
        rhs[cenx] = Ffun(x_0[cenx],y_0[cenx],s*Dt)+u0[cenx]/Dt;
      }
    }
  }


  //Dirichlet boundary condition
  for(int i=0;i<N+1+2*k;i++){
    if(i <=k || i >= N+k){
      for(int j=0;j<N+1+k;j++){
        rhs[i*(N+1+k)+j] = u_exact(x_0[i*(N+1+k)+j],y_0[i*(N+1+k)+j],s*Dt);
      }
    }
    else{
      for(int j=0;j<=k;j++){
        rhs[i*(N+1+k)+j] = u_exact(x_0[i*(N+1+k)+j],y_0[i*(N+1+k)+j],s*Dt);
      }
    }
  }




  for(int i=0;i<dim*dim;i++){
    stiffK[i] = stiffKT[i];
  }



  dgetrf_(&dim,&dim,stiffK,&dim,IPIV,&info);
  dgetrs_(&C,&dim,&nrhs,stiffK,&dim, IPIV,rhs,&dim,&info);

  for(int i=0;i<N+1;i++){
    for(int j=0;j<N+1;j++){
      cenx = (i+k)*(N+1+k)+j+k;
      du0[cenx] = (rhs[cenx]-u0[cenx])/Dt;
    }
  }

  if(UPDATE)
  {
    for(int i=0;i<dim;i++){
      u0[i] = rhs[i];
    }
  }
 
  /*ofstream myfile;
  myfile.open("Nonlocalu_x2.txt");
  for(int i=0;i<N+1+k;i++){
    myfile<<x_0[N/2*(N+1+k)+i]<<" "<<u0[N/2*(N+1+k)+i]<<endl;
  }

  myfile.close();*/
  for(int i=1;i<N;i++){
    int j=N;
    int cenx = (i+k)*(N+1+k)+j+k;
    v_ptr[i-1] = rhs[cenx];
 }

  double error=0.;
  double errorinf=0.;
if(UPDATE)
  {
      for(int i=0;i<N+1+2*k;i++){
        for(int j=0;j<N+1+k;j++){
          error += pow(u0[i*(N+1+k)+j]-u_exact(x_0[i*(N+1+k)+j],y_0[i*(N+1+k)+j],s*Dt),2);
          //  error += pow(u[i*(N+1+k)+j]-pow(x[i*(N+1+k)+j],2),2) + pow(u[i*(N+1+k)+j+(N+1+k)*(N+1+2*k)]-pow(x[i*(N+1+k)+j+(N+1+k)*(N+1+2*k)],2),2);
          //error += pow(u[i*(N+1+k)+j] - pow(x[i*(N+1+k)+j],2),2);
        }
      }
      for(int i=0;i<N+1+2*k;i++){
        for(int j=0;j<N+1+k;j++){
          if(fabs(u0[i*(N+1+k)+j]-u_exact(x_0[i*(N+1+k)+j],y_0[i*(N+1+k)+j],s*Dt)) > errorinf){
            errorinf = fabs(u0[i*(N+1+k)+j]-u_exact(x_0[i*(N+1+k)+j],y_0[i*(N+1+k)+j],s*Dt));
          }
        }
      }

      error /= (N+1)*(N+1);
      error = sqrt(error);
  }
  cout << "nonlocal: time: " << s*Dt <<"  L2 error: "<<error<<"   errorinf: "<<errorinf<<endl;
}

PYBIND11_MODULE(SIGNATURE, m)
{
    m.def("Preprocess",&Preprocess,"");
    m.def("BE",&BE,"");
}
"""

DELTA_Tf = Tf/float(N_ELEM*N_ELEM/10)

mesh = UnitSquareMesh(N_ELEM, N_ELEM)
localx = mesh.coordinates()
for i in range(0,len(localx)):
    localx[i,0] += 1.0

TOL = 1E-14
n = FacetNormal(mesh)


#u_E = Expression('t*(x[0]*x[0]+x[1]*x[1])',degree = 2,t = 0,domain=mesh)

u_E = Expression('t*sin(x[0])*cos(x[1])',degree = 2,t = 0,domain = mesh)
#u_E = Expression('x[0]*x[0]',degree = 2,t = 0,domain = mesh)


d = mesh.geometry().dim()
V = FunctionSpace(mesh,'P', 1)
V_vec = VectorFunctionSpace(mesh, 'DG',0)
VscalarE = FiniteElement("Lagrange",mesh.ufl_cell(),1)
Vscalar = FunctionSpace(mesh,VscalarE)
#V = Vscalar

u = TrialFunction(V)
v = TestFunction(V)
u_current = Function(V)
u_old = Function(V)
u_lastIter  = Function(V)

ubc = Function(Vscalar)
#uDirichletBC = DirichletBC(V,ubc,boundary_l)
uDirichletBC = DirichletBC(V,ubc,"on_boundary")


u_old = interpolate(u_E,V)
def getNodeCoordinates(V,nFields):
    dofxFlat = V.tabulate_dof_coordinates()
    #print(dofxFlat)
    dofx = dofxFlat.reshape((-1,d))
    #print(dofx)
    nNodes = dofx.shape[0]//nFields
    nodex = zeros((nNodes,d))
    for i in range(0,nNodes):
        nodex[i,:] = dofx[i,:]
    return nodex, nNodes

fluidNodex, nFluidNodes = getNodeCoordinates(V,1)

class Bdry(SubDomain):
    def inside(self,x,on_boundary):
        return on_boundary



class Interior(SubDomain):
    def inside(self,x,on_boundary):
        #return (not on_boundary) # Why doesn't this work!?!
        return (not (near(x[0],1) or near(x[0],2)
                     or near(x[1],0) or near(x[1],1)))

class BdryOfInterest(SubDomain):
    def inside(self,x,on_boundary):
        return (x[0] < 1.0 + DOLFIN_EPS) and on_boundary and (not (near(x[1],1) or near(x[1],0)))

class AntiBdry(SubDomain):
    def inside(self,x,on_boundary):
        return x[0] > 1.0 + DOLFIN_EPS or (near(x[1],1) or near(x[1],0))

FLUX_BDRY = 1
COMPLEMENT_FLUX_BDRY = 2
boundaryMarkers = MeshFunction("size_t",mesh,mesh.topology().dim()-1,
                               COMPLEMENT_FLUX_BDRY)
BdryOfInterest().mark(boundaryMarkers,FLUX_BDRY)
ds_marked = ds(subdomain_data=boundaryMarkers)

Bdy = zeros(N_NODES*2)
oriBdy = zeros(N_NODES*2)
k = 0
for i in range(0,nFluidNodes):
    if(near(fluidNodex[i,0],1.0) and (not near(fluidNodex[i,1],1.0)) and (not near(fluidNodex[i,1],0.0))):
        oriBdy[k] = i
        k+= 1

for i in range(N_NODES):
    Bdy[i] = oriBdy[N_NODES-3-i]


#print(Bdy)
antiBCs = [DirichletBC(V,Constant(0.0),AntiBdry()),]



#f = Expression('(x[0]*x[0]+x[1]*x[1])-4*t',degree = 2, t = 0)

#f = Constant(-2)
f = Expression('sin(x[0])*cos(x[1])+2*t*sin(x[0])*cos(x[1])',degree = 2, t = 0,domain = mesh)


#F = u*v*dx + DELTA_Tf*dot(grad(u), grad(v))*dx - (u_old + DELTA_Tf*f)*v*dx
F = (u*v/DELTA_Tf - u_old*v/DELTA_Tf + inner(grad(u),grad(v)) - f*v)*dx
a,L = lhs(F), rhs(F)

#A = assemble(a)
#rh = assemble(L)

#np.savetxt("./Best_robincoe/stiffmatrixgrad20.txt",A.array())
#np.savetxt("rhsqn.txt",rh)

def res(u,v):
    return (u*v/DELTA_Tf - u_old*v/DELTA_Tf + inner(grad(u),grad(v)) - f*v)*dx
qn = TrialFunction(V)
n = FacetNormal(mesh)
consistencyTerm = inner(grad(u_current),n)*v*ds_marked(COMPLEMENT_FLUX_BDRY)
     
FBdry = qn*v*ds_marked(FLUX_BDRY) - res(u_current,v) + consistencyTerm
#FBdry = qn*v*ds_marked(FLUX_BDRY) + consistencyTerm
#FBdry = qn*v*ds_marked(FLUX_BDRY) - res(u_current,v)
        
        
def solveFor_qn_h(FBdry, BCs):
    aBdry = lhs(FBdry)
    LBdry = rhs(FBdry)
    ABdry = assemble(aBdry,keep_diagonal=True)
    bBdry = assemble(LBdry)
    #np.savetxt("fluxmatrix1.txt",ABdry.array())
    #np.savetxt("fluxrhs1.txt",bBdry)
    [BC.apply(ABdry,bBdry) for BC in BCs]
    #np.savetxt("fluxmatrix2.txt",ABdry.array())
    #np.savetxt("fluxrhsres.txt",bBdry)
    qn_h = Function(V)
    solve(ABdry,qn_h.vector(),bBdry)
    return qn_h

ext_module = compile_cpp_code(cpp_code, libs=["lapack"])
ext_module.Preprocess()

gin = zeros(N_NODES-2,dtype="float64")
gin_last = zeros(N_NODES-2,dtype = "float64")
uin_last = zeros(N_NODES-2,dtype = "float64")
vout = zeros(N_NODES-2,dtype="float64")


N_TIMESTEPS = int(N_ELEM*N_ELEM/10)
MAX_SUBITERS = 200
SUBITER_TOL = 1e-8


for i in range(0,N_NODES-2):
    gin[i] = DELTA_Tf*cos(fluidNodex[int(Bdy[i]),0])*cos(fluidNodex[int(Bdy[i]),1]) + Rbeta*(DELTA_Tf*sin(fluidNodex[int(Bdy[i]),0])*cos(fluidNodex[int(Bdy[i]),1]))
    #gin[i] = 2*fluidNodex[int(Bdy[i]),0]*DELTA_Tf+Rbeta*DELTA_Tf*(fluidNodex[int(Bdy[i]),0]*fluidNodex[int(Bdy[i]),0]+fluidNodex[int(Bdy[i]),1]*fluidNodex[int(Bdy[i]),1])
    #gin_last[i] = 0. #2*fluidNodex[int(Bdy[i]),0]*DELTA_Tf+Rbeta*DELTA_Tf*(fluidNodex[int(Bdy[i]),0]*fluidNodex[int(Bdy[i]),0]+fluidNodex[int(Bdy[i]),1]*fluidNodex[int(Bdy[i]),1])
    #gin[i] = 2*fluidNodex[int(Bdy[i]),0] + Rbeta*(fluidNodex[int(Bdy[i]),0]*fluidNodex[int(Bdy[i]),0])
    #gin[i] = 1 + Rbeta*(fluidNodex[int(Bdy[i]),0])
SUB_TOTAL = 0;
tau1 = 1./2 
tau  = 0.1
Q_n = zeros(N_NODES-2)
Q_last = zeros(N_NODES-2)
Q_n1 = zeros(N_NODES-2)
Q_last1 = zeros(N_NODES-2)
taumin = -0.5
taumax = 0.95


t = 0

for timeStep in range(0, N_TIMESTEPS):
    t += DELTA_Tf
    f.t =t
    u_E.t = t 
    print("------- Time step "+str(timeStep+1)
          +" , t = "+str(t)+"-------")

    for subIter in range(0, 1):
        SUB_TOTAL = SUB_TOTAL + 1
        ext_module.BE(timeStep+1,gin,vout,1)
        uin = zeros(N_NODES-2)

        for i in range(0,N_NODES-2):
            uin[i] = vout[i]

        ubc.assign(interpolate(u_E,V))
        if subIter == 0:
            uBCarray = ubc.vector().get_local()
            for i in range(0,N_NODES-2):
                dofi = int(Bdy[i])
                uBCarray[dofi] = uin[i]
                Q_n1[i] = uin[i]

            ubc.vector().set_local(uBCarray)
        else:
            uBCarray = ubc.vector().get_local()
            for i in range(0,N_NODES-2):
                Q_n1[i] = uin[i] - u_lastIter.vector().get_local()[int(Bdy[i])]
            
            if(LA.norm(Q_last1-Q_n1) > 1E-10):
                tau1 = tau1 + (tau1-1)*(np.dot(Q_last1-Q_n1,Q_n1)/LA.norm(Q_last1-Q_n1)/LA.norm(Q_last1-Q_n1))
            if (tau1 < -0.5):
                tau1 = -0.5
            elif (tau1 > 0.95):
                tau1 = 0.95
            
            #print(tau1)
            #tau1 = 0.9
            for i in range(0,N_NODES-2):
                dofi = int(Bdy[i])
                uBCarray[dofi] = uin[i]
                #uBCarray[int(Bdy[i])] = (1-tau1)*uin[i] + tau1*u_lastIter.vector().get_local()[int(Bdy[i])]

            ubc.vector().set_local(uBCarray)
        
        solve(a == L, u_current, uDirichletBC)
        
        #print(u_current.vector().get_local())
        #np.savetxt("u_current.txt",u_current.vector().get_local())



        
        #qn_h = solveFor_qn_h(FBdry, antiBCs)
        
        gradu = project(grad(u_current), V_vec)
        
        errinf = 0 

        #for i in range(0,N_NODES-2):
        #    if(abs(qn_h.vector().get_local()[int(Bdy[i])]+2*fluidNodex[int(Bdy[i]),0]*t) > errinf):
         #       errinf =abs(qn_h.vector().get_local()[int(Bdy[i])]+2*fluidNodex[int(Bdy[i]),0]*t) 
        
        #print(errinf)
        #def fluxErr(qn):
        #    err = inner(grad(u_E),n) - qn
        #    return math.sqrt(assemble(err*err*ds_marked(FLUX_BDRY)))
        #print(fluxErr(qn_h))


        if subIter == 0:
            for i in range(0,N_NODES-2):
                #gin[i] = 2*((-1*qn_h.vector().get_local()[int(Bdy[i])]) +  Rbeta*(u_current.vector().get_local()[int(Bdy[i])])) - gin_last[i]
                #gin_last[i] =(-1*qn_h.vector().get_local()[int(Bdy[i])]) +  Rbeta*(u_current.vector().get_local()[int(Bdy[i])]) 
                if (timeStep==0):
                    gin[i] = gradu(fluidNodex[int(Bdy[i])])[0] +  Rbeta*(u_current.vector().get_local()[int(Bdy[i])])
                else:
                    gin[i] = gradu(fluidNodex[int(Bdy[i])])[0] +  Rbeta*(u_current.vector().get_local()[int(Bdy[i])])
                    #gin[i] = gradu(fluidNodex[int(Bdy[i])])[0] +  Rbeta*(2.*u_current.vector().get_local()[int(Bdy[i])]-u_lastIter.vector().get_local()[int(Bdy[i])])
                #gin[i] = gradu(fluidNodex[int(Bdy[i])])[0] + Rbeta*(u_current.vector().get_local()[int(Bdy[i])])
                #gin[i] = 2*fluidNodex[int(Bdy[i]),0]*t_last+Rbeta*t_last*(fluidNodex[int(Bdy[i]),0]*fluidNodex[int(Bdy[i]),0]+fluidNodex[int(Bdy[i]),1]*fluidNodex[int(Bdy[i]),1])
                #print(gin[i])
                #print(fluidNodex[int(Bdy[i])])
                #Q_n[i] = gin[i]
                #gin[i] = Rbeta*(u_current.vector().get_local()[int(Bdy[i])])
        else:
            for i in range(0,N_NODES-2):
                gin[i] = 2*((-1*qn_h.vector().get_local()[int(Bdy[i])]) +  Rbeta*(u_current.vector().get_local()[int(Bdy[i])])) - gin_last[i]
                gin_last[i] =(-1*qn_h.vector().get_local()[int(Bdy[i])]) +  Rbeta*(u_current.vector().get_local()[int(Bdy[i])]) 
                #gin[i] = (-1*qn_h.vector().get_local()[int(Bdy[i])]) +  Rbeta*(u_current.vector().get_local()[int(Bdy[i])])
                #gin[i] = 2*fluidNodex[int(Bdy[i]),0]*t+  Rbeta*(u_current.vector().get_local()[int(Bdy[i])])
                #gin[i] = 2*fluidNodex[int(Bdy[i]),0]*t_last+Rbeta*t_last*(fluidNodex[int(Bdy[i]),0]*fluidNodex[int(Bdy[i]),0]+fluidNodex[int(Bdy[i]),1]*fluidNodex[int(Bdy[i]),1])
                #gin[i] = 0.1*((-1*qn_h.vector().get_local()[int(Bdy[i])]) +  Rbeta*(u_current.vector().get_local()[int(Bdy[i])]))+0.9*gin_last[i]
                #gin[i] = gradu(fluidNodex[int(Bdy[i])])[0] + Rbeta*(u_current.vector().get_local()[int(Bdy[i])])
                #gin[i] = Rbeta*(u_current.vector().get_local()[int(Bdy[i])])
                Q_n[i] = gin[i] - gin_last[i]
        

            if(LA.norm(Q_last-Q_n) > 1E-10):
                tau = tau + (tau-1)*(np.dot(Q_last-Q_n,Q_n)/LA.norm(Q_last-Q_n)/LA.norm(Q_last-Q_n))
            if (tau < -0.5):
                tau = -0.5
            elif (tau > 0.95):
                tau = 0.95

            
            #print(tau)
            #tau = 0.9

            #for i in range(0,N_NODES-2):
                #gin[i] = (1-tau)*gin[i] + tau*gin_last[i]
                #gin[i] = 2*gin[i] - gin_last[i]

        for i in range(0,N_NODES-2):
            gin_last[i] = gin[i]
            Q_last[i] = Q_n[i]
            Q_last1[i] = Q_n1[i]
            #ubc_last[i] = uin[i]
        #gradu = project(grad(u_current), V_vec)

        #print(gin)

#        u_old.assign(u_current)

        diff = Function(V)
        diff.assign(u_current - u_lastIter)

        u_lastIter.assign(u_current)

        normDiff = math.sqrt(assemble(((diff)**2)*dx))
        normLastIter = math.sqrt(assemble((u_lastIter**2)*dx))
        rel = normDiff/normLastIter

        print("Relative change: "+str(rel))

        if(rel < SUBITER_TOL):
            break

    #ext_module.BE(timeStep+1,gin,vout,1)
    u_old.assign(u_current)
    
    u_e = interpolate(u_E,V)
    error = np.abs(u_e.vector().get_local()-u_current.vector().get_local()).max()
    error2 = math.sqrt(assemble(((u_current-u_e)**2)*dx))
    #error = math.sqrt(assemble(((u_current-u_E(xFluid))**2)*dx))
    print('error=%.10f' %(error))
    print('error2=%.10f' %(error2))
    
SUB_TOTAL = SUB_TOTAL/N_TIMESTEPS
print('Number of subiteration: %.10f' %(SUB_TOTAL))
#for i in range(nFluidNodes):
#    if near(fluidNodex[i,1],0.5):
#        print(fluidNodex[i,0],u_current(fluidNodex[i]))

