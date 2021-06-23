from dolfin import *
from mshr import *
import numpy as np
import ufl
from numpy import zeros
from numpy.linalg import norm as npNorm
import math


T = 1.0
N = 20
num_steps = int(N*N/400)
dt = T/num_steps
Rbeta = 2
mesh = Mesh("corner10.xml")

cpp_code = r"""
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include<iostream>
#include<vector>
#include<cmath>
#include<cstdlib>
#include <iomanip>
#include <fstream>
#include<algorithm>

namespace py = pybind11;


typedef py::array_t<double, py::array::c_style | py::array::forcecast> 
    npArray;


extern "C" void dgetrf_(int*, int*, double*, int*, int*, int*);
extern "C" void dgetri_(int* , double* , int* , int* , double* , int* , int* );
extern "C" void dgetrs_(char*, int*, int*, double*, int*, int*, double*, int*, int*);
extern "C" void dgesvd_(char*, char*, int*, int*, double*, int*, double*, double*, int*, double*, int*, double*, int*, int*);

using namespace std;
const double pi = 3.1415926535897932384626;

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





double u_exact(double x, double y,double t){
  //return x+y;//cos(myB*pi*x)*sin(myB*pi*y);//t*cos(myB*pi*x)*sin(myB*pi*y);
  //return t*sin(0.1*x);
  //return t*(x*x+y*y+x*y);
  //return t*t;
  //return x;
  return t*t*(sin(x)*cos(y));
}



double Ffun(double x,double y,double t){
  //return 0;//4*Bulk*myB*pi*myB*pi*cos(myB*pi*x)*sin(myB*pi*y)/5;//4*Bulk*myB*pi*myB*pi*t*cos(myB*pi*x)*sin(myB*pi*y)/5;
  //return 2*t;
  //return (x*x+y*y+x*y)-4*t;
  //return sin(0.1*x)+0.01*t*sin(0.1*x);
  return 2*t*sin(x)*cos(y) + 2*t*t*sin(x)*cos(y);    
}

double Gfun(double x, double y,double n1,double n2,double t){
  //return n1+n2;//-4.*Bulk*myB*pi*sin(myB*pi*x)*sin(myB*pi*y)/5.;//-4.*Bulk*myB*pi*t*sin(myB*pi*x)*sin(myB*pi*y)/5.;
  //return 0.1*t*cos(0.1*x)*n1;
  return t*t*(cos(x)*cos(y)*n1-sin(x)*sin(y)*n2);
  //return t*(2*x+y)*n1+t*(x+2*y)*n2;
  //return 0;
}
double dGfunx(double x,double y,double n1,double n2, double t){
  //return (-0.01*t*sin(0.1*x))*n1+0*n2;
  //return 2*t*n1+t*n2;
  //return t*t*(-sin(x)*cos(y)*n1-cos(x)*sin(y)*n2);
  //return 0*n1+0*n2;
  return 0;
}
double dGfuny(double x,double y,double n1,double n2, double t){
  //return 0*n1+0*n2;
  //return t*t*(-cos(x)*sin(y)*n1-sin(x)*cos(y)*n2);
  //return t*n1+2*t*n2;
  //return 0*n1+0*n2;
  return 0;
}

int N="""+str(N)+""";
double rbeta = """+str(Rbeta)+""";
const double delta=3.5;
double h = 2./N;            // lattice constant, all the nodes are evenly distributed
int k = floor(delta);
//int basedim = 10;      //linear basis funtions
double* x_0;    //initial configuration
double* y_0;
double* u;   //displacement in x direction and y direction
double* u0;
double* du0;
double* rhs;
double* stiffK;
double* stiffKT;
double *gin;
double *uout;
double* normal = new double[2];
double* tanvec = new double[2];
double** localbeta;
double* localbdy;
int* proj_nei = new int[(2*k)*(2*k)];
double* projbeta = new double[2*k*2*k];
int projsize;
vector< vector<int> > bdynei;
int dim;
int Nodeid;


double diff = 100;
const double totalT="""+str(T)+""";
const int nstep = N*N/400;//time-step
const double Dt = totalT/nstep;//0.0001; // time-stepsize

int* IPIV;
int info;
int nrhs = 1;
char C='T';
char D = 'A';
int N_point = 0;
double Tol = 1e-10;

bool ifinhorizon(double x, double y, double x0,double y0){
  if((x-x0)*(x-x0)+(y-y0)*(y-y0) <= delta*delta*h*h+Tol){
    return true;
  }
  else{
    return false;
  }
}

int search_projection(double x_proj,double y_proj){
  for(int i=0;i<N_point;i++){
    if(fabs(x_proj-x_0[i]) + fabs(y_proj-y_0[i]) < 1e-8){
      return i;
    }
  }
  return -1;
}



void Preprocess(){
  double x,y;
  for(int i=0;i<N+1;i++){
    for(int j=0;j<N+1;j++){
      x = -1 + i*h;
      y = -1 + j*h;
      if(min(abs(x),abs(y))<= 0.5+Tol){
            N_point += 1;
          }
    }
  }
  x_0 = new double[N_point];
  y_0 = new double[N_point];
  u = new double[N_point];
  rhs = new double[N_point];
  u0 = new double[N_point];
  du0 = new double[N_point];
  stiffK = new double[N_point*N_point];
  stiffKT = new double[N_point*N_point];
  dim = N_point;


  int s=0;
  
  for(int i=0;i<N+1;i++){
    for(int j=0;j<N+1;j++){
      x = -1 + i*h;
      y = -1 + j*h;
      if(min(abs(x),abs(y))<= 0.5+Tol){
          x_0[s] = x;
          y_0[s] = y;
          s = s+1;
          }
    }
  }

  
  vector< vector<int> > nei(N_point); //neighborlist
  
  
  for(int i=0;i<N_point;i++){
    for(int j=0;j<N_point;j++){
      if((x_0[j]-x_0[i])*(x_0[j]-x_0[i]) + (y_0[j]-y_0[i])*(y_0[j]-y_0[i]) < delta*delta*h*h){
        nei[i].push_back(j);
      }  
    }
  }
  

  for(int i=0; i<N_point;i++){
    u0[i] = u_exact(x_0[i],y_0[i],0);
  }

  for(int i=0;i<dim*dim;i++){
    stiffKT[i] = 0;
  }


  for(int i=1;i<N_point;i++){
        double* P = new double[nei[i].size()*6];
        double* M = new double[6*6];
        double* D = new double[nei[i].size()]; 
        double* Beta = new double[nei[i].size()*6];
  
        if(abs(x_0[i]+1) <= delta*h+Tol && abs(y_0[i]-0.5)<=delta*h+Tol){  
            normal[0] = -1;
            normal[1] = 0;
            tanvec[0] = 0;
            tanvec[1] = 1;
            for(int s=0;s!=nei[i].size();s++){
              P[s*6] = 1;
              P[s*6+1] = (x_0[nei[i][s]] - x_0[i])*normal[0] + (y_0[nei[i][s]]-y_0[i])*normal[1]; 
              P[s*6+2] = (x_0[nei[i][s]] - x_0[i])*tanvec[0] + (y_0[nei[i][s]]-y_0[i])*tanvec[1];
              P[s*6+3] = pow((x_0[nei[i][s]] - x_0[i])*normal[0] + (y_0[nei[i][s]]-y_0[i])*normal[1],2);
              P[s*6+4] = pow((x_0[nei[i][s]] - x_0[i])*tanvec[0] + (y_0[nei[i][s]]-y_0[i])*tanvec[1],2);
              P[s*6+5] =((x_0[nei[i][s]] - x_0[i])*normal[0] + (y_0[nei[i][s]]-y_0[i])*normal[1])*((x_0[nei[i][s]] - x_0[i])*tanvec[0] + (y_0[nei[i][s]]-y_0[i])*tanvec[1]);
              D[s] =  weight(sqrt(pow(P[s*6+1],2)+pow(P[s*6+2],2)),delta*h);
            }
            for(int m=0;m<6;m++){
              for(int n=0;n<6;n++){
                  M[m*6+n] = 0;
                  for(int s=0;s<nei[i].size();s++){
                      M[m*6+n] += P[s*6+m]*D[s]*P[s*6+n];
                  }
              }
            }
        
          inverse(M,6);
          for(int m=0;m<nei[i].size();m++){
             for(int n=0;n<6;n++){
                  Beta[m*6+n] = 0;
                  for(int s=0;s<6;s++){
                      Beta[m*6+n] += D[m]*P[m*6+s]*M[s*6+n];
                  }
              }
          }
          double dx = 1 + x_0[i];
          double dy = 0.5 - y_0[i];
          
          if(ifinhorizon(x_0[i],y_0[i],-1,0.5)){
            double cx = 8*(-pow(dy,3)/3-pow(sqrt(delta*delta*h*h-dx*dx),3)/3+dy*(dx*dx+dy*dy-delta*delta*h*h)/2)/pi/pow(delta*h,4);
            double cy = 8*(-pow(dx,3)/3-pow(sqrt(delta*delta*h*h-dy*dy),3)/3+dx*(dx*dx+dy*dy-delta*delta*h*h)/2)/pi/pow(delta*h,4);
            double cxx = ((dx*(2*dx*dx-delta*delta*h*h)*sqrt(delta*delta*h*h-dx*dx)-sqrt(delta*delta*h*h-dy*dy)*(delta*delta*h*h-2*dy*dy)*dy)/pow(delta*h,4)
                +atan(dx/sqrt(delta*delta*h*h-dx*dx))+atan(-sqrt(delta*delta*h*h-dy*dy)/dy)+pi)/pi
                +8*dy*(pow(dx,3)+pow(sqrt(delta*delta*h*h-dy*dy),3))/3/pi/pow(delta*h,4);
            double cyy = ((dy*(2*dy*dy-delta*delta*h*h)*sqrt(delta*delta*h*h-dy*dy)-sqrt(delta*delta*h*h-dx*dx)*(delta*delta*h*h-2*dx*dx)*dx)/pow(delta*h,4)
                +atan(dy/sqrt(delta*delta*h*h-dy*dy))+atan(-sqrt(delta*delta*h*h-dx*dx)/dx)+pi)/pi
                +8*dx*(pow(dy,3)+pow(sqrt(delta*delta*h*h-dx*dx),3))/3/pi/pow(delta*h,4);
            double cxy = pow(dx*dx+dy*dy-delta*delta*h*h,2)/pi/pow(delta*h,4);
            double x_proj1 = -1;
            double y_proj1 = y_0[i];
            double x_proj2 = x_0[i];
            double y_proj2 = 0.5;
            
            for(int s=0;s<nei[i].size();s++){
              stiffKT[i*dim+nei[i][s]] -= Beta[s*6+1]*cx+Beta[s*6+2]*cy+Beta[s*6+3]*cxx+Beta[s*6+4]*cyy+Beta[s*6+5]*cxy+Beta[s*6+4]*(0.5*(cxx-cyy)+dy*cy-dx*cx)*2;
            }
            stiffKT[i*dim+i] += (0.5*cxx-cx*dx)/Dt;
            Nodeid = search_projection(x_proj1,y_proj1);
            stiffKT[i*dim+Nodeid] += -cx*rbeta; 
            Nodeid = search_projection(x_proj2,y_proj2);
            stiffKT[i*dim+Nodeid] += -cy*rbeta; 
          }
          else{
            double cx = -16*pow(sqrt(delta*delta*h*h-dx*dx),3)/3/pi/pow(delta*h,4);
            double cy = -16*pow(sqrt(delta*delta*h*h-dy*dy),3)/3/pi/pow(delta*h,4);
            double cxx = 2*((dx*(2*dx*dx-delta*delta*h*h)*sqrt(delta*delta*h*h-dx*dx)+dy*(5*delta*delta*h*h-2*dy*dy)*sqrt(delta*delta*h*h-dy*dy)/3)/pow(delta*h,4)
                +atan(dx/sqrt(delta*delta*h*h-dx*dx))+atan(dy/sqrt(delta*delta*h*h-dx*dx)))/pi; 
            double cyy = 2*((dy*(2*dy*dy-delta*delta*h*h)*sqrt(delta*delta*h*h-dy*dy)+dx*(5*delta*delta*h*h-2*dx*dx)*sqrt(delta*delta*h*h-dx*dx)/3)/pow(delta*h,4)
                +atan(dx/sqrt(delta*delta*h*h-dx*dx))+atan(dy/sqrt(delta*delta*h*h-dx*dx)))/pi; 
            double x_proj1 = -1;
            double y_proj1 = y_0[i];
            double x_proj2 = x_0[i];
            double y_proj2 = 0.5;
            for(int s=0;s<nei[i].size();s++){
              stiffKT[i*dim+nei[i][s]] -= Beta[s*6+1]*cx+Beta[s*6+2]*cy+Beta[s*6+3]*cxx+Beta[s*6+4]*cyy+Beta[s*6+4]*(0.5*(cxx-cyy)+dy*cy-dx*cx)*2;
            }
            stiffKT[i*dim+i] += (0.5*cxx-cx*dx)/Dt;
            Nodeid = search_projection(x_proj1,y_proj1);
            stiffKT[i*dim+Nodeid] += -cx*rbeta; 
            Nodeid = search_projection(x_proj2,y_proj2);
            stiffKT[i*dim+Nodeid] += -cy*rbeta; 
          }
        }
        else if(abs(x_0[i]+1) <= delta*h+Tol && abs(y_0[i]+0.5)<=delta*h+Tol){
            normal[0] = -1;
            normal[1] = 0;
            tanvec[0] = 0;
            tanvec[1] = -1;
            for(int s=0;s!=nei[i].size();s++){
              P[s*6] = 1;
              P[s*6+1] = (x_0[nei[i][s]] - x_0[i])*normal[0] + (y_0[nei[i][s]]-y_0[i])*normal[1]; 
              P[s*6+2] = (x_0[nei[i][s]] - x_0[i])*tanvec[0] + (y_0[nei[i][s]]-y_0[i])*tanvec[1];
              P[s*6+3] = pow((x_0[nei[i][s]] - x_0[i])*normal[0] + (y_0[nei[i][s]]-y_0[i])*normal[1],2);
              P[s*6+4] = pow((x_0[nei[i][s]] - x_0[i])*tanvec[0] + (y_0[nei[i][s]]-y_0[i])*tanvec[1],2);
              P[s*6+5] =((x_0[nei[i][s]] - x_0[i])*normal[0] + (y_0[nei[i][s]]-y_0[i])*normal[1])*((x_0[nei[i][s]] - x_0[i])*tanvec[0] + (y_0[nei[i][s]]-y_0[i])*tanvec[1]);
              D[s] =  weight(sqrt(pow(P[s*6+1],2)+pow(P[s*6+2],2)),delta*h);
            }
            for(int m=0;m<6;m++){
              for(int n=0;n<6;n++){
                  M[m*6+n] = 0;
                  for(int s=0;s<nei[i].size();s++){
                      M[m*6+n] += P[s*6+m]*D[s]*P[s*6+n];
                  }
              }
            }
        
          inverse(M,6);
          for(int m=0;m<nei[i].size();m++){
             for(int n=0;n<6;n++){
                  Beta[m*6+n] = 0;
                  for(int s=0;s<6;s++){
                      Beta[m*6+n] += D[m]*P[m*6+s]*M[s*6+n];
                  }
              }
          }
          double dx = 1 + x_0[i];
          double dy = 0.5 + y_0[i];
          
          if(ifinhorizon(x_0[i],y_0[i],-1,-0.5)){
            double cx = 8*(-pow(dy,3)/3-pow(sqrt(delta*delta*h*h-dx*dx),3)/3+dy*(dx*dx+dy*dy-delta*delta*h*h)/2)/pi/pow(delta*h,4);
            double cy = 8*(-pow(dx,3)/3-pow(sqrt(delta*delta*h*h-dy*dy),3)/3+dx*(dx*dx+dy*dy-delta*delta*h*h)/2)/pi/pow(delta*h,4);
            double cxx = ((dx*(2*dx*dx-delta*delta*h*h)*sqrt(delta*delta*h*h-dx*dx)-sqrt(delta*delta*h*h-dy*dy)*(delta*delta*h*h-2*dy*dy)*dy)/pow(delta*h,4)
                +atan(dx/sqrt(delta*delta*h*h-dx*dx))+atan(-sqrt(delta*delta*h*h-dy*dy)/dy)+pi)/pi
                +8*dy*(pow(dx,3)+pow(sqrt(delta*delta*h*h-dy*dy),3))/3/pi/pow(delta*h,4);
            double cyy = ((dy*(2*dy*dy-delta*delta*h*h)*sqrt(delta*delta*h*h-dy*dy)-sqrt(delta*delta*h*h-dx*dx)*(delta*delta*h*h-2*dx*dx)*dx)/pow(delta*h,4)
                +atan(dy/sqrt(delta*delta*h*h-dy*dy))+atan(-sqrt(delta*delta*h*h-dx*dx)/dx)+pi)/pi
                +8*dx*(pow(dy,3)+pow(sqrt(delta*delta*h*h-dx*dx),3))/3/pi/pow(delta*h,4);
            double cxy = pow(dx*dx+dy*dy-delta*delta*h*h,2)/pi/pow(delta*h,4);
            for(int s=0;s<nei[i].size();s++){
              stiffKT[i*dim+nei[i][s]] -= Beta[s*6+1]*cx+Beta[s*6+2]*cy+Beta[s*6+3]*cxx+Beta[s*6+4]*cyy+Beta[s*6+5]*cxy+Beta[s*6+4]*(0.5*(cxx-cyy)+dy*cy-dx*cx)*2;
            }
            stiffKT[i*dim+i] += (0.5*cxx-cx*dx)/Dt;
            double x_proj1 = -1;
            double y_proj1 = y_0[i];
            double x_proj2 = x_0[i];
            double y_proj2 = -0.5;
            Nodeid = search_projection(x_proj1,y_proj1);
            stiffKT[i*dim+Nodeid] += -cx*rbeta; 
            Nodeid = search_projection(x_proj2,y_proj2);
            stiffKT[i*dim+Nodeid] += -cy*rbeta; 
          }
          else{
            double cx = -16*pow(sqrt(delta*delta*h*h-dx*dx),3)/3/pi/pow(delta*h,4);
            double cy = -16*pow(sqrt(delta*delta*h*h-dy*dy),3)/3/pi/pow(delta*h,4);
            double cxx = 2*((dx*(2*dx*dx-delta*delta*h*h)*sqrt(delta*delta*h*h-dx*dx)+dy*(5*delta*delta*h*h-2*dy*dy)*sqrt(delta*delta*h*h-dy*dy)/3)/pow(delta*h,4)
                +atan(dx/sqrt(delta*delta*h*h-dx*dx))+atan(dy/sqrt(delta*delta*h*h-dx*dx)))/pi; 
            double cyy = 2*((dy*(2*dy*dy-delta*delta*h*h)*sqrt(delta*delta*h*h-dy*dy)+dx*(5*delta*delta*h*h-2*dx*dx)*sqrt(delta*delta*h*h-dx*dx)/3)/pow(delta*h,4)
                +atan(dx/sqrt(delta*delta*h*h-dx*dx))+atan(dy/sqrt(delta*delta*h*h-dx*dx)))/pi; 
            for(int s=0;s<nei[i].size();s++){
              stiffKT[i*dim+nei[i][s]] -= Beta[s*6+1]*cx+Beta[s*6+2]*cy+Beta[s*6+3]*cxx+Beta[s*6+4]*cyy+Beta[s*6+4]*(0.5*(cxx-cyy)+dy*cy-dx*cx)*2;
            }
            stiffKT[i*dim+i] += (0.5*cxx-cx*dx)/Dt;
            double x_proj1 = -1;
            double y_proj1 = y_0[i];
            double x_proj2 = x_0[i];
            double y_proj2 = -0.5;
            Nodeid = search_projection(x_proj1,y_proj1);
            stiffKT[i*dim+Nodeid] += -cx*rbeta; 
            Nodeid = search_projection(x_proj2,y_proj2);
            stiffKT[i*dim+Nodeid] += -cy*rbeta; 
          }
        }
        else if(abs(x_0[i]+0.5) <= delta*h+Tol && abs(y_0[i]-1)<=delta*h+Tol){
            normal[0] = -1;
            normal[1] = 0;
            tanvec[0] = 0;
            tanvec[1] = 1;
            for(int s=0;s!=nei[i].size();s++){
              P[s*6] = 1;
              P[s*6+1] = (x_0[nei[i][s]] - x_0[i])*normal[0] + (y_0[nei[i][s]]-y_0[i])*normal[1]; 
              P[s*6+2] = (x_0[nei[i][s]] - x_0[i])*tanvec[0] + (y_0[nei[i][s]]-y_0[i])*tanvec[1];
              P[s*6+3] = pow((x_0[nei[i][s]] - x_0[i])*normal[0] + (y_0[nei[i][s]]-y_0[i])*normal[1],2);
              P[s*6+4] = pow((x_0[nei[i][s]] - x_0[i])*tanvec[0] + (y_0[nei[i][s]]-y_0[i])*tanvec[1],2);
              P[s*6+5] =((x_0[nei[i][s]] - x_0[i])*normal[0] + (y_0[nei[i][s]]-y_0[i])*normal[1])*((x_0[nei[i][s]] - x_0[i])*tanvec[0] + (y_0[nei[i][s]]-y_0[i])*tanvec[1]);
              D[s] =  weight(sqrt(pow(P[s*6+1],2)+pow(P[s*6+2],2)),delta*h);
            }
            for(int m=0;m<6;m++){
              for(int n=0;n<6;n++){
                  M[m*6+n] = 0;
                  for(int s=0;s<nei[i].size();s++){
                      M[m*6+n] += P[s*6+m]*D[s]*P[s*6+n];
                  }
              }
            }
        
          inverse(M,6);
          for(int m=0;m<nei[i].size();m++){
             for(int n=0;n<6;n++){
                  Beta[m*6+n] = 0;
                  for(int s=0;s<6;s++){
                      Beta[m*6+n] += D[m]*P[m*6+s]*M[s*6+n];
                  }
              }
          }
          double dx = 0.5 + x_0[i];
          double dy = 1 - y_0[i];
          
          if(ifinhorizon(x_0[i],y_0[i],-0.5,1)){
            double cx = 8*(-pow(dy,3)/3-pow(sqrt(delta*delta*h*h-dx*dx),3)/3+dy*(dx*dx+dy*dy-delta*delta*h*h)/2)/pi/pow(delta*h,4);
            double cy = 8*(-pow(dx,3)/3-pow(sqrt(delta*delta*h*h-dy*dy),3)/3+dx*(dx*dx+dy*dy-delta*delta*h*h)/2)/pi/pow(delta*h,4);
            double cxx = ((dx*(2*dx*dx-delta*delta*h*h)*sqrt(delta*delta*h*h-dx*dx)-sqrt(delta*delta*h*h-dy*dy)*(delta*delta*h*h-2*dy*dy)*dy)/pow(delta*h,4)
                +atan(dx/sqrt(delta*delta*h*h-dx*dx))+atan(-sqrt(delta*delta*h*h-dy*dy)/dy)+pi)/pi
                +8*dy*(pow(dx,3)+pow(sqrt(delta*delta*h*h-dy*dy),3))/3/pi/pow(delta*h,4);
            double cyy = ((dy*(2*dy*dy-delta*delta*h*h)*sqrt(delta*delta*h*h-dy*dy)-sqrt(delta*delta*h*h-dx*dx)*(delta*delta*h*h-2*dx*dx)*dx)/pow(delta*h,4)
                +atan(dy/sqrt(delta*delta*h*h-dy*dy))+atan(-sqrt(delta*delta*h*h-dx*dx)/dx)+pi)/pi
                +8*dx*(pow(dy,3)+pow(sqrt(delta*delta*h*h-dx*dx),3))/3/pi/pow(delta*h,4);
            double cxy = pow(dx*dx+dy*dy-delta*delta*h*h,2)/pi/pow(delta*h,4);
            for(int s=0;s<nei[i].size();s++){
              stiffKT[i*dim+nei[i][s]] -= Beta[s*6+1]*cx+Beta[s*6+2]*cy+Beta[s*6+3]*cxx+Beta[s*6+4]*cyy+Beta[s*6+5]*cxy+Beta[s*6+4]*(0.5*(cxx-cyy)+dy*cy-dx*cx)*2;
            }
            stiffKT[i*dim+i] += (0.5*cxx-cx*dx)/Dt;
            double x_proj1 = -0.5;
            double y_proj1 = y_0[i];
            double x_proj2 = x_0[i];
            double y_proj2 = 1;
            Nodeid = search_projection(x_proj1,y_proj1);
            stiffKT[i*dim+Nodeid] += -cx*rbeta; 
            Nodeid = search_projection(x_proj2,y_proj2);
            stiffKT[i*dim+Nodeid] += -cy*rbeta; 
          }
          else{
            double cx = -16*pow(sqrt(delta*delta*h*h-dx*dx),3)/3/pi/pow(delta*h,4);
            double cy = -16*pow(sqrt(delta*delta*h*h-dy*dy),3)/3/pi/pow(delta*h,4);
            double cxx = 2*((dx*(2*dx*dx-delta*delta*h*h)*sqrt(delta*delta*h*h-dx*dx)+dy*(5*delta*delta*h*h-2*dy*dy)*sqrt(delta*delta*h*h-dy*dy)/3)/pow(delta*h,4)
                +atan(dx/sqrt(delta*delta*h*h-dx*dx))+atan(dy/sqrt(delta*delta*h*h-dx*dx)))/pi; 
            double cyy = 2*((dy*(2*dy*dy-delta*delta*h*h)*sqrt(delta*delta*h*h-dy*dy)+dx*(5*delta*delta*h*h-2*dx*dx)*sqrt(delta*delta*h*h-dx*dx)/3)/pow(delta*h,4)
                +atan(dx/sqrt(delta*delta*h*h-dx*dx))+atan(dy/sqrt(delta*delta*h*h-dx*dx)))/pi; 
            for(int s=0;s<nei[i].size();s++){
              stiffKT[i*dim+nei[i][s]] -= Beta[s*6+1]*cx+Beta[s*6+2]*cy+Beta[s*6+3]*cxx+Beta[s*6+4]*cyy+Beta[s*6+4]*(0.5*(cxx-cyy)+dy*cy-dx*cx)*2;
            }
          stiffKT[i*dim+i] += (0.5*cxx-cx*dx)/Dt;
            double x_proj1 = -0.5;
            double y_proj1 = y_0[i];
            double x_proj2 = x_0[i];
            double y_proj2 = 1;
            Nodeid = search_projection(x_proj1,y_proj1);
            stiffKT[i*dim+Nodeid] += -cx*rbeta; 
            Nodeid = search_projection(x_proj2,y_proj2);
            stiffKT[i*dim+Nodeid] += -cy*rbeta; 
          }
        }
        else if(abs(x_0[i]-0.5) <= delta*h+Tol && abs(y_0[i]-1)<=delta*h+Tol){
            normal[0] = 1;
            normal[1] = 0;
            tanvec[0] = 0;
            tanvec[1] = 1;
            for(int s=0;s!=nei[i].size();s++){
              P[s*6] = 1;
              P[s*6+1] = (x_0[nei[i][s]] - x_0[i])*normal[0] + (y_0[nei[i][s]]-y_0[i])*normal[1]; 
              P[s*6+2] = (x_0[nei[i][s]] - x_0[i])*tanvec[0] + (y_0[nei[i][s]]-y_0[i])*tanvec[1];
              P[s*6+3] = pow((x_0[nei[i][s]] - x_0[i])*normal[0] + (y_0[nei[i][s]]-y_0[i])*normal[1],2);
              P[s*6+4] = pow((x_0[nei[i][s]] - x_0[i])*tanvec[0] + (y_0[nei[i][s]]-y_0[i])*tanvec[1],2);
              P[s*6+5] =((x_0[nei[i][s]] - x_0[i])*normal[0] + (y_0[nei[i][s]]-y_0[i])*normal[1])*((x_0[nei[i][s]] - x_0[i])*tanvec[0] + (y_0[nei[i][s]]-y_0[i])*tanvec[1]);
              D[s] =  weight(sqrt(pow(P[s*6+1],2)+pow(P[s*6+2],2)),delta*h);
            }
            for(int m=0;m<6;m++){
              for(int n=0;n<6;n++){
                  M[m*6+n] = 0;
                  for(int s=0;s<nei[i].size();s++){
                      M[m*6+n] += P[s*6+m]*D[s]*P[s*6+n];
                  }
              }
            }
        
          inverse(M,6);
          for(int m=0;m<nei[i].size();m++){
             for(int n=0;n<6;n++){
                  Beta[m*6+n] = 0;
                  for(int s=0;s<6;s++){
                      Beta[m*6+n] += D[m]*P[m*6+s]*M[s*6+n];
                  }
              }
          }
          double dx = 0.5 - x_0[i];
          double dy = 1 - y_0[i];
          
          if(ifinhorizon(x_0[i],y_0[i],0.5,1)){
            double cx = 8*(-pow(dy,3)/3-pow(sqrt(delta*delta*h*h-dx*dx),3)/3+dy*(dx*dx+dy*dy-delta*delta*h*h)/2)/pi/pow(delta*h,4);
            double cy = 8*(-pow(dx,3)/3-pow(sqrt(delta*delta*h*h-dy*dy),3)/3+dx*(dx*dx+dy*dy-delta*delta*h*h)/2)/pi/pow(delta*h,4);
            double cxx = ((dx*(2*dx*dx-delta*delta*h*h)*sqrt(delta*delta*h*h-dx*dx)-sqrt(delta*delta*h*h-dy*dy)*(delta*delta*h*h-2*dy*dy)*dy)/pow(delta*h,4)
                +atan(dx/sqrt(delta*delta*h*h-dx*dx))+atan(-sqrt(delta*delta*h*h-dy*dy)/dy)+pi)/pi
                +8*dy*(pow(dx,3)+pow(sqrt(delta*delta*h*h-dy*dy),3))/3/pi/pow(delta*h,4);
            double cyy = ((dy*(2*dy*dy-delta*delta*h*h)*sqrt(delta*delta*h*h-dy*dy)-sqrt(delta*delta*h*h-dx*dx)*(delta*delta*h*h-2*dx*dx)*dx)/pow(delta*h,4)
                +atan(dy/sqrt(delta*delta*h*h-dy*dy))+atan(-sqrt(delta*delta*h*h-dx*dx)/dx)+pi)/pi
                +8*dx*(pow(dy,3)+pow(sqrt(delta*delta*h*h-dx*dx),3))/3/pi/pow(delta*h,4);
            double cxy = pow(dx*dx+dy*dy-delta*delta*h*h,2)/pi/pow(delta*h,4);
            for(int s=0;s<nei[i].size();s++){
              stiffKT[i*dim+nei[i][s]] -= Beta[s*6+1]*cx+Beta[s*6+2]*cy+Beta[s*6+3]*cxx+Beta[s*6+4]*cyy+Beta[s*6+5]*cxy+Beta[s*6+4]*(0.5*(cxx-cyy)+dy*cy-dx*cx)*2;
            }
            stiffKT[i*dim+i] += (0.5*cxx-cx*dx)/Dt;
            double x_proj1 = 0.5;
            double y_proj1 = y_0[i];
            double x_proj2 = x_0[i];
            double y_proj2 = 1;
            Nodeid = search_projection(x_proj1,y_proj1);
            stiffKT[i*dim+Nodeid] += -cx*rbeta; 
            Nodeid = search_projection(x_proj2,y_proj2);
            stiffKT[i*dim+Nodeid] += -cy*rbeta; 
          }
          else{
            double cx = -16*pow(sqrt(delta*delta*h*h-dx*dx),3)/3/pi/pow(delta*h,4);
            double cy = -16*pow(sqrt(delta*delta*h*h-dy*dy),3)/3/pi/pow(delta*h,4);
            double cxx = 2*((dx*(2*dx*dx-delta*delta*h*h)*sqrt(delta*delta*h*h-dx*dx)+dy*(5*delta*delta*h*h-2*dy*dy)*sqrt(delta*delta*h*h-dy*dy)/3)/pow(delta*h,4)
                +atan(dx/sqrt(delta*delta*h*h-dx*dx))+atan(dy/sqrt(delta*delta*h*h-dx*dx)))/pi; 
            double cyy = 2*((dy*(2*dy*dy-delta*delta*h*h)*sqrt(delta*delta*h*h-dy*dy)+dx*(5*delta*delta*h*h-2*dx*dx)*sqrt(delta*delta*h*h-dx*dx)/3)/pow(delta*h,4)
                +atan(dx/sqrt(delta*delta*h*h-dx*dx))+atan(dy/sqrt(delta*delta*h*h-dx*dx)))/pi; 
            for(int s=0;s<nei[i].size();s++){
              stiffKT[i*dim+nei[i][s]] -= Beta[s*6+1]*cx+Beta[s*6+2]*cy+Beta[s*6+3]*cxx+Beta[s*6+4]*cyy+Beta[s*6+4]*(0.5*(cxx-cyy)+dy*cy-dx*cx)*2;
            }
            stiffKT[i*dim+i] += (0.5*cxx-cx*dx)/Dt;
            double x_proj1 = 0.5;
            double y_proj1 = y_0[i];
            double x_proj2 = x_0[i];
            double y_proj2 = 1;
            Nodeid = search_projection(x_proj1,y_proj1);
            stiffKT[i*dim+Nodeid] += -cx*rbeta; 
            Nodeid = search_projection(x_proj2,y_proj2);
            stiffKT[i*dim+Nodeid] += -cy*rbeta; 
          }
        }
        else if(abs(x_0[i]-1) <= delta*h+Tol && abs(y_0[i]-0.5)<=delta*h+Tol){
            normal[0] = 1;
            normal[1] = 0;
            tanvec[0] = 0;
            tanvec[1] = 1;
            for(int s=0;s!=nei[i].size();s++){
              P[s*6] = 1;
              P[s*6+1] = (x_0[nei[i][s]] - x_0[i])*normal[0] + (y_0[nei[i][s]]-y_0[i])*normal[1]; 
              P[s*6+2] = (x_0[nei[i][s]] - x_0[i])*tanvec[0] + (y_0[nei[i][s]]-y_0[i])*tanvec[1];
              P[s*6+3] = pow((x_0[nei[i][s]] - x_0[i])*normal[0] + (y_0[nei[i][s]]-y_0[i])*normal[1],2);
              P[s*6+4] = pow((x_0[nei[i][s]] - x_0[i])*tanvec[0] + (y_0[nei[i][s]]-y_0[i])*tanvec[1],2);
              P[s*6+5] =((x_0[nei[i][s]] - x_0[i])*normal[0] + (y_0[nei[i][s]]-y_0[i])*normal[1])*((x_0[nei[i][s]] - x_0[i])*tanvec[0] + (y_0[nei[i][s]]-y_0[i])*tanvec[1]);
              D[s] =  weight(sqrt(pow(P[s*6+1],2)+pow(P[s*6+2],2)),delta*h);
            }
            for(int m=0;m<6;m++){
              for(int n=0;n<6;n++){
                  M[m*6+n] = 0;
                  for(int s=0;s<nei[i].size();s++){
                      M[m*6+n] += P[s*6+m]*D[s]*P[s*6+n];
                  }
              }
            }
        
          inverse(M,6);
          for(int m=0;m<nei[i].size();m++){
             for(int n=0;n<6;n++){
                  Beta[m*6+n] = 0;
                  for(int s=0;s<6;s++){
                      Beta[m*6+n] += D[m]*P[m*6+s]*M[s*6+n];
                  }
              }
          }
          double dx = 1 - x_0[i];
          double dy = 0.5 - y_0[i];
          
          if(ifinhorizon(x_0[i],y_0[i],1,0.5)){
            double cx = 8*(-pow(dy,3)/3-pow(sqrt(delta*delta*h*h-dx*dx),3)/3+dy*(dx*dx+dy*dy-delta*delta*h*h)/2)/pi/pow(delta*h,4);
            double cy = 8*(-pow(dx,3)/3-pow(sqrt(delta*delta*h*h-dy*dy),3)/3+dx*(dx*dx+dy*dy-delta*delta*h*h)/2)/pi/pow(delta*h,4);
            double cxx = ((dx*(2*dx*dx-delta*delta*h*h)*sqrt(delta*delta*h*h-dx*dx)-sqrt(delta*delta*h*h-dy*dy)*(delta*delta*h*h-2*dy*dy)*dy)/pow(delta*h,4)
                +atan(dx/sqrt(delta*delta*h*h-dx*dx))+atan(-sqrt(delta*delta*h*h-dy*dy)/dy)+pi)/pi
                +8*dy*(pow(dx,3)+pow(sqrt(delta*delta*h*h-dy*dy),3))/3/pi/pow(delta*h,4);
            double cyy = ((dy*(2*dy*dy-delta*delta*h*h)*sqrt(delta*delta*h*h-dy*dy)-sqrt(delta*delta*h*h-dx*dx)*(delta*delta*h*h-2*dx*dx)*dx)/pow(delta*h,4)
                +atan(dy/sqrt(delta*delta*h*h-dy*dy))+atan(-sqrt(delta*delta*h*h-dx*dx)/dx)+pi)/pi
                +8*dx*(pow(dy,3)+pow(sqrt(delta*delta*h*h-dx*dx),3))/3/pi/pow(delta*h,4);
            double cxy = pow(dx*dx+dy*dy-delta*delta*h*h,2)/pi/pow(delta*h,4);
            for(int s=0;s<nei[i].size();s++){
              stiffKT[i*dim+nei[i][s]] -= Beta[s*6+1]*cx+Beta[s*6+2]*cy+Beta[s*6+3]*cxx+Beta[s*6+4]*cyy+Beta[s*6+5]*cxy+Beta[s*6+4]*(0.5*(cxx-cyy)+dy*cy-dx*cx)*2;
            }
            stiffKT[i*dim+i] += (0.5*cxx-cx*dx)/Dt;
            double x_proj1 = 1;
            double y_proj1 = y_0[i];
            double x_proj2 = x_0[i];
            double y_proj2 = 0.5;
            Nodeid = search_projection(x_proj1,y_proj1);
            stiffKT[i*dim+Nodeid] += -cx*rbeta; 
            Nodeid = search_projection(x_proj2,y_proj2);
            stiffKT[i*dim+Nodeid] += -cy*rbeta; 
          }
          else{
            double cx = -16*pow(sqrt(delta*delta*h*h-dx*dx),3)/3/pi/pow(delta*h,4);
            double cy = -16*pow(sqrt(delta*delta*h*h-dy*dy),3)/3/pi/pow(delta*h,4);
            double cxx = 2*((dx*(2*dx*dx-delta*delta*h*h)*sqrt(delta*delta*h*h-dx*dx)+dy*(5*delta*delta*h*h-2*dy*dy)*sqrt(delta*delta*h*h-dy*dy)/3)/pow(delta*h,4)
                +atan(dx/sqrt(delta*delta*h*h-dx*dx))+atan(dy/sqrt(delta*delta*h*h-dx*dx)))/pi; 
            double cyy = 2*((dy*(2*dy*dy-delta*delta*h*h)*sqrt(delta*delta*h*h-dy*dy)+dx*(5*delta*delta*h*h-2*dx*dx)*sqrt(delta*delta*h*h-dx*dx)/3)/pow(delta*h,4)
                +atan(dx/sqrt(delta*delta*h*h-dx*dx))+atan(dy/sqrt(delta*delta*h*h-dx*dx)))/pi; 
            for(int s=0;s<nei[i].size();s++){
              stiffKT[i*dim+nei[i][s]] -= Beta[s*6+1]*cx+Beta[s*6+2]*cy+Beta[s*6+3]*cxx+Beta[s*6+4]*cyy+Beta[s*6+4]*(0.5*(cxx-cyy)+dy*cy-dx*cx)*2;
            }
            stiffKT[i*dim+i] += (0.5*cxx-cx*dx)/Dt;
            double x_proj1 = 1;
            double y_proj1 = y_0[i];
            double x_proj2 = x_0[i];
            double y_proj2 = 0.5;
            Nodeid = search_projection(x_proj1,y_proj1);
            stiffKT[i*dim+Nodeid] += -cx*rbeta; 
            Nodeid = search_projection(x_proj2,y_proj2);
            stiffKT[i*dim+Nodeid] += -cy*rbeta; 
          }
        }
        else if(abs(x_0[i]-1) <= delta*h+Tol && abs(y_0[i]+0.5)<=delta*h+Tol){
            normal[0] = 1;
            normal[1] = 0;
            tanvec[0] = 0;
            tanvec[1] = -1;
            for(int s=0;s!=nei[i].size();s++){
              P[s*6] = 1;
              P[s*6+1] = (x_0[nei[i][s]] - x_0[i])*normal[0] + (y_0[nei[i][s]]-y_0[i])*normal[1]; 
              P[s*6+2] = (x_0[nei[i][s]] - x_0[i])*tanvec[0] + (y_0[nei[i][s]]-y_0[i])*tanvec[1];
              P[s*6+3] = pow((x_0[nei[i][s]] - x_0[i])*normal[0] + (y_0[nei[i][s]]-y_0[i])*normal[1],2);
              P[s*6+4] = pow((x_0[nei[i][s]] - x_0[i])*tanvec[0] + (y_0[nei[i][s]]-y_0[i])*tanvec[1],2);
              P[s*6+5] =((x_0[nei[i][s]] - x_0[i])*normal[0] + (y_0[nei[i][s]]-y_0[i])*normal[1])*((x_0[nei[i][s]] - x_0[i])*tanvec[0] + (y_0[nei[i][s]]-y_0[i])*tanvec[1]);
              D[s] =  weight(sqrt(pow(P[s*6+1],2)+pow(P[s*6+2],2)),delta*h);
            }
            for(int m=0;m<6;m++){
              for(int n=0;n<6;n++){
                  M[m*6+n] = 0;
                  for(int s=0;s<nei[i].size();s++){
                      M[m*6+n] += P[s*6+m]*D[s]*P[s*6+n];
                  }
              }
            }
        
          inverse(M,6);
          for(int m=0;m<nei[i].size();m++){
             for(int n=0;n<6;n++){
                  Beta[m*6+n] = 0;
                  for(int s=0;s<6;s++){
                      Beta[m*6+n] += D[m]*P[m*6+s]*M[s*6+n];
                  }
              }
          }
          double dx = 1 - x_0[i];
          double dy = 0.5 + y_0[i];
          
          if(ifinhorizon(x_0[i],y_0[i],1,-0.5)){
            double cx = 8*(-pow(dy,3)/3-pow(sqrt(delta*delta*h*h-dx*dx),3)/3+dy*(dx*dx+dy*dy-delta*delta*h*h)/2)/pi/pow(delta*h,4);
            double cy = 8*(-pow(dx,3)/3-pow(sqrt(delta*delta*h*h-dy*dy),3)/3+dx*(dx*dx+dy*dy-delta*delta*h*h)/2)/pi/pow(delta*h,4);
            double cxx = ((dx*(2*dx*dx-delta*delta*h*h)*sqrt(delta*delta*h*h-dx*dx)-sqrt(delta*delta*h*h-dy*dy)*(delta*delta*h*h-2*dy*dy)*dy)/pow(delta*h,4)
                +atan(dx/sqrt(delta*delta*h*h-dx*dx))+atan(-sqrt(delta*delta*h*h-dy*dy)/dy)+pi)/pi
                +8*dy*(pow(dx,3)+pow(sqrt(delta*delta*h*h-dy*dy),3))/3/pi/pow(delta*h,4);
            double cyy = ((dy*(2*dy*dy-delta*delta*h*h)*sqrt(delta*delta*h*h-dy*dy)-sqrt(delta*delta*h*h-dx*dx)*(delta*delta*h*h-2*dx*dx)*dx)/pow(delta*h,4)
                +atan(dy/sqrt(delta*delta*h*h-dy*dy))+atan(-sqrt(delta*delta*h*h-dx*dx)/dx)+pi)/pi
                +8*dx*(pow(dy,3)+pow(sqrt(delta*delta*h*h-dx*dx),3))/3/pi/pow(delta*h,4);
            double cxy = pow(dx*dx+dy*dy-delta*delta*h*h,2)/pi/pow(delta*h,4);
            for(int s=0;s<nei[i].size();s++){
              stiffKT[i*dim+nei[i][s]] -= Beta[s*6+1]*cx+Beta[s*6+2]*cy+Beta[s*6+3]*cxx+Beta[s*6+4]*cyy+Beta[s*6+5]*cxy+Beta[s*6+4]*(0.5*(cxx-cyy)+dy*cy-dx*cx)*2;
            }
            stiffKT[i*dim+i] += (0.5*cxx-cx*dx)/Dt;
            double x_proj1 = 1;
            double y_proj1 = y_0[i];
            double x_proj2 = x_0[i];
            double y_proj2 = -0.5;
            Nodeid = search_projection(x_proj1,y_proj1);
            stiffKT[i*dim+Nodeid] += -cx*rbeta; 
            Nodeid = search_projection(x_proj2,y_proj2);
            stiffKT[i*dim+Nodeid] += -cy*rbeta; 
          }
          else{
            double cx = -16*pow(sqrt(delta*delta*h*h-dx*dx),3)/3/pi/pow(delta*h,4);
            double cy = -16*pow(sqrt(delta*delta*h*h-dy*dy),3)/3/pi/pow(delta*h,4);
            double cxx = 2*((dx*(2*dx*dx-delta*delta*h*h)*sqrt(delta*delta*h*h-dx*dx)+dy*(5*delta*delta*h*h-2*dy*dy)*sqrt(delta*delta*h*h-dy*dy)/3)/pow(delta*h,4)
                +atan(dx/sqrt(delta*delta*h*h-dx*dx))+atan(dy/sqrt(delta*delta*h*h-dx*dx)))/pi; 
            double cyy = 2*((dy*(2*dy*dy-delta*delta*h*h)*sqrt(delta*delta*h*h-dy*dy)+dx*(5*delta*delta*h*h-2*dx*dx)*sqrt(delta*delta*h*h-dx*dx)/3)/pow(delta*h,4)
                +atan(dx/sqrt(delta*delta*h*h-dx*dx))+atan(dy/sqrt(delta*delta*h*h-dx*dx)))/pi; 
            for(int s=0;s<nei[i].size();s++){
              stiffKT[i*dim+nei[i][s]] -= Beta[s*6+1]*cx+Beta[s*6+2]*cy+Beta[s*6+3]*cxx+Beta[s*6+4]*cyy+Beta[s*6+4]*(0.5*(cxx-cyy)+dy*cy-dx*cx)*2;
            }
            stiffKT[i*dim+i] += (0.5*cxx-cx*dx)/Dt;
            double x_proj1 = 1;
            double y_proj1 = y_0[i];
            double x_proj2 = x_0[i];
            double y_proj2 = -0.5;
            Nodeid = search_projection(x_proj1,y_proj1);
            stiffKT[i*dim+Nodeid] += -cx*rbeta; 
            Nodeid = search_projection(x_proj2,y_proj2);
            stiffKT[i*dim+Nodeid] += -cy*rbeta; 
          }
        }
        else if(abs(x_0[i]+0.5) <= delta*h+Tol && abs(y_0[i]+1)<=delta*h+Tol){
            normal[0] = -1;
            normal[1] = 0;
            tanvec[0] = 0;
            tanvec[1] = -1;
            for(int s=0;s!=nei[i].size();s++){
              P[s*6] = 1;
              P[s*6+1] = (x_0[nei[i][s]] - x_0[i])*normal[0] + (y_0[nei[i][s]]-y_0[i])*normal[1]; 
              P[s*6+2] = (x_0[nei[i][s]] - x_0[i])*tanvec[0] + (y_0[nei[i][s]]-y_0[i])*tanvec[1];
              P[s*6+3] = pow((x_0[nei[i][s]] - x_0[i])*normal[0] + (y_0[nei[i][s]]-y_0[i])*normal[1],2);
              P[s*6+4] = pow((x_0[nei[i][s]] - x_0[i])*tanvec[0] + (y_0[nei[i][s]]-y_0[i])*tanvec[1],2);
              P[s*6+5] =((x_0[nei[i][s]] - x_0[i])*normal[0] + (y_0[nei[i][s]]-y_0[i])*normal[1])*((x_0[nei[i][s]] - x_0[i])*tanvec[0] + (y_0[nei[i][s]]-y_0[i])*tanvec[1]);
              D[s] =  weight(sqrt(pow(P[s*6+1],2)+pow(P[s*6+2],2)),delta*h);
            }
            for(int m=0;m<6;m++){
              for(int n=0;n<6;n++){
                  M[m*6+n] = 0;
                  for(int s=0;s<nei[i].size();s++){
                      M[m*6+n] += P[s*6+m]*D[s]*P[s*6+n];
                  }
              }
            }
        
          inverse(M,6);
          for(int m=0;m<nei[i].size();m++){
             for(int n=0;n<6;n++){
                  Beta[m*6+n] = 0;
                  for(int s=0;s<6;s++){
                      Beta[m*6+n] += D[m]*P[m*6+s]*M[s*6+n];
                  }
              }
          }
          double dx = 0.5 + x_0[i];
          double dy = 1 + y_0[i];
          
          if(ifinhorizon(x_0[i],y_0[i],-0.5,-1)){
            double cx = 8*(-pow(dy,3)/3-pow(sqrt(delta*delta*h*h-dx*dx),3)/3+dy*(dx*dx+dy*dy-delta*delta*h*h)/2)/pi/pow(delta*h,4);
            double cy = 8*(-pow(dx,3)/3-pow(sqrt(delta*delta*h*h-dy*dy),3)/3+dx*(dx*dx+dy*dy-delta*delta*h*h)/2)/pi/pow(delta*h,4);
            double cxx = ((dx*(2*dx*dx-delta*delta*h*h)*sqrt(delta*delta*h*h-dx*dx)-sqrt(delta*delta*h*h-dy*dy)*(delta*delta*h*h-2*dy*dy)*dy)/pow(delta*h,4)
                +atan(dx/sqrt(delta*delta*h*h-dx*dx))+atan(-sqrt(delta*delta*h*h-dy*dy)/dy)+pi)/pi
                +8*dy*(pow(dx,3)+pow(sqrt(delta*delta*h*h-dy*dy),3))/3/pi/pow(delta*h,4);
            double cyy = ((dy*(2*dy*dy-delta*delta*h*h)*sqrt(delta*delta*h*h-dy*dy)-sqrt(delta*delta*h*h-dx*dx)*(delta*delta*h*h-2*dx*dx)*dx)/pow(delta*h,4)
                +atan(dy/sqrt(delta*delta*h*h-dy*dy))+atan(-sqrt(delta*delta*h*h-dx*dx)/dx)+pi)/pi
                +8*dx*(pow(dy,3)+pow(sqrt(delta*delta*h*h-dx*dx),3))/3/pi/pow(delta*h,4);
            double cxy = pow(dx*dx+dy*dy-delta*delta*h*h,2)/pi/pow(delta*h,4);
            for(int s=0;s<nei[i].size();s++){
              stiffKT[i*dim+nei[i][s]] -= Beta[s*6+1]*cx+Beta[s*6+2]*cy+Beta[s*6+3]*cxx+Beta[s*6+4]*cyy+Beta[s*6+5]*cxy+Beta[s*6+4]*(0.5*(cxx-cyy)+dy*cy-dx*cx)*2;
            }
            stiffKT[i*dim+i] += (0.5*cxx-cx*dx)/Dt;
            double x_proj1 = -0.5;
            double y_proj1 = y_0[i];
            double x_proj2 = x_0[i];
            double y_proj2 = -1;
            Nodeid = search_projection(x_proj1,y_proj1);
            stiffKT[i*dim+Nodeid] += -cx*rbeta; 
            Nodeid = search_projection(x_proj2,y_proj2);
            stiffKT[i*dim+Nodeid] += -cy*rbeta; 
          }
          else{
            double cx = -16*pow(sqrt(delta*delta*h*h-dx*dx),3)/3/pi/pow(delta*h,4);
            double cy = -16*pow(sqrt(delta*delta*h*h-dy*dy),3)/3/pi/pow(delta*h,4);
            double cxx = 2*((dx*(2*dx*dx-delta*delta*h*h)*sqrt(delta*delta*h*h-dx*dx)+dy*(5*delta*delta*h*h-2*dy*dy)*sqrt(delta*delta*h*h-dy*dy)/3)/pow(delta*h,4)
                +atan(dx/sqrt(delta*delta*h*h-dx*dx))+atan(dy/sqrt(delta*delta*h*h-dx*dx)))/pi; 
            double cyy = 2*((dy*(2*dy*dy-delta*delta*h*h)*sqrt(delta*delta*h*h-dy*dy)+dx*(5*delta*delta*h*h-2*dx*dx)*sqrt(delta*delta*h*h-dx*dx)/3)/pow(delta*h,4)
                +atan(dx/sqrt(delta*delta*h*h-dx*dx))+atan(dy/sqrt(delta*delta*h*h-dx*dx)))/pi; 
            for(int s=0;s<nei[i].size();s++){
              stiffKT[i*dim+nei[i][s]] -= Beta[s*6+1]*cx+Beta[s*6+2]*cy+Beta[s*6+3]*cxx+Beta[s*6+4]*cyy+Beta[s*6+4]*(0.5*(cxx-cyy)+dy*cy-dx*cx)*2;
            }
          stiffKT[i*dim+i] += (0.5*cxx-cx*dx)/Dt;
            double x_proj1 = -0.5;
            double y_proj1 = y_0[i];
            double x_proj2 = x_0[i];
            double y_proj2 = -1;
            Nodeid = search_projection(x_proj1,y_proj1);
            stiffKT[i*dim+Nodeid] += -cx*rbeta; 
            Nodeid = search_projection(x_proj2,y_proj2);
            stiffKT[i*dim+Nodeid] += -cy*rbeta; 
          }
        }
        else if(abs(x_0[i]-0.5) <= delta*h+Tol && abs(y_0[i]+1)<=delta*h+Tol){
            normal[0] = 1;
            normal[1] = 0;
            tanvec[0] = 0;
            tanvec[1] = -1;
            for(int s=0;s!=nei[i].size();s++){
              P[s*6] = 1;
              P[s*6+1] = (x_0[nei[i][s]] - x_0[i])*normal[0] + (y_0[nei[i][s]]-y_0[i])*normal[1]; 
              P[s*6+2] = (x_0[nei[i][s]] - x_0[i])*tanvec[0] + (y_0[nei[i][s]]-y_0[i])*tanvec[1];
              P[s*6+3] = pow((x_0[nei[i][s]] - x_0[i])*normal[0] + (y_0[nei[i][s]]-y_0[i])*normal[1],2);
              P[s*6+4] = pow((x_0[nei[i][s]] - x_0[i])*tanvec[0] + (y_0[nei[i][s]]-y_0[i])*tanvec[1],2);
              P[s*6+5] =((x_0[nei[i][s]] - x_0[i])*normal[0] + (y_0[nei[i][s]]-y_0[i])*normal[1])*((x_0[nei[i][s]] - x_0[i])*tanvec[0] + (y_0[nei[i][s]]-y_0[i])*tanvec[1]);
              D[s] =  weight(sqrt(pow(P[s*6+1],2)+pow(P[s*6+2],2)),delta*h);
            }
            for(int m=0;m<6;m++){
              for(int n=0;n<6;n++){
                  M[m*6+n] = 0;
                  for(int s=0;s<nei[i].size();s++){
                      M[m*6+n] += P[s*6+m]*D[s]*P[s*6+n];
                  }
              }
            }
        
          inverse(M,6);
          for(int m=0;m<nei[i].size();m++){
             for(int n=0;n<6;n++){
                  Beta[m*6+n] = 0;
                  for(int s=0;s<6;s++){
                      Beta[m*6+n] += D[m]*P[m*6+s]*M[s*6+n];
                  }
              }
          }
          double dx = 0.5 - x_0[i];
          double dy = 1 + y_0[i];
          
          if(ifinhorizon(x_0[i],y_0[i],0.5,-1)){
            double cx = 8*(-pow(dy,3)/3-pow(sqrt(delta*delta*h*h-dx*dx),3)/3+dy*(dx*dx+dy*dy-delta*delta*h*h)/2)/pi/pow(delta*h,4);
            double cy = 8*(-pow(dx,3)/3-pow(sqrt(delta*delta*h*h-dy*dy),3)/3+dx*(dx*dx+dy*dy-delta*delta*h*h)/2)/pi/pow(delta*h,4);
            double cxx = ((dx*(2*dx*dx-delta*delta*h*h)*sqrt(delta*delta*h*h-dx*dx)-sqrt(delta*delta*h*h-dy*dy)*(delta*delta*h*h-2*dy*dy)*dy)/pow(delta*h,4)
                +atan(dx/sqrt(delta*delta*h*h-dx*dx))+atan(-sqrt(delta*delta*h*h-dy*dy)/dy)+pi)/pi
                +8*dy*(pow(dx,3)+pow(sqrt(delta*delta*h*h-dy*dy),3))/3/pi/pow(delta*h,4);
            double cyy = ((dy*(2*dy*dy-delta*delta*h*h)*sqrt(delta*delta*h*h-dy*dy)-sqrt(delta*delta*h*h-dx*dx)*(delta*delta*h*h-2*dx*dx)*dx)/pow(delta*h,4)
                +atan(dy/sqrt(delta*delta*h*h-dy*dy))+atan(-sqrt(delta*delta*h*h-dx*dx)/dx)+pi)/pi
                +8*dx*(pow(dy,3)+pow(sqrt(delta*delta*h*h-dx*dx),3))/3/pi/pow(delta*h,4);
            double cxy = pow(dx*dx+dy*dy-delta*delta*h*h,2)/pi/pow(delta*h,4);
            for(int s=0;s<nei[i].size();s++){
              stiffKT[i*dim+nei[i][s]] -= Beta[s*6+1]*cx+Beta[s*6+2]*cy+Beta[s*6+3]*cxx+Beta[s*6+4]*cyy+Beta[s*6+5]*cxy+Beta[s*6+4]*(0.5*(cxx-cyy)+dy*cy-dx*cx)*2;
            }
            stiffKT[i*dim+i] += (0.5*cxx-cx*dx)/Dt;
            double x_proj1 = 0.5;
            double y_proj1 = y_0[i];
            double x_proj2 = x_0[i];
            double y_proj2 = -1;
            Nodeid = search_projection(x_proj1,y_proj1);
            stiffKT[i*dim+Nodeid] += -cx*rbeta; 
            Nodeid = search_projection(x_proj2,y_proj2);
            stiffKT[i*dim+Nodeid] += -cy*rbeta; 
          }
          else{
            double cx = -16*pow(sqrt(delta*delta*h*h-dx*dx),3)/3/pi/pow(delta*h,4);
            double cy = -16*pow(sqrt(delta*delta*h*h-dy*dy),3)/3/pi/pow(delta*h,4);
            double cxx = 2*((dx*(2*dx*dx-delta*delta*h*h)*sqrt(delta*delta*h*h-dx*dx)+dy*(5*delta*delta*h*h-2*dy*dy)*sqrt(delta*delta*h*h-dy*dy)/3)/pow(delta*h,4)
                +atan(dx/sqrt(delta*delta*h*h-dx*dx))+atan(dy/sqrt(delta*delta*h*h-dx*dx)))/pi; 
            double cyy = 2*((dy*(2*dy*dy-delta*delta*h*h)*sqrt(delta*delta*h*h-dy*dy)+dx*(5*delta*delta*h*h-2*dx*dx)*sqrt(delta*delta*h*h-dx*dx)/3)/pow(delta*h,4)
                +atan(dx/sqrt(delta*delta*h*h-dx*dx))+atan(dy/sqrt(delta*delta*h*h-dx*dx)))/pi; 
            for(int s=0;s<nei[i].size();s++){
              stiffKT[i*dim+nei[i][s]] -= Beta[s*6+1]*cx+Beta[s*6+2]*cy+Beta[s*6+3]*cxx+Beta[s*6+4]*cyy+Beta[s*6+4]*(0.5*(cxx-cyy)+dy*cy-dx*cx)*2;
            }
            stiffKT[i*dim+i] += (0.5*cxx-cx*dx)/Dt;
            double x_proj1 = 0.5;
            double y_proj1 = y_0[i];
            double x_proj2 = x_0[i];
            double y_proj2 = -1;
            Nodeid = search_projection(x_proj1,y_proj1);
            stiffKT[i*dim+Nodeid] += -cx*rbeta; 
            Nodeid = search_projection(x_proj2,y_proj2);
            stiffKT[i*dim+Nodeid] += -cy*rbeta; 
          }
        }
        else if(ifinhorizon(x_0[i],y_0[i],-0.5,0.5)){
            normal[0] = -1;
            normal[1] = 0;
            tanvec[0] = 0;
            tanvec[1] = 1;
            for(int s=0;s!=nei[i].size();s++){
              P[s*6] = 1;
              P[s*6+1] = (x_0[nei[i][s]] - x_0[i])*normal[0] + (y_0[nei[i][s]]-y_0[i])*normal[1]; 
              P[s*6+2] = (x_0[nei[i][s]] - x_0[i])*tanvec[0] + (y_0[nei[i][s]]-y_0[i])*tanvec[1];
              P[s*6+3] = pow((x_0[nei[i][s]] - x_0[i])*normal[0] + (y_0[nei[i][s]]-y_0[i])*normal[1],2);
              P[s*6+4] = pow((x_0[nei[i][s]] - x_0[i])*tanvec[0] + (y_0[nei[i][s]]-y_0[i])*tanvec[1],2);
              P[s*6+5] =((x_0[nei[i][s]] - x_0[i])*normal[0] + (y_0[nei[i][s]]-y_0[i])*normal[1])*((x_0[nei[i][s]] - x_0[i])*tanvec[0] + (y_0[nei[i][s]]-y_0[i])*tanvec[1]);
              D[s] =  weight(sqrt(pow(P[s*6+1],2)+pow(P[s*6+2],2)),delta*h);
            }
            for(int m=0;m<6;m++){
              for(int n=0;n<6;n++){
                  M[m*6+n] = 0;
                  for(int s=0;s<nei[i].size();s++){
                      M[m*6+n] += P[s*6+m]*D[s]*P[s*6+n];
                  }
              }
            }
        
          inverse(M,6);
          for(int m=0;m<nei[i].size();m++){
             for(int n=0;n<6;n++){
                  Beta[m*6+n] = 0;
                  for(int s=0;s<6;s++){
                      Beta[m*6+n] += D[m]*P[m*6+s]*M[s*6+n];
                  }
              }
          }

          double dx = 0.5 + x_0[i];
          double dy = 0.5 - y_0[i];
          if(dx > 0 && dy > 0){
            double cx = 8*(-1*pow(sqrt(delta*delta*h*h-dx*dx),3)/3+dy*(delta*delta*h*h-dy*dy-dx*dx)/2+pow(dy,3)/3)/pi/pow(delta*h,4);
            double cy = 8*(-1*pow(sqrt(delta*delta*h*h-dy*dy),3)/3+dx*(delta*delta*h*h-dy*dy-dx*dx)/2+pow(dx,3)/3)/pi/pow(delta*h,4);
            double cxx = (dx*(2*dx*dx-delta*delta*h*h)*sqrt(delta*delta*h*h-dx*dx)/pow(delta*h,4)+dy*(2*dy*dy-delta*delta*h*h)*sqrt(delta*delta*h*h-dy*dy)/pow(delta*h,4)
                +atan(dx/sqrt(delta*delta*h*h-dx*dx))+atan(dy/sqrt(delta*delta*h*h-dy*dy)+2*pi))/pi + 8*dy*(pow(sqrt(delta*delta*h*h-dy*dy),3)-pow(dx,3))/3/pi/pow(delta*h,4);  
            double cyy = (dx*(2*dx*dx-delta*delta*h*h)*sqrt(delta*delta*h*h-dx*dx)/pow(delta*h,4)+dy*(2*dy*dy-delta*delta*h*h)*sqrt(delta*delta*h*h-dy*dy)/pow(delta*h,4)
                +atan(dx/sqrt(delta*delta*h*h-dx*dx))+atan(dy/sqrt(delta*delta*h*h-dy*dy)+2*pi))/pi + 8*dx*(pow(sqrt(delta*delta*h*h-dx*dx),3)-pow(dy,3))/3/pi/pow(delta*h,4);  
            double cxy = -1*pow(delta*delta*h*h-dx*dx-dy*dy,2)/pi/pow(delta*h,4);
          for(int s=0;s<nei[i].size();s++){
            stiffKT[i*dim+nei[i][s]] -= Beta[s*6+1]*cx+Beta[s*6+2]*cy+Beta[s*6+3]*cxx+Beta[s*6+4]*cyy+Beta[s*6+5]*cxy+Beta[s*6+4]*(0.5*(cxx-cyy)-cx*dx+cy*dy)*2;
          }
            stiffKT[i*dim+i] += (0.5*cxx-cx*dx)/Dt;
            double x_proj1 = -0.5;
            double y_proj1 = 0.5;
            double x_proj2 = -0.5;
            double y_proj2 = 0.5;
            Nodeid = search_projection(x_proj1,y_proj1);
            stiffKT[i*dim+Nodeid] += -cx*rbeta; 
            Nodeid = search_projection(x_proj2,y_proj2);
            stiffKT[i*dim+Nodeid] += -cy*rbeta; 
          }
          if(dx >= 0-Tol && dy <= 0){
            double cx = 8*(-1*pow(sqrt(delta*delta*h*h-dx*dx),3)/3+(delta*delta*h*h-dx*dx)*dy/2)/pi/pow(delta*h,4);
            double cy = 8*((dy*dy-delta*delta*h*h)*(delta*h-dx)/2+pow(delta*h,3)/6-pow(dx,3)/6)/pi/pow(delta*h,4);
            double cxx = (dx*(2*dx*dx-delta*delta*h*h)*sqrt(delta*delta*h*h-dx*dx)/pow(delta*h,4)+atan(dx/sqrt(delta*delta*h*h-dx*dx)+3*pi/2))/pi
              +8*dy*(pow(delta*delta*h*h,3)-pow(dx,3))/3/pi/pow(delta*h,4);
            double cyy = (dx*(5*delta*delta*h*h-2*dx*dx)*sqrt(delta*delta*h*h-dx*dx)/3/pow(delta*h,4)+atan(dx/sqrt(delta*delta*h*h-dx*dx))+3*pi/2)/pi
              +8*pow(dy,3)*(delta*h-dx)/3/pi/pow(delta*h,4);
            double cxy = (2*(dy*dy-delta*delta*h*h)*(delta*delta*h*h-dx*dx)+pow(delta*h,4)-pow(dx,4))/pi/pow(delta*h,4);
          for(int s=0;s<nei[i].size();s++){
            stiffKT[i*dim+nei[i][s]] -= Beta[s*6+1]*cx+Beta[s*6+2]*cy+Beta[s*6+3]*cxx+Beta[s*6+4]*cyy+Beta[s*6+5]*cxy+Beta[s*6+4]*(0.5*(cxx-cyy)-cx*dx+cy*dy)*2;
          }
            stiffKT[i*dim+i] += (0.5*cxx-cx*dx)/Dt;
            double x_proj1 = -0.5;
            double y_proj1 = y_0[i];
            double x_proj2 = -0.5;
            double y_proj2 = 0.5;
            Nodeid = search_projection(x_proj1,y_proj1);
            stiffKT[i*dim+Nodeid] += -cx*rbeta; 
            Nodeid = search_projection(x_proj2,y_proj2);
            stiffKT[i*dim+Nodeid] += -cy*rbeta; 
          }
          if(dx <= 0+Tol && dy >=0-Tol && !(fabs(dx)<= 0+Tol && fabs(dy) <= 0+Tol)){
            double cx = 8*((dx*dx-delta*delta*h*h)*(delta*h-dy)/2+pow(delta*h,3)/6-pow(dy,3)/6)/pi/pow(delta*h,4); 
            double cy = 8*(-1*pow(sqrt(delta*delta*h*h-dy*dy),3)/3+(delta*delta*h*h-dy*dy)*dx/2)/pi/pow(delta*h,4);
            double cxx = (dy*(5*delta*delta*h*h-2*dy*dy)*sqrt(delta*delta*h*h-dy*dy)/3/pow(delta*h,4)+atan(dy/sqrt(delta*delta*h*h-dy*dy))+3*pi/2)/pi
              +8*pow(dx,3)*(delta*h-dy)/3/pi/pow(delta*h,4);
            double cyy = (dy*(2*dy*dy-delta*delta*h*h)*sqrt(delta*delta*h*h-dy*dy)/pow(delta*h,4)+atan(dy/sqrt(delta*delta*h*h-dy*dy)+3*pi/2))/pi
              +8*dx*(pow(delta*delta*h*h,3)-pow(dy,3))/3/pi/pow(delta*h,4);
            double cxy = (2*(dx*dx-delta*delta*h*h)*(delta*delta*h*h-dy*dy)+pow(delta*h,4)-pow(dy,4))/pi/pow(delta*h,4);
          
          for(int s=0;s<nei[i].size();s++){
            stiffKT[i*dim+nei[i][s]] -= Beta[s*6+1]*cx+Beta[s*6+2]*cy+Beta[s*6+3]*cxx+Beta[s*6+4]*cyy+Beta[s*6+5]*cxy+Beta[s*6+4]*(0.5*(cxx-cyy)-cx*dx+cy*dy)*2;
          }
            stiffKT[i*dim+i] += (0.5*cxx-cx*dx)/Dt;
            double x_proj1 = -0.5;
            double y_proj1 = 0.5;
            double x_proj2 = x_0[i];
            double y_proj2 = 0.5;
            Nodeid = search_projection(x_proj1,y_proj1);
            stiffKT[i*dim+Nodeid] += -cx*rbeta; 
            Nodeid = search_projection(x_proj2,y_proj2);
            stiffKT[i*dim+Nodeid] += -cy*rbeta; 
          }   
        }
        else if(ifinhorizon(x_0[i],y_0[i],0.5,0.5)){
            normal[0] = 1;
            normal[1] = 0;
            tanvec[0] = 0;
            tanvec[1] = 1;
            for(int s=0;s!=nei[i].size();s++){
              P[s*6] = 1;
              P[s*6+1] = (x_0[nei[i][s]] - x_0[i])*normal[0] + (y_0[nei[i][s]]-y_0[i])*normal[1]; 
              P[s*6+2] = (x_0[nei[i][s]] - x_0[i])*tanvec[0] + (y_0[nei[i][s]]-y_0[i])*tanvec[1];
              P[s*6+3] = pow((x_0[nei[i][s]] - x_0[i])*normal[0] + (y_0[nei[i][s]]-y_0[i])*normal[1],2);
              P[s*6+4] = pow((x_0[nei[i][s]] - x_0[i])*tanvec[0] + (y_0[nei[i][s]]-y_0[i])*tanvec[1],2);
              P[s*6+5] =((x_0[nei[i][s]] - x_0[i])*normal[0] + (y_0[nei[i][s]]-y_0[i])*normal[1])*((x_0[nei[i][s]] - x_0[i])*tanvec[0] + (y_0[nei[i][s]]-y_0[i])*tanvec[1]);
              D[s] =  weight(sqrt(pow(P[s*6+1],2)+pow(P[s*6+2],2)),delta*h);
            }
            for(int m=0;m<6;m++){
              for(int n=0;n<6;n++){
                  M[m*6+n] = 0;
                  for(int s=0;s<nei[i].size();s++){
                      M[m*6+n] += P[s*6+m]*D[s]*P[s*6+n];
                  }
              }
            }
        
          inverse(M,6);
          for(int m=0;m<nei[i].size();m++){
             for(int n=0;n<6;n++){
                  Beta[m*6+n] = 0;
                  for(int s=0;s<6;s++){
                      Beta[m*6+n] += D[m]*P[m*6+s]*M[s*6+n];
                  }
              }
          }

          double dx = 0.5 - x_0[i];
          double dy = 0.5 - y_0[i];
          if(dx > 0 && dy > 0){
            double cx = 8*(-1*pow(sqrt(delta*delta*h*h-dx*dx),3)/3+dy*(delta*delta*h*h-dy*dy-dx*dx)/2+pow(dy,3)/3)/pi/pow(delta*h,4);
            double cy = 8*(-1*pow(sqrt(delta*delta*h*h-dy*dy),3)/3+dx*(delta*delta*h*h-dy*dy-dx*dx)/2+pow(dx,3)/3)/pi/pow(delta*h,4);
            double cxx = (dx*(2*dx*dx-delta*delta*h*h)*sqrt(delta*delta*h*h-dx*dx)/pow(delta*h,4)+dy*(2*dy*dy-delta*delta*h*h)*sqrt(delta*delta*h*h-dy*dy)/pow(delta*h,4)
                +atan(dx/sqrt(delta*delta*h*h-dx*dx))+atan(dy/sqrt(delta*delta*h*h-dy*dy)+2*pi))/pi + 8*dy*(pow(sqrt(delta*delta*h*h-dy*dy),3)-pow(dx,3))/3/pi/pow(delta*h,4);  
            double cyy = (dx*(2*dx*dx-delta*delta*h*h)*sqrt(delta*delta*h*h-dx*dx)/pow(delta*h,4)+dy*(2*dy*dy-delta*delta*h*h)*sqrt(delta*delta*h*h-dy*dy)/pow(delta*h,4)
                +atan(dx/sqrt(delta*delta*h*h-dx*dx))+atan(dy/sqrt(delta*delta*h*h-dy*dy)+2*pi))/pi + 8*dx*(pow(sqrt(delta*delta*h*h-dx*dx),3)-pow(dy,3))/3/pi/pow(delta*h,4);  
            double cxy = -1*pow(delta*delta*h*h-dx*dx-dy*dy,2)/pi/pow(delta*h,4);
          for(int s=0;s<nei[i].size();s++){
            stiffKT[i*dim+nei[i][s]] -= Beta[s*6+1]*cx+Beta[s*6+2]*cy+Beta[s*6+3]*cxx+Beta[s*6+4]*cyy+Beta[s*6+5]*cxy+Beta[s*6+4]*(0.5*(cxx-cyy)-cx*dx+cy*dy)*2;
          }
            stiffKT[i*dim+i] += (0.5*cxx-cx*dx)/Dt;
            double x_proj1 = 0.5;
            double y_proj1 = 0.5;
            double x_proj2 = 0.5;
            double y_proj2 = 0.5;
            Nodeid = search_projection(x_proj1,y_proj1);
            stiffKT[i*dim+Nodeid] += -cx*rbeta; 
            Nodeid = search_projection(x_proj2,y_proj2);
            stiffKT[i*dim+Nodeid] += -cy*rbeta; 
          }
          if(dx >= 0-Tol && dy <= 0){
            double cx = 8*(-1*pow(sqrt(delta*delta*h*h-dx*dx),3)/3+(delta*delta*h*h-dx*dx)*dy/2)/pi/pow(delta*h,4);
            double cy = 8*((dy*dy-delta*delta*h*h)*(delta*h-dx)/2+pow(delta*h,3)/6-pow(dx,3)/6)/pi/pow(delta*h,4);
            double cxx = (dx*(2*dx*dx-delta*delta*h*h)*sqrt(delta*delta*h*h-dx*dx)/pow(delta*h,4)+atan(dx/sqrt(delta*delta*h*h-dx*dx)+3*pi/2))/pi
              +8*dy*(pow(delta*delta*h*h,3)-pow(dx,3))/3/pi/pow(delta*h,4);
            double cyy = (dx*(5*delta*delta*h*h-2*dx*dx)*sqrt(delta*delta*h*h-dx*dx)/3/pow(delta*h,4)+atan(dx/sqrt(delta*delta*h*h-dx*dx))+3*pi/2)/pi
              +8*pow(dy,3)*(delta*h-dx)/3/pi/pow(delta*h,4);
            double cxy = (2*(dy*dy-delta*delta*h*h)*(delta*delta*h*h-dx*dx)+pow(delta*h,4)-pow(dx,4))/pi/pow(delta*h,4);
          for(int s=0;s<nei[i].size();s++){
            stiffKT[i*dim+nei[i][s]] -= Beta[s*6+1]*cx+Beta[s*6+2]*cy+Beta[s*6+3]*cxx+Beta[s*6+4]*cyy+Beta[s*6+5]*cxy+Beta[s*6+4]*(0.5*(cxx-cyy)-cx*dx+cy*dy)*2;
          }
            stiffKT[i*dim+i] += (0.5*cxx-cx*dx)/Dt;
            double x_proj1 = 0.5;
            double y_proj1 = y_0[i];
            double x_proj2 = 0.5;
            double y_proj2 = 0.5;
            Nodeid = search_projection(x_proj1,y_proj1);
            stiffKT[i*dim+Nodeid] += -cx*rbeta; 
            Nodeid = search_projection(x_proj2,y_proj2);
            stiffKT[i*dim+Nodeid] += -cy*rbeta; 
          }
          if(dx <= 0+Tol && dy >=0-Tol && !(fabs(dx)<= 0+Tol && fabs(dy) <= 0+Tol)){
            double cx = 8*((dx*dx-delta*delta*h*h)*(delta*h-dy)/2+pow(delta*h,3)/6-pow(dy,3)/6)/pi/pow(delta*h,4); 
            double cy = 8*(-1*pow(sqrt(delta*delta*h*h-dy*dy),3)/3+(delta*delta*h*h-dy*dy)*dx/2)/pi/pow(delta*h,4);
            double cxx = (dy*(5*delta*delta*h*h-2*dy*dy)*sqrt(delta*delta*h*h-dy*dy)/3/pow(delta*h,4)+atan(dy/sqrt(delta*delta*h*h-dy*dy))+3*pi/2)/pi
              +8*pow(dx,3)*(delta*h-dy)/3/pi/pow(delta*h,4);
            double cyy = (dy*(2*dy*dy-delta*delta*h*h)*sqrt(delta*delta*h*h-dy*dy)/pow(delta*h,4)+atan(dy/sqrt(delta*delta*h*h-dy*dy)+3*pi/2))/pi
              +8*dx*(pow(delta*delta*h*h,3)-pow(dy,3))/3/pi/pow(delta*h,4);
            double cxy = (2*(dx*dx-delta*delta*h*h)*(delta*delta*h*h-dy*dy)+pow(delta*h,4)-pow(dy,4))/pi/pow(delta*h,4);
          
          for(int s=0;s<nei[i].size();s++){
            stiffKT[i*dim+nei[i][s]] -= Beta[s*6+1]*cx+Beta[s*6+2]*cy+Beta[s*6+3]*cxx+Beta[s*6+4]*cyy+Beta[s*6+5]*cxy+Beta[s*6+4]*(0.5*(cxx-cyy)-cx*dx+cy*dy)*2;
          }
          stiffKT[i*dim+i] += (0.5*cxx-cx*dx)/Dt;
            double x_proj1 = 0.5;
            double y_proj1 = 0.5;
            double x_proj2 = x_0[i];
            double y_proj2 = 0.5;
            Nodeid = search_projection(x_proj1,y_proj1);
            stiffKT[i*dim+Nodeid] += -cx*rbeta; 
            Nodeid = search_projection(x_proj2,y_proj2);
            stiffKT[i*dim+Nodeid] += -cy*rbeta; 
          }   
        }
        else if(ifinhorizon(x_0[i],y_0[i],-0.5,-0.5)){
            normal[0] = -1;
            normal[1] = 0;
            tanvec[0] = 0;
            tanvec[1] = -1;
            for(int s=0;s!=nei[i].size();s++){
              P[s*6] = 1;
              P[s*6+1] = (x_0[nei[i][s]] - x_0[i])*normal[0] + (y_0[nei[i][s]]-y_0[i])*normal[1]; 
              P[s*6+2] = (x_0[nei[i][s]] - x_0[i])*tanvec[0] + (y_0[nei[i][s]]-y_0[i])*tanvec[1];
              P[s*6+3] = pow((x_0[nei[i][s]] - x_0[i])*normal[0] + (y_0[nei[i][s]]-y_0[i])*normal[1],2);
              P[s*6+4] = pow((x_0[nei[i][s]] - x_0[i])*tanvec[0] + (y_0[nei[i][s]]-y_0[i])*tanvec[1],2);
              P[s*6+5] =((x_0[nei[i][s]] - x_0[i])*normal[0] + (y_0[nei[i][s]]-y_0[i])*normal[1])*((x_0[nei[i][s]] - x_0[i])*tanvec[0] + (y_0[nei[i][s]]-y_0[i])*tanvec[1]);
              D[s] =  weight(sqrt(pow(P[s*6+1],2)+pow(P[s*6+2],2)),delta*h);
            }
            for(int m=0;m<6;m++){
              for(int n=0;n<6;n++){
                  M[m*6+n] = 0;
                  for(int s=0;s<nei[i].size();s++){
                      M[m*6+n] += P[s*6+m]*D[s]*P[s*6+n];
                  }
              }
            }
        
          inverse(M,6);
          for(int m=0;m<nei[i].size();m++){
             for(int n=0;n<6;n++){
                  Beta[m*6+n] = 0;
                  for(int s=0;s<6;s++){
                      Beta[m*6+n] += D[m]*P[m*6+s]*M[s*6+n];
                  }
              }
          }

          double dx = 0.5 + x_0[i];
          double dy = 0.5 + y_0[i];
          if(dx > 0 && dy > 0){
            double cx = 8*(-1*pow(sqrt(delta*delta*h*h-dx*dx),3)/3+dy*(delta*delta*h*h-dy*dy-dx*dx)/2+pow(dy,3)/3)/pi/pow(delta*h,4);
            double cy = 8*(-1*pow(sqrt(delta*delta*h*h-dy*dy),3)/3+dx*(delta*delta*h*h-dy*dy-dx*dx)/2+pow(dx,3)/3)/pi/pow(delta*h,4);
            double cxx = (dx*(2*dx*dx-delta*delta*h*h)*sqrt(delta*delta*h*h-dx*dx)/pow(delta*h,4)+dy*(2*dy*dy-delta*delta*h*h)*sqrt(delta*delta*h*h-dy*dy)/pow(delta*h,4)
                +atan(dx/sqrt(delta*delta*h*h-dx*dx))+atan(dy/sqrt(delta*delta*h*h-dy*dy)+2*pi))/pi + 8*dy*(pow(sqrt(delta*delta*h*h-dy*dy),3)-pow(dx,3))/3/pi/pow(delta*h,4);  
            double cyy = (dx*(2*dx*dx-delta*delta*h*h)*sqrt(delta*delta*h*h-dx*dx)/pow(delta*h,4)+dy*(2*dy*dy-delta*delta*h*h)*sqrt(delta*delta*h*h-dy*dy)/pow(delta*h,4)
                +atan(dx/sqrt(delta*delta*h*h-dx*dx))+atan(dy/sqrt(delta*delta*h*h-dy*dy)+2*pi))/pi + 8*dx*(pow(sqrt(delta*delta*h*h-dx*dx),3)-pow(dy,3))/3/pi/pow(delta*h,4);  
            double cxy = -1*pow(delta*delta*h*h-dx*dx-dy*dy,2)/pi/pow(delta*h,4);
          for(int s=0;s<nei[i].size();s++){
            stiffKT[i*dim+nei[i][s]] -= Beta[s*6+1]*cx+Beta[s*6+2]*cy+Beta[s*6+3]*cxx+Beta[s*6+4]*cyy+Beta[s*6+5]*cxy+Beta[s*6+4]*(0.5*(cxx-cyy)-cx*dx+cy*dy)*2;
          }
            stiffKT[i*dim+i] += (0.5*cxx-cx*dx)/Dt;
            double x_proj1 = -0.5;
            double y_proj1 = -0.5;
            double x_proj2 = -0.5;
            double y_proj2 = -0.5;
            Nodeid = search_projection(x_proj1,y_proj1);
            stiffKT[i*dim+Nodeid] += -cx*rbeta; 
            Nodeid = search_projection(x_proj2,y_proj2);
            stiffKT[i*dim+Nodeid] += -cy*rbeta; 
          }
          if(dx >= 0-Tol && dy <= 0){
            double cx = 8*(-1*pow(sqrt(delta*delta*h*h-dx*dx),3)/3+(delta*delta*h*h-dx*dx)*dy/2)/pi/pow(delta*h,4);
            double cy = 8*((dy*dy-delta*delta*h*h)*(delta*h-dx)/2+pow(delta*h,3)/6-pow(dx,3)/6)/pi/pow(delta*h,4);
            double cxx = (dx*(2*dx*dx-delta*delta*h*h)*sqrt(delta*delta*h*h-dx*dx)/pow(delta*h,4)+atan(dx/sqrt(delta*delta*h*h-dx*dx)+3*pi/2))/pi
              +8*dy*(pow(delta*delta*h*h,3)-pow(dx,3))/3/pi/pow(delta*h,4);
            double cyy = (dx*(5*delta*delta*h*h-2*dx*dx)*sqrt(delta*delta*h*h-dx*dx)/3/pow(delta*h,4)+atan(dx/sqrt(delta*delta*h*h-dx*dx))+3*pi/2)/pi
              +8*pow(dy,3)*(delta*h-dx)/3/pi/pow(delta*h,4);
            double cxy = (2*(dy*dy-delta*delta*h*h)*(delta*delta*h*h-dx*dx)+pow(delta*h,4)-pow(dx,4))/pi/pow(delta*h,4);
          for(int s=0;s<nei[i].size();s++){
            stiffKT[i*dim+nei[i][s]] -= Beta[s*6+1]*cx+Beta[s*6+2]*cy+Beta[s*6+3]*cxx+Beta[s*6+4]*cyy+Beta[s*6+5]*cxy+Beta[s*6+4]*(0.5*(cxx-cyy)-cx*dx+cy*dy)*2;
          }
            stiffKT[i*dim+i] += (0.5*cxx-cx*dx)/Dt;
            double x_proj1 = -0.5;
            double y_proj1 = y_0[i];
            double x_proj2 = -0.5;
            double y_proj2 = -0.5;
            Nodeid = search_projection(x_proj1,y_proj1);
            stiffKT[i*dim+Nodeid] += -cx*rbeta; 
            Nodeid = search_projection(x_proj2,y_proj2);
            stiffKT[i*dim+Nodeid] += -cy*rbeta; 
          }
          if(dx <= 0+Tol && dy >=0-Tol && !(fabs(dx)<= 0+Tol && fabs(dy) <= 0+Tol)){
            double cx = 8*((dx*dx-delta*delta*h*h)*(delta*h-dy)/2+pow(delta*h,3)/6-pow(dy,3)/6)/pi/pow(delta*h,4); 
            double cy = 8*(-1*pow(sqrt(delta*delta*h*h-dy*dy),3)/3+(delta*delta*h*h-dy*dy)*dx/2)/pi/pow(delta*h,4);
            double cxx = (dy*(5*delta*delta*h*h-2*dy*dy)*sqrt(delta*delta*h*h-dy*dy)/3/pow(delta*h,4)+atan(dy/sqrt(delta*delta*h*h-dy*dy))+3*pi/2)/pi
              +8*pow(dx,3)*(delta*h-dy)/3/pi/pow(delta*h,4);
            double cyy = (dy*(2*dy*dy-delta*delta*h*h)*sqrt(delta*delta*h*h-dy*dy)/pow(delta*h,4)+atan(dy/sqrt(delta*delta*h*h-dy*dy)+3*pi/2))/pi
              +8*dx*(pow(delta*delta*h*h,3)-pow(dy,3))/3/pi/pow(delta*h,4);
            double cxy = (2*(dx*dx-delta*delta*h*h)*(delta*delta*h*h-dy*dy)+pow(delta*h,4)-pow(dy,4))/pi/pow(delta*h,4);
          
          for(int s=0;s<nei[i].size();s++){
            stiffKT[i*dim+nei[i][s]] -= Beta[s*6+1]*cx+Beta[s*6+2]*cy+Beta[s*6+3]*cxx+Beta[s*6+4]*cyy+Beta[s*6+5]*cxy+Beta[s*6+4]*(0.5*(cxx-cyy)-cx*dx+cy*dy)*2;
          }
            stiffKT[i*dim+i] += (0.5*cxx-cx*dx)/Dt;
            double x_proj1 = -0.5;
            double y_proj1 = -0.5;
            double x_proj2 = x_0[i];
            double y_proj2 = -0.5;
            Nodeid = search_projection(x_proj1,y_proj1);
            stiffKT[i*dim+Nodeid] += -cx*rbeta; 
            Nodeid = search_projection(x_proj2,y_proj2);
            stiffKT[i*dim+Nodeid] += -cy*rbeta; 
          }   
        }
        else if(ifinhorizon(x_0[i],y_0[i],0.5,-0.5)){
            normal[0] = 1;
            normal[1] = 0;
            tanvec[0] = 0;
            tanvec[1] = -1;
            for(int s=0;s!=nei[i].size();s++){
              P[s*6] = 1;
              P[s*6+1] = (x_0[nei[i][s]] - x_0[i])*normal[0] + (y_0[nei[i][s]]-y_0[i])*normal[1]; 
              P[s*6+2] = (x_0[nei[i][s]] - x_0[i])*tanvec[0] + (y_0[nei[i][s]]-y_0[i])*tanvec[1];
              P[s*6+3] = pow((x_0[nei[i][s]] - x_0[i])*normal[0] + (y_0[nei[i][s]]-y_0[i])*normal[1],2);
              P[s*6+4] = pow((x_0[nei[i][s]] - x_0[i])*tanvec[0] + (y_0[nei[i][s]]-y_0[i])*tanvec[1],2);
              P[s*6+5] =((x_0[nei[i][s]] - x_0[i])*normal[0] + (y_0[nei[i][s]]-y_0[i])*normal[1])*((x_0[nei[i][s]] - x_0[i])*tanvec[0] + (y_0[nei[i][s]]-y_0[i])*tanvec[1]);
              D[s] =  weight(sqrt(pow(P[s*6+1],2)+pow(P[s*6+2],2)),delta*h);
            }
            for(int m=0;m<6;m++){
              for(int n=0;n<6;n++){
                  M[m*6+n] = 0;
                  for(int s=0;s<nei[i].size();s++){
                      M[m*6+n] += P[s*6+m]*D[s]*P[s*6+n];
                  }
              }
            }
        
          inverse(M,6);
          for(int m=0;m<nei[i].size();m++){
             for(int n=0;n<6;n++){
                  Beta[m*6+n] = 0;
                  for(int s=0;s<6;s++){
                      Beta[m*6+n] += D[m]*P[m*6+s]*M[s*6+n];
                  }
              }
          }

          double dx = 0.5 - x_0[i];
          double dy = 0.5 + y_0[i];
          if(dx > 0 && dy > 0){
            double cx = 8*(-1*pow(sqrt(delta*delta*h*h-dx*dx),3)/3+dy*(delta*delta*h*h-dy*dy-dx*dx)/2+pow(dy,3)/3)/pi/pow(delta*h,4);
            double cy = 8*(-1*pow(sqrt(delta*delta*h*h-dy*dy),3)/3+dx*(delta*delta*h*h-dy*dy-dx*dx)/2+pow(dx,3)/3)/pi/pow(delta*h,4);
            double cxx = (dx*(2*dx*dx-delta*delta*h*h)*sqrt(delta*delta*h*h-dx*dx)/pow(delta*h,4)+dy*(2*dy*dy-delta*delta*h*h)*sqrt(delta*delta*h*h-dy*dy)/pow(delta*h,4)
                +atan(dx/sqrt(delta*delta*h*h-dx*dx))+atan(dy/sqrt(delta*delta*h*h-dy*dy)+2*pi))/pi + 8*dy*(pow(sqrt(delta*delta*h*h-dy*dy),3)-pow(dx,3))/3/pi/pow(delta*h,4);  
            double cyy = (dx*(2*dx*dx-delta*delta*h*h)*sqrt(delta*delta*h*h-dx*dx)/pow(delta*h,4)+dy*(2*dy*dy-delta*delta*h*h)*sqrt(delta*delta*h*h-dy*dy)/pow(delta*h,4)
                +atan(dx/sqrt(delta*delta*h*h-dx*dx))+atan(dy/sqrt(delta*delta*h*h-dy*dy)+2*pi))/pi + 8*dx*(pow(sqrt(delta*delta*h*h-dx*dx),3)-pow(dy,3))/3/pi/pow(delta*h,4);  
            double cxy = -1*pow(delta*delta*h*h-dx*dx-dy*dy,2)/pi/pow(delta*h,4);
          for(int s=0;s<nei[i].size();s++){
            stiffKT[i*dim+nei[i][s]] -= Beta[s*6+1]*cx+Beta[s*6+2]*cy+Beta[s*6+3]*cxx+Beta[s*6+4]*cyy+Beta[s*6+5]*cxy+Beta[s*6+4]*(0.5*(cxx-cyy)-cx*dx+cy*dy)*2;
          }
            stiffKT[i*dim+i] += (0.5*cxx-cx*dx)/Dt;
            double x_proj1 = 0.5;
            double y_proj1 = -0.5;
            double x_proj2 = 0.5;
            double y_proj2 = -0.5;
            Nodeid = search_projection(x_proj1,y_proj1);
            stiffKT[i*dim+Nodeid] += -cx*rbeta; 
            Nodeid = search_projection(x_proj2,y_proj2);
            stiffKT[i*dim+Nodeid] += -cy*rbeta; 
          }
          if(dx >= 0-Tol && dy <= 0){
            double cx = 8*(-1*pow(sqrt(delta*delta*h*h-dx*dx),3)/3+(delta*delta*h*h-dx*dx)*dy/2)/pi/pow(delta*h,4);
            double cy = 8*((dy*dy-delta*delta*h*h)*(delta*h-dx)/2+pow(delta*h,3)/6-pow(dx,3)/6)/pi/pow(delta*h,4);
            double cxx = (dx*(2*dx*dx-delta*delta*h*h)*sqrt(delta*delta*h*h-dx*dx)/pow(delta*h,4)+atan(dx/sqrt(delta*delta*h*h-dx*dx)+3*pi/2))/pi
              +8*dy*(pow(delta*delta*h*h,3)-pow(dx,3))/3/pi/pow(delta*h,4);
            double cyy = (dx*(5*delta*delta*h*h-2*dx*dx)*sqrt(delta*delta*h*h-dx*dx)/3/pow(delta*h,4)+atan(dx/sqrt(delta*delta*h*h-dx*dx))+3*pi/2)/pi
              +8*pow(dy,3)*(delta*h-dx)/3/pi/pow(delta*h,4);
            double cxy = (2*(dy*dy-delta*delta*h*h)*(delta*delta*h*h-dx*dx)+pow(delta*h,4)-pow(dx,4))/pi/pow(delta*h,4);
          for(int s=0;s<nei[i].size();s++){
            stiffKT[i*dim+nei[i][s]] -= Beta[s*6+1]*cx+Beta[s*6+2]*cy+Beta[s*6+3]*cxx+Beta[s*6+4]*cyy+Beta[s*6+5]*cxy+Beta[s*6+4]*(0.5*(cxx-cyy)-cx*dx+cy*dy)*2;
          }
            stiffKT[i*dim+i] += (0.5*cxx-cx*dx)/Dt;
            double x_proj1 = 0.5;
            double y_proj1 = y_0[i];
            double x_proj2 = 0.5;
            double y_proj2 = -0.5;
            Nodeid = search_projection(x_proj1,y_proj1);
            stiffKT[i*dim+Nodeid] += -cx*rbeta; 
            Nodeid = search_projection(x_proj2,y_proj2);
            stiffKT[i*dim+Nodeid] += -cy*rbeta; 
          }
          if(dx <= 0+Tol && dy >=0-Tol && !(fabs(dx)<= 0+Tol && fabs(dy) <= 0+Tol)){
            double cx = 8*((dx*dx-delta*delta*h*h)*(delta*h-dy)/2+pow(delta*h,3)/6-pow(dy,3)/6)/pi/pow(delta*h,4); 
            double cy = 8*(-1*pow(sqrt(delta*delta*h*h-dy*dy),3)/3+(delta*delta*h*h-dy*dy)*dx/2)/pi/pow(delta*h,4);
            double cxx = (dy*(5*delta*delta*h*h-2*dy*dy)*sqrt(delta*delta*h*h-dy*dy)/3/pow(delta*h,4)+atan(dy/sqrt(delta*delta*h*h-dy*dy))+3*pi/2)/pi
              +8*pow(dx,3)*(delta*h-dy)/3/pi/pow(delta*h,4);
            double cyy = (dy*(2*dy*dy-delta*delta*h*h)*sqrt(delta*delta*h*h-dy*dy)/pow(delta*h,4)+atan(dy/sqrt(delta*delta*h*h-dy*dy)+3*pi/2))/pi
              +8*dx*(pow(delta*delta*h*h,3)-pow(dy,3))/3/pi/pow(delta*h,4);
            double cxy = (2*(dx*dx-delta*delta*h*h)*(delta*delta*h*h-dy*dy)+pow(delta*h,4)-pow(dy,4))/pi/pow(delta*h,4);
          
          for(int s=0;s<nei[i].size();s++){
            stiffKT[i*dim+nei[i][s]] -= Beta[s*6+1]*cx+Beta[s*6+2]*cy+Beta[s*6+3]*cxx+Beta[s*6+4]*cyy+Beta[s*6+5]*cxy+Beta[s*6+4]*(0.5*(cxx-cyy)-cx*dx+cy*dy)*2;
          }
            stiffKT[i*dim+i] += (0.5*cxx-cx*dx)/Dt;
            double x_proj1 = 0.5;
            double y_proj1 = -0.5;
            double x_proj2 = x_0[i];
            double y_proj2 = -0.5;
            Nodeid = search_projection(x_proj1,y_proj1);
            stiffKT[i*dim+Nodeid] += -cx*rbeta; 
            Nodeid = search_projection(x_proj2,y_proj2);
            stiffKT[i*dim+Nodeid] += -cy*rbeta; 
          }   
        }
        //straight line cases
        else if(abs(x_0[i]+1) <= delta*h+Tol){ 
            normal[0] = -1;
            normal[1] = 0;
            tanvec[0] = 0;
            tanvec[1] = -1;
            for(int s=0;s!=nei[i].size();s++){
              P[s*6] = 1;
              P[s*6+1] = (x_0[nei[i][s]] - x_0[i])*normal[0] + (y_0[nei[i][s]]-y_0[i])*normal[1]; 
              P[s*6+2] = (x_0[nei[i][s]] - x_0[i])*tanvec[0] + (y_0[nei[i][s]]-y_0[i])*tanvec[1];
              P[s*6+3] = pow((x_0[nei[i][s]] - x_0[i])*normal[0] + (y_0[nei[i][s]]-y_0[i])*normal[1],2);
              P[s*6+4] = pow((x_0[nei[i][s]] - x_0[i])*tanvec[0] + (y_0[nei[i][s]]-y_0[i])*tanvec[1],2);
              P[s*6+5] =((x_0[nei[i][s]] - x_0[i])*normal[0] + (y_0[nei[i][s]]-y_0[i])*normal[1])*((x_0[nei[i][s]] - x_0[i])*tanvec[0] + (y_0[nei[i][s]]-y_0[i])*tanvec[1]);
              D[s] =  weight(sqrt(pow(P[s*6+1],2)+pow(P[s*6+2],2)),delta*h);
            }
            for(int m=0;m<6;m++){
              for(int n=0;n<6;n++){
                  M[m*6+n] = 0;
                  for(int s=0;s<nei[i].size();s++){
                      M[m*6+n] += P[s*6+m]*D[s]*P[s*6+n];
                  }
              }
            }
        
          inverse(M,6);
          for(int m=0;m<nei[i].size();m++){
             for(int n=0;n<6;n++){
                  Beta[m*6+n] = 0;
                  for(int s=0;s<6;s++){
                      Beta[m*6+n] += D[m]*P[m*6+s]*M[s*6+n];
                  }
              }
          }

          double dist = x_0[i]+1;
          double cx = -16*pow(sqrt(delta*delta*h*h-dist*dist),3)/3/pi/pow(delta*h,4);
          double cxx = (2*dist*(2*dist*dist-delta*h*delta*h)*sqrt(delta*delta*h*h-dist*dist)/pow(delta*h,4)+2*atan(dist/sqrt(delta*delta*h*h-dist*dist))
                +pi)/pi;
          double cyy = (2*dist*(5*delta*delta*h*h-2*dist*dist)*sqrt(delta*delta*h*h-dist*dist)/3/pow(delta*h,4)+2*atan(dist/sqrt(delta*delta*h*h-dist*dist))+pi)/pi;
          
          for(int s=0;s<nei[i].size();s++){
            stiffKT[i*dim+nei[i][s]] -= Beta[s*6+1]*cx + Beta[s*6+3]*cxx + Beta[s*6+4]*cyy+Beta[s*6+4]*(0.5*(cxx-cyy)-cx*dist)*2;  
          }
            stiffKT[i*dim+i] += (0.5*cxx-cx*dist)/Dt;
          //stiffKT[i*dim+i] = 1;
            double x_proj = -1;
            double y_proj = y_0[i];
            Nodeid = search_projection(x_proj,y_proj);
            stiffKT[i*dim+Nodeid] += -cx*rbeta; 
        }
        else if(abs(x_0[i]-1) <= delta*h+Tol){ 
            normal[0] = 1;
            normal[1] = 0;
            tanvec[0] = 0;
            tanvec[1] = 1;
            for(int s=0;s!=nei[i].size();s++){
              P[s*6] = 1;
              P[s*6+1] = (x_0[nei[i][s]] - x_0[i])*normal[0] + (y_0[nei[i][s]]-y_0[i])*normal[1]; 
              P[s*6+2] = (x_0[nei[i][s]] - x_0[i])*tanvec[0] + (y_0[nei[i][s]]-y_0[i])*tanvec[1];
              P[s*6+3] = pow((x_0[nei[i][s]] - x_0[i])*normal[0] + (y_0[nei[i][s]]-y_0[i])*normal[1],2);
              P[s*6+4] = pow((x_0[nei[i][s]] - x_0[i])*tanvec[0] + (y_0[nei[i][s]]-y_0[i])*tanvec[1],2);
              P[s*6+5] =((x_0[nei[i][s]] - x_0[i])*normal[0] + (y_0[nei[i][s]]-y_0[i])*normal[1])*((x_0[nei[i][s]] - x_0[i])*tanvec[0] + (y_0[nei[i][s]]-y_0[i])*tanvec[1]);
              D[s] =  weight(sqrt(pow(P[s*6+1],2)+pow(P[s*6+2],2)),delta*h);
            }
            for(int m=0;m<6;m++){
              for(int n=0;n<6;n++){
                  M[m*6+n] = 0;
                  for(int s=0;s<nei[i].size();s++){
                      M[m*6+n] += P[s*6+m]*D[s]*P[s*6+n];
                  }
              }
            }
        
          inverse(M,6);
          for(int m=0;m<nei[i].size();m++){
             for(int n=0;n<6;n++){
                  Beta[m*6+n] = 0;
                  for(int s=0;s<6;s++){
                      Beta[m*6+n] += D[m]*P[m*6+s]*M[s*6+n];
                  }
              }
          }

          double dist = -x_0[i]+1;
          double cx = -16*pow(sqrt(delta*delta*h*h-dist*dist),3)/3/pi/pow(delta*h,4);
          double cxx = (2*dist*(2*dist*dist-delta*h*delta*h)*sqrt(delta*delta*h*h-dist*dist)/pow(delta*h,4)+2*atan(dist/sqrt(delta*delta*h*h-dist*dist))
                +pi)/pi;
          double cyy = (2*dist*(5*delta*delta*h*h-2*dist*dist)*sqrt(delta*delta*h*h-dist*dist)/3/pow(delta*h,4)+2*atan(dist/sqrt(delta*delta*h*h-dist*dist))+pi)/pi;
          for(int s=0;s<nei[i].size();s++){
            stiffKT[i*dim+nei[i][s]] -= Beta[s*6+1]*cx + Beta[s*6+3]*cxx + Beta[s*6+4]*cyy+Beta[s*6+4]*(0.5*(cxx-cyy)-cx*dist)*2;  
          }
            stiffKT[i*dim+i] += (0.5*cxx-cx*dist)/Dt;
            double x_proj = 1;
            double y_proj = y_0[i];
            Nodeid = search_projection(x_proj,y_proj);
            stiffKT[i*dim+Nodeid] += -cx*rbeta; 
        }
        else if(abs(y_0[i]+1) <= delta*h+Tol){ 
            normal[0] = 0;
            normal[1] = -1;
            tanvec[0] = 1;
            tanvec[1] = 0;
            for(int s=0;s!=nei[i].size();s++){
              P[s*6] = 1;
              P[s*6+1] = (x_0[nei[i][s]] - x_0[i])*normal[0] + (y_0[nei[i][s]]-y_0[i])*normal[1]; 
              P[s*6+2] = (x_0[nei[i][s]] - x_0[i])*tanvec[0] + (y_0[nei[i][s]]-y_0[i])*tanvec[1];
              P[s*6+3] = pow((x_0[nei[i][s]] - x_0[i])*normal[0] + (y_0[nei[i][s]]-y_0[i])*normal[1],2);
              P[s*6+4] = pow((x_0[nei[i][s]] - x_0[i])*tanvec[0] + (y_0[nei[i][s]]-y_0[i])*tanvec[1],2);
              P[s*6+5] =((x_0[nei[i][s]] - x_0[i])*normal[0] + (y_0[nei[i][s]]-y_0[i])*normal[1])*((x_0[nei[i][s]] - x_0[i])*tanvec[0] + (y_0[nei[i][s]]-y_0[i])*tanvec[1]);
              D[s] =  weight(sqrt(pow(P[s*6+1],2)+pow(P[s*6+2],2)),delta*h);
            }
            for(int m=0;m<6;m++){
              for(int n=0;n<6;n++){
                  M[m*6+n] = 0;
                  for(int s=0;s<nei[i].size();s++){
                      M[m*6+n] += P[s*6+m]*D[s]*P[s*6+n];
                  }
              }
            }
        
          inverse(M,6);
          for(int m=0;m<nei[i].size();m++){
             for(int n=0;n<6;n++){
                  Beta[m*6+n] = 0;
                  for(int s=0;s<6;s++){
                      Beta[m*6+n] += D[m]*P[m*6+s]*M[s*6+n];
                  }
              }
          }

          double dist = y_0[i]+1;
          double cx = -16*pow(sqrt(delta*delta*h*h-dist*dist),3)/3/pi/pow(delta*h,4);
          double cxx = (2*dist*(2*dist*dist-delta*h*delta*h)*sqrt(delta*delta*h*h-dist*dist)/pow(delta*h,4)+2*atan(dist/sqrt(delta*delta*h*h-dist*dist))
                +pi)/pi;
          double cyy = (2*dist*(5*delta*delta*h*h-2*dist*dist)*sqrt(delta*delta*h*h-dist*dist)/3/pow(delta*h,4)+2*atan(dist/sqrt(delta*delta*h*h-dist*dist))+pi)/pi;
          
          for(int s=0;s<nei[i].size();s++){
            stiffKT[i*dim+nei[i][s]] -= Beta[s*6+1]*cx + Beta[s*6+3]*cxx + Beta[s*6+4]*cyy+Beta[s*6+4]*(0.5*(cxx-cyy)-cx*dist)*2;  
          }
            stiffKT[i*dim+i] += (0.5*cxx-cx*dist)/Dt;
            double x_proj = x_0[i];
            double y_proj = -1;
            Nodeid = search_projection(x_proj,y_proj);
            stiffKT[i*dim+Nodeid] += -cx*rbeta; 
        }
        else if(abs(y_0[i]-1) <= delta*h+Tol){ 
            normal[0] = 0;
            normal[1] = 1;
            tanvec[0] = -1;
            tanvec[1] = 0;
            for(int s=0;s!=nei[i].size();s++){
              P[s*6] = 1;
              P[s*6+1] = (x_0[nei[i][s]] - x_0[i])*normal[0] + (y_0[nei[i][s]]-y_0[i])*normal[1]; 
              P[s*6+2] = (x_0[nei[i][s]] - x_0[i])*tanvec[0] + (y_0[nei[i][s]]-y_0[i])*tanvec[1];
              P[s*6+3] = pow((x_0[nei[i][s]] - x_0[i])*normal[0] + (y_0[nei[i][s]]-y_0[i])*normal[1],2);
              P[s*6+4] = pow((x_0[nei[i][s]] - x_0[i])*tanvec[0] + (y_0[nei[i][s]]-y_0[i])*tanvec[1],2);
              P[s*6+5] =((x_0[nei[i][s]] - x_0[i])*normal[0] + (y_0[nei[i][s]]-y_0[i])*normal[1])*((x_0[nei[i][s]] - x_0[i])*tanvec[0] + (y_0[nei[i][s]]-y_0[i])*tanvec[1]);
              D[s] =  weight(sqrt(pow(P[s*6+1],2)+pow(P[s*6+2],2)),delta*h);
            }
            for(int m=0;m<6;m++){
              for(int n=0;n<6;n++){
                  M[m*6+n] = 0;
                  for(int s=0;s<nei[i].size();s++){
                      M[m*6+n] += P[s*6+m]*D[s]*P[s*6+n];
                  }
              }
            }
        
          inverse(M,6);
          for(int m=0;m<nei[i].size();m++){
             for(int n=0;n<6;n++){
                  Beta[m*6+n] = 0;
                  for(int s=0;s<6;s++){
                      Beta[m*6+n] += D[m]*P[m*6+s]*M[s*6+n];
                  }
              }
          }

          double dist = -y_0[i]+1;
          double cx = -16*pow(sqrt(delta*delta*h*h-dist*dist),3)/3/pi/pow(delta*h,4);
          double cxx = (2*dist*(2*dist*dist-delta*h*delta*h)*sqrt(delta*delta*h*h-dist*dist)/pow(delta*h,4)+2*atan(dist/sqrt(delta*delta*h*h-dist*dist))
                +pi)/pi;
          double cyy = (2*dist*(5*delta*delta*h*h-2*dist*dist)*sqrt(delta*delta*h*h-dist*dist)/3/pow(delta*h,4)+2*atan(dist/sqrt(delta*delta*h*h-dist*dist))+pi)/pi;
          
          for(int s=0;s<nei[i].size();s++){
            stiffKT[i*dim+nei[i][s]] -= Beta[s*6+1]*cx + Beta[s*6+3]*cxx + Beta[s*6+4]*cyy+Beta[s*6+4]*(0.5*(cxx-cyy)-cx*dist)*2;  
          }
            stiffKT[i*dim+i] += (0.5*cxx-cx*dist)/Dt;
            double x_proj = x_0[i];
            double y_proj = 1;
            Nodeid = search_projection(x_proj,y_proj);
            stiffKT[i*dim+Nodeid] += -cx*rbeta; 
        }
        else if((x_0[i]+0.5) <= delta*h+Tol && x_0[i] > -0.5-Tol&& (y_0[i] >= 0.5-Tol || y_0[i]<= -0.5+Tol)){ 
            normal[0] = -1;
            normal[1] = 0;
            tanvec[0] = 0;
            tanvec[1] = -1;
            for(int s=0;s!=nei[i].size();s++){
              P[s*6] = 1;
              P[s*6+1] = (x_0[nei[i][s]] - x_0[i])*normal[0] + (y_0[nei[i][s]]-y_0[i])*normal[1]; 
              P[s*6+2] = (x_0[nei[i][s]] - x_0[i])*tanvec[0] + (y_0[nei[i][s]]-y_0[i])*tanvec[1];
              P[s*6+3] = pow((x_0[nei[i][s]] - x_0[i])*normal[0] + (y_0[nei[i][s]]-y_0[i])*normal[1],2);
              P[s*6+4] = pow((x_0[nei[i][s]] - x_0[i])*tanvec[0] + (y_0[nei[i][s]]-y_0[i])*tanvec[1],2);
              P[s*6+5] =((x_0[nei[i][s]] - x_0[i])*normal[0] + (y_0[nei[i][s]]-y_0[i])*normal[1])*((x_0[nei[i][s]] - x_0[i])*tanvec[0] + (y_0[nei[i][s]]-y_0[i])*tanvec[1]);
              D[s] =  weight(sqrt(pow(P[s*6+1],2)+pow(P[s*6+2],2)),delta*h);
            }
            for(int m=0;m<6;m++){
              for(int n=0;n<6;n++){
                  M[m*6+n] = 0;
                  for(int s=0;s<nei[i].size();s++){
                      M[m*6+n] += P[s*6+m]*D[s]*P[s*6+n];
                  }
              }
            }
        
          inverse(M,6);
          for(int m=0;m<nei[i].size();m++){
             for(int n=0;n<6;n++){
                  Beta[m*6+n] = 0;
                  for(int s=0;s<6;s++){
                      Beta[m*6+n] += D[m]*P[m*6+s]*M[s*6+n];
                  }
              }
          }

          double dist = x_0[i]+0.5;
          double cx = -16*pow(sqrt(delta*delta*h*h-dist*dist),3)/3/pi/pow(delta*h,4);
          double cxx = (2*dist*(2*dist*dist-delta*h*delta*h)*sqrt(delta*delta*h*h-dist*dist)/pow(delta*h,4)+2*atan(dist/sqrt(delta*delta*h*h-dist*dist))
                +pi)/pi;
          double cyy = (2*dist*(5*delta*delta*h*h-2*dist*dist)*sqrt(delta*delta*h*h-dist*dist)/3/pow(delta*h,4)+2*atan(dist/sqrt(delta*delta*h*h-dist*dist))+pi)/pi;
          
          for(int s=0;s<nei[i].size();s++){
            stiffKT[i*dim+nei[i][s]] -= Beta[s*6+1]*cx + Beta[s*6+3]*cxx + Beta[s*6+4]*cyy+Beta[s*6+4]*(0.5*(cxx-cyy)-cx*dist)*2;  
          }
            stiffKT[i*dim+i] += (0.5*cxx-cx*dist)/Dt;
            double x_proj = -0.5;
            double y_proj = y_0[i];
            Nodeid = search_projection(x_proj,y_proj);
            stiffKT[i*dim+Nodeid] += -cx*rbeta; 
        }
        else if((-x_0[i]+0.5) <= delta*h+Tol && x_0[i] < 0.5+Tol&& (y_0[i] >= 0.5-Tol || y_0[i]<= -0.5+Tol)){ 
            normal[0] = 1;
            normal[1] = 0;
            tanvec[0] = 0;
            tanvec[1] = 1;
            for(int s=0;s!=nei[i].size();s++){
              P[s*6] = 1;
              P[s*6+1] = (x_0[nei[i][s]] - x_0[i])*normal[0] + (y_0[nei[i][s]]-y_0[i])*normal[1]; 
              P[s*6+2] = (x_0[nei[i][s]] - x_0[i])*tanvec[0] + (y_0[nei[i][s]]-y_0[i])*tanvec[1];
              P[s*6+3] = pow((x_0[nei[i][s]] - x_0[i])*normal[0] + (y_0[nei[i][s]]-y_0[i])*normal[1],2);
              P[s*6+4] = pow((x_0[nei[i][s]] - x_0[i])*tanvec[0] + (y_0[nei[i][s]]-y_0[i])*tanvec[1],2);
              P[s*6+5] =((x_0[nei[i][s]] - x_0[i])*normal[0] + (y_0[nei[i][s]]-y_0[i])*normal[1])*((x_0[nei[i][s]] - x_0[i])*tanvec[0] + (y_0[nei[i][s]]-y_0[i])*tanvec[1]);
              D[s] =  weight(sqrt(pow(P[s*6+1],2)+pow(P[s*6+2],2)),delta*h);
            }
            for(int m=0;m<6;m++){
              for(int n=0;n<6;n++){
                  M[m*6+n] = 0;
                  for(int s=0;s<nei[i].size();s++){
                      M[m*6+n] += P[s*6+m]*D[s]*P[s*6+n];
                  }
              }
            }
        
          inverse(M,6);
          for(int m=0;m<nei[i].size();m++){
             for(int n=0;n<6;n++){
                  Beta[m*6+n] = 0;
                  for(int s=0;s<6;s++){
                      Beta[m*6+n] += D[m]*P[m*6+s]*M[s*6+n];
                  }
              }
          }

          double dist = -x_0[i]+0.5;
          double cx = -16*pow(sqrt(delta*delta*h*h-dist*dist),3)/3/pi/pow(delta*h,4);
          double cxx = (2*dist*(2*dist*dist-delta*h*delta*h)*sqrt(delta*delta*h*h-dist*dist)/pow(delta*h,4)+2*atan(dist/sqrt(delta*delta*h*h-dist*dist))
                +pi)/pi;
          double cyy = (2*dist*(5*delta*delta*h*h-2*dist*dist)*sqrt(delta*delta*h*h-dist*dist)/3/pow(delta*h,4)+2*atan(dist/sqrt(delta*delta*h*h-dist*dist))+pi)/pi;
          
          for(int s=0;s<nei[i].size();s++){
            stiffKT[i*dim+nei[i][s]] -= Beta[s*6+1]*cx + Beta[s*6+3]*cxx + Beta[s*6+4]*cyy+Beta[s*6+4]*(0.5*(cxx-cyy)-cx*dist)*2;  
          }
            stiffKT[i*dim+i] += (0.5*cxx-cx*dist)/Dt;
            double x_proj = 0.5;
            double y_proj = y_0[i];
            Nodeid = search_projection(x_proj,y_proj);
            stiffKT[i*dim+Nodeid] += -cx*rbeta; 
        }
        else if((y_0[i]+0.5) <= delta*h+Tol && y_0[i]>-0.5-Tol && (x_0[i] >= 0.5-Tol || x_0[i]<= -0.5+Tol)){ 
            normal[0] = 0;
            normal[1] = -1;
            tanvec[0] = 1;
            tanvec[1] = 0;
            for(int s=0;s!=nei[i].size();s++){
              P[s*6] = 1;
              P[s*6+1] = (x_0[nei[i][s]] - x_0[i])*normal[0] + (y_0[nei[i][s]]-y_0[i])*normal[1]; 
              P[s*6+2] = (x_0[nei[i][s]] - x_0[i])*tanvec[0] + (y_0[nei[i][s]]-y_0[i])*tanvec[1];
              P[s*6+3] = pow((x_0[nei[i][s]] - x_0[i])*normal[0] + (y_0[nei[i][s]]-y_0[i])*normal[1],2);
              P[s*6+4] = pow((x_0[nei[i][s]] - x_0[i])*tanvec[0] + (y_0[nei[i][s]]-y_0[i])*tanvec[1],2);
              P[s*6+5] =((x_0[nei[i][s]] - x_0[i])*normal[0] + (y_0[nei[i][s]]-y_0[i])*normal[1])*((x_0[nei[i][s]] - x_0[i])*tanvec[0] + (y_0[nei[i][s]]-y_0[i])*tanvec[1]);
              D[s] =  weight(sqrt(pow(P[s*6+1],2)+pow(P[s*6+2],2)),delta*h);
            }
            for(int m=0;m<6;m++){
              for(int n=0;n<6;n++){
                  M[m*6+n] = 0;
                  for(int s=0;s<nei[i].size();s++){
                      M[m*6+n] += P[s*6+m]*D[s]*P[s*6+n];
                  }
              }
            }
        
          inverse(M,6);
          for(int m=0;m<nei[i].size();m++){
             for(int n=0;n<6;n++){
                  Beta[m*6+n] = 0;
                  for(int s=0;s<6;s++){
                      Beta[m*6+n] += D[m]*P[m*6+s]*M[s*6+n];
                  }
              }
          }

          double dist = y_0[i]+0.5;
          double cx = -16*pow(sqrt(delta*delta*h*h-dist*dist),3)/3/pi/pow(delta*h,4);
          double cxx = (2*dist*(2*dist*dist-delta*h*delta*h)*sqrt(delta*delta*h*h-dist*dist)/pow(delta*h,4)+2*atan(dist/sqrt(delta*delta*h*h-dist*dist))
                +pi)/pi;
          double cyy = (2*dist*(5*delta*delta*h*h-2*dist*dist)*sqrt(delta*delta*h*h-dist*dist)/3/pow(delta*h,4)+2*atan(dist/sqrt(delta*delta*h*h-dist*dist))+pi)/pi;
          
          for(int s=0;s<nei[i].size();s++){
            stiffKT[i*dim+nei[i][s]] -= Beta[s*6+1]*cx + Beta[s*6+3]*cxx + Beta[s*6+4]*cyy+Beta[s*6+4]*(0.5*(cxx-cyy)-cx*dist)*2;  
          }
            stiffKT[i*dim+i] += (0.5*cxx-cx*dist)/Dt;
            double x_proj = x_0[i];
            double y_proj = -0.5;
            Nodeid = search_projection(x_proj,y_proj);
            stiffKT[i*dim+Nodeid] += -cx*rbeta; 
        }
        else if((-y_0[i]+0.5) <= delta*h+Tol && y_0[i] < 0.5+Tol &&(x_0[i] >= 0.5-Tol || x_0[i]<= -0.5+Tol) ){ 
            normal[0] = 0;
            normal[1] = 1;
            tanvec[0] = -1;
            tanvec[1] = 0;
            for(int s=0;s!=nei[i].size();s++){
              P[s*6] = 1;
              P[s*6+1] = (x_0[nei[i][s]] - x_0[i])*normal[0] + (y_0[nei[i][s]]-y_0[i])*normal[1]; 
              P[s*6+2] = (x_0[nei[i][s]] - x_0[i])*tanvec[0] + (y_0[nei[i][s]]-y_0[i])*tanvec[1];
              P[s*6+3] = pow((x_0[nei[i][s]] - x_0[i])*normal[0] + (y_0[nei[i][s]]-y_0[i])*normal[1],2);
              P[s*6+4] = pow((x_0[nei[i][s]] - x_0[i])*tanvec[0] + (y_0[nei[i][s]]-y_0[i])*tanvec[1],2);
              P[s*6+5] =((x_0[nei[i][s]] - x_0[i])*normal[0] + (y_0[nei[i][s]]-y_0[i])*normal[1])*((x_0[nei[i][s]] - x_0[i])*tanvec[0] + (y_0[nei[i][s]]-y_0[i])*tanvec[1]);
              D[s] =  weight(sqrt(pow(P[s*6+1],2)+pow(P[s*6+2],2)),delta*h);
            }
            for(int m=0;m<6;m++){
              for(int n=0;n<6;n++){
                  M[m*6+n] = 0;
                  for(int s=0;s<nei[i].size();s++){
                      M[m*6+n] += P[s*6+m]*D[s]*P[s*6+n];
                  }
              }
            }
        
          inverse(M,6);
          for(int m=0;m<nei[i].size();m++){
             for(int n=0;n<6;n++){
                  Beta[m*6+n] = 0;
                  for(int s=0;s<6;s++){
                      Beta[m*6+n] += D[m]*P[m*6+s]*M[s*6+n];
                  }
              }
          }

          double dist = -y_0[i]+0.5;
          double cx = -16*pow(sqrt(delta*delta*h*h-dist*dist),3)/3/pi/pow(delta*h,4);
          double cxx = (2*dist*(2*dist*dist-delta*h*delta*h)*sqrt(delta*delta*h*h-dist*dist)/pow(delta*h,4)+2*atan(dist/sqrt(delta*delta*h*h-dist*dist))
                +pi)/pi;
          double cyy = (2*dist*(5*delta*delta*h*h-2*dist*dist)*sqrt(delta*delta*h*h-dist*dist)/3/pow(delta*h,4)+2*atan(dist/sqrt(delta*delta*h*h-dist*dist))+pi)/pi;
          
          for(int s=0;s<nei[i].size();s++){
            stiffKT[i*dim+nei[i][s]] -= Beta[s*6+1]*cx + Beta[s*6+3]*cxx + Beta[s*6+4]*cyy+Beta[s*6+4]*(0.5*(cxx-cyy)-cx*dist)*2;  
          }
            stiffKT[i*dim+i] += (0.5*cxx-cx*dist)/Dt;
            double x_proj = x_0[i];
            double y_proj = 0.5;
            Nodeid = search_projection(x_proj,y_proj);
            stiffKT[i*dim+Nodeid] += -cx*rbeta; 
        }
        else{
            for(int s=0;s<nei[i].size();s++){
                P[s*6] = 1;
                P[s*6+1] = x_0[nei[i][s]]-x_0[i];
                P[s*6+2] = y_0[nei[i][s]]-y_0[i];
                P[s*6+3] = pow(x_0[nei[i][s]]-x_0[i],2);
                P[s*6+4] = pow(y_0[nei[i][s]]-y_0[i],2);
                P[s*6+5] = (x_0[nei[i][s]]-x_0[i])*(y_0[nei[i][s]]-y_0[i]);
                D[s] =  weight(sqrt(pow(P[s*6+1],2)+pow(P[s*6+2],2)),delta*h); 
            }
      
            for(int m=0;m<6;m++){
                for(int n=0;n<6;n++){
                    M[m*6+n] = 0;
                    for(int s=0;s<nei[i].size();s++){
                        M[m*6+n] += P[s*6+m]*D[s]*P[s*6+n];
                    }
                }
            }
        
            inverse(M,6);
            for(int m=0;m<nei[i].size();m++){
              for(int n=0;n<6;n++){
                  Beta[m*6+n] = 0;
                  for(int s=0;s<6;s++){
                      Beta[m*6+n] += D[m]*P[m*6+s]*M[s*6+n];
                  }
              }
            }
            for(int s=0;s<nei[i].size();s++){
              stiffKT[i*dim+nei[i][s]]-= Beta[s*6+3]*2+Beta[s*6+4]*2; 
            }
            stiffKT[i*dim+i] += 1./Dt;
        }
        


          
        delete[] P;
        delete[] M;
        delete[] D;
        delete[] Beta;
  }
  stiffKT[0] = 1.0;
  /*for(int i=0;i<dim;i++){
    for(int j=0;j<dim;j++){
        cout<<stiffKT[i*dim+j]<<" ";
    }
    cout<<endl;
  }*/
}

void getBdycoord(int kdim, npArray coord){
    auto coord_buffer = coord.request();
    double *coord_ptr = (double *)coord_buffer.ptr; 

    localbdy = new double[2*kdim];
    
    for(int i=0;i<kdim;i++){
        localbdy[2*i] = coord_ptr[2*i];
        localbdy[2*i+1] = coord_ptr[2*i+1];
    }

    bdynei.resize(kdim);
    for(int i=0;i<kdim;i++){
        for(int j=0;j<N_point;j++){
            if(pow(x_0[j]-localbdy[2*i],2)+pow(y_0[j]-localbdy[2*i+1],2) < delta*delta*h*h){
                bdynei[i].push_back(j);
            }
        }
    }
    
    localbeta = new double*[kdim]; 
    
  for(int i=0;i<kdim;i++){
        double* P = new double[bdynei[i].size()*6];
        double* M = new double[6*6];
        double* D = new double[bdynei[i].size()]; 
        localbeta[i] = new double[bdynei[i].size()*6];

        for(int s=0;s<bdynei[i].size();s++){
           P[s*6] = 1;
           P[s*6+1] = x_0[bdynei[i][s]] - localbdy[2*i];  
           P[s*6+2] = y_0[bdynei[i][s]] - localbdy[2*i+1];
           P[s*6+3] = pow(x_0[bdynei[i][s]] - localbdy[2*i],2);
           P[s*6+4] = pow(y_0[bdynei[i][s]] - localbdy[2*i+1],2);
           P[s*6+5] =(x_0[bdynei[i][s]] - localbdy[2*i])*(y_0[bdynei[i][s]] - localbdy[2*i+1]); 
           D[s] =  weight(sqrt(pow(P[s*6+1],2)+pow(P[s*6+2],2)),delta*h);
        }

        for(int m=0;m<6;m++){
            for(int n=0;n<6;n++){
                  M[m*6+n] = 0;
                  for(int s=0;s<bdynei[i].size();s++){
                      M[m*6+n] += P[s*6+m]*D[s]*P[s*6+n];
                  }
              }
          }
        
          inverse(M,6);
            
          for(int m=0;m<bdynei[i].size();m++){
              for(int n=0;n<6;n++){
                  localbeta[i][m*6+n] = 0;
                  for(int s=0;s<6;s++){
                      localbeta[i][m*6+n] += D[m]*P[m*6+s]*M[s*6+n];
                  }
              }
          }
        
        }

}

void projBdy(npArray pdim, npArray proj){
    auto pdim_buffer = pdim.request();
    double *pdim_ptr = (double *)pdim_buffer.ptr;
    auto proj_buffer = proj.request();
    double *proj_ptr = (double *)proj_buffer.ptr;
    
    int pp = 0;
    for(int i=1;i<N_point;i++){
        if(abs(x_0[i]+1) <= delta*h+Tol && abs(y_0[i]-0.5)<=delta*h+Tol){  
            proj_ptr[4*pp] = -1;
            proj_ptr[4*pp+1] = y_0[i];
            proj_ptr[4*pp+2] = -1;
            proj_ptr[4*pp+3] = 0;
            proj_ptr[4*pp+4] = x_0[i];
            proj_ptr[4*pp+5] = 0.5;
            proj_ptr[4*pp+6] = 0;
            proj_ptr[4*pp+7] = 1;
            pp = pp + 2;
        }
        else if(abs(x_0[i]+1) <= delta*h+Tol && abs(y_0[i]+0.5)<=delta*h+Tol){
            proj_ptr[4*pp] = -1;
            proj_ptr[4*pp+1] = y_0[i];
            proj_ptr[4*pp+2] = -1;
            proj_ptr[4*pp+3] = 0;
            proj_ptr[4*pp+4] = x_0[i];
            proj_ptr[4*pp+5] = -0.5;
            proj_ptr[4*pp+6] = 0;
            proj_ptr[4*pp+7] = -1;
            pp = pp + 2;
        } 
        else if(abs(x_0[i]+0.5) <= delta*h+Tol && abs(y_0[i]-1)<=delta*h+Tol){
            proj_ptr[4*pp] = -0.5;
            proj_ptr[4*pp+1] = y_0[i];
            proj_ptr[4*pp+2] = -1;
            proj_ptr[4*pp+3] = 0;
            proj_ptr[4*pp+4] = x_0[i];
            proj_ptr[4*pp+5] = 1;
            proj_ptr[4*pp+6] = 0;
            proj_ptr[4*pp+7] = 1;
            pp = pp + 2;
        }
        else if(abs(x_0[i]-0.5) <= delta*h+Tol && abs(y_0[i]-1)<=delta*h+Tol){
            proj_ptr[4*pp] = 0.5;
            proj_ptr[4*pp+1] = y_0[i];
            proj_ptr[4*pp+2] = 1;
            proj_ptr[4*pp+3] = 0;
            proj_ptr[4*pp+4] = x_0[i];
            proj_ptr[4*pp+5] = 1;
            proj_ptr[4*pp+6] = 0;
            proj_ptr[4*pp+7] = 1;
            pp = pp + 2;
        }
        else if(abs(x_0[i]-1) <= delta*h+Tol && abs(y_0[i]-0.5)<=delta*h+Tol){
            proj_ptr[4*pp] = 1;
            proj_ptr[4*pp+1] = y_0[i];
            proj_ptr[4*pp+2] = 1;
            proj_ptr[4*pp+3] = 0;
            proj_ptr[4*pp+4] = x_0[i];
            proj_ptr[4*pp+5] = 0.5;
            proj_ptr[4*pp+6] = 0;
            proj_ptr[4*pp+7] = 1;
            pp = pp + 2;
        }
        else if(abs(x_0[i]-1) <= delta*h+Tol && abs(y_0[i]+0.5)<=delta*h+Tol){
            proj_ptr[4*pp] = 1;
            proj_ptr[4*pp+1] = y_0[i];
            proj_ptr[4*pp+2] = 1;
            proj_ptr[4*pp+3] = 0;
            proj_ptr[4*pp+4] = x_0[i];
            proj_ptr[4*pp+5] = -0.5;
            proj_ptr[4*pp+6] = 0;
            proj_ptr[4*pp+7] = -1;
            pp = pp + 2;
        }
        else if(abs(x_0[i]+0.5) <= delta*h+Tol && abs(y_0[i]+1)<=delta*h+Tol){
            proj_ptr[4*pp] = -0.5;
            proj_ptr[4*pp+1] = y_0[i];
            proj_ptr[4*pp+2] = -1;
            proj_ptr[4*pp+3] = 0;
            proj_ptr[4*pp+4] = x_0[i];
            proj_ptr[4*pp+5] = -1;
            proj_ptr[4*pp+6] = 0;
            proj_ptr[4*pp+7] = -1;
            pp = pp + 2;
        }
        else if(abs(x_0[i]-0.5) <= delta*h+Tol && abs(y_0[i]+1)<=delta*h+Tol){
            proj_ptr[4*pp] = 0.5;
            proj_ptr[4*pp+1] = y_0[i];
            proj_ptr[4*pp+2] = 1;
            proj_ptr[4*pp+3] = 0;
            proj_ptr[4*pp+4] = x_0[i];
            proj_ptr[4*pp+5] = -1;
            proj_ptr[4*pp+6] = 0;
            proj_ptr[4*pp+7] = -1;
            pp = pp + 2;
        }
        else if(ifinhorizon(x_0[i],y_0[i],-0.5,0.5)){
          double dx = 0.5 + x_0[i];
          double dy = 0.5 - y_0[i];
          if(dx > 0 && dy > 0){
            proj_ptr[4*pp] = -0.5;
            proj_ptr[4*pp+1] = 0.5;
            proj_ptr[4*pp+2] = -1;
            proj_ptr[4*pp+3] = 0;
            proj_ptr[4*pp+4] = -0.5;
            proj_ptr[4*pp+5] = 0.5;
            proj_ptr[4*pp+6] = 0;
            proj_ptr[4*pp+7] = 1;
            pp = pp + 2;
          }  
          if(dx >= 0-Tol && dy <= 0+Tol){
            proj_ptr[4*pp] = -0.5;
            proj_ptr[4*pp+1] = y_0[i];
            proj_ptr[4*pp+2] = -1;
            proj_ptr[4*pp+3] = 0;
            proj_ptr[4*pp+4] = -0.5;
            proj_ptr[4*pp+5] = 0.5;
            proj_ptr[4*pp+6] = 0;
            proj_ptr[4*pp+7] = 1;
            pp = pp + 2;
          }
          if(dx <= 0+Tol && dy >=0-Tol && !(fabs(dx)<= 0+Tol && fabs(dy) <=0+Tol)){
            proj_ptr[4*pp] = -0.5;
            proj_ptr[4*pp+1] = 0.5;
            proj_ptr[4*pp+2] = -1;
            proj_ptr[4*pp+3] = 0;
            proj_ptr[4*pp+4] = x_0[i];
            proj_ptr[4*pp+5] = 0.5;
            proj_ptr[4*pp+6] = 0;
            proj_ptr[4*pp+7] = 1;
            pp = pp+2;
            }
        }
        else if(ifinhorizon(x_0[i],y_0[i],0.5,0.5)){
          double dx = 0.5 - x_0[i];
          double dy = 0.5 - y_0[i];
          if(dx > 0 && dy > 0){
            proj_ptr[4*pp] = 0.5;
            proj_ptr[4*pp+1] = 0.5;
            proj_ptr[4*pp+2] = 1;
            proj_ptr[4*pp+3] = 0;
            proj_ptr[4*pp+4] = 0.5;
            proj_ptr[4*pp+5] = 0.5;
            proj_ptr[4*pp+6] = 0;
            proj_ptr[4*pp+7] = 1;
            pp = pp + 2;
          }  
          if(dx >= 0-Tol && dy <= 0+Tol){
            proj_ptr[4*pp] = 0.5;
            proj_ptr[4*pp+1] = y_0[i];
            proj_ptr[4*pp+2] = 1;
            proj_ptr[4*pp+3] = 0;
            proj_ptr[4*pp+4] = 0.5;
            proj_ptr[4*pp+5] = 0.5;
            proj_ptr[4*pp+6] = 0;
            proj_ptr[4*pp+7] = 1;
            pp = pp + 2;
          }
          if(dx <= 0+Tol && dy >=0-Tol && !(fabs(dx)<= 0+Tol && fabs(dy) <=0+Tol)){
            proj_ptr[4*pp] = 0.5;
            proj_ptr[4*pp+1] = 0.5;
            proj_ptr[4*pp+2] = 1;
            proj_ptr[4*pp+3] = 0;
            proj_ptr[4*pp+4] = x_0[i];
            proj_ptr[4*pp+5] = 0.5;
            proj_ptr[4*pp+6] = 0;
            proj_ptr[4*pp+7] = 1;
            pp = pp + 2;
            }
        }
        else if(ifinhorizon(x_0[i],y_0[i],-0.5,-0.5)){
          double dx = 0.5 + x_0[i];
          double dy = 0.5 + y_0[i];
          if(dx > 0 && dy > 0){
            proj_ptr[4*pp] = -0.5;
            proj_ptr[4*pp+1] = -0.5;
            proj_ptr[4*pp+2] = -1;
            proj_ptr[4*pp+3] = 0;
            proj_ptr[4*pp+4] = -0.5;
            proj_ptr[4*pp+5] = -0.5;
            proj_ptr[4*pp+6] = 0;
            proj_ptr[4*pp+7] = -1;
            pp = pp + 2;
          }  
          if(dx >= 0-Tol && dy <= 0+Tol){
            proj_ptr[4*pp] = -0.5;
            proj_ptr[4*pp+1] = y_0[i];
            proj_ptr[4*pp+2] = -1;
            proj_ptr[4*pp+3] = 0;
            proj_ptr[4*pp+4] = -0.5;
            proj_ptr[4*pp+5] = -0.5;
            proj_ptr[4*pp+6] = 0;
            proj_ptr[4*pp+7] = -1;
            pp = pp + 2;
          }
          if(dx <= 0+Tol && dy >=0-Tol && !(fabs(dx)<= 0+Tol && fabs(dy) <=0+Tol)){
            proj_ptr[4*pp] = -0.5;
            proj_ptr[4*pp+1] = -0.5;
            proj_ptr[4*pp+2] = -1;
            proj_ptr[4*pp+3] = 0;
            proj_ptr[4*pp+4] = x_0[i];
            proj_ptr[4*pp+5] = -0.5;
            proj_ptr[4*pp+6] = 0;
            proj_ptr[4*pp+7] = -1;
            pp = pp + 2;
            }
        }
        else if(ifinhorizon(x_0[i],y_0[i],0.5,-0.5)){
          double dx = 0.5 - x_0[i];
          double dy = 0.5 + y_0[i];
          if(dx > 0 && dy > 0){
            proj_ptr[4*pp] = 0.5;
            proj_ptr[4*pp+1] = -0.5;
            proj_ptr[4*pp+2] = 1;
            proj_ptr[4*pp+3] = 0;
            proj_ptr[4*pp+4] = 0.5;
            proj_ptr[4*pp+5] = -0.5;
            proj_ptr[4*pp+6] = 0;
            proj_ptr[4*pp+7] = -1;
            pp = pp + 2;
          }  
          if(dx >= 0-Tol && dy <= 0+Tol){
            proj_ptr[4*pp] = 0.5;
            proj_ptr[4*pp+1] = y_0[i];
            proj_ptr[4*pp+2] = 1;
            proj_ptr[4*pp+3] = 0;
            proj_ptr[4*pp+4] = 0.5;
            proj_ptr[4*pp+5] = -0.5;
            proj_ptr[4*pp+6] = 0;
            proj_ptr[4*pp+7] = -1;
            pp = pp + 2;
          }
          if(dx <= 0+Tol && dy >=0-Tol && !(fabs(dx)<= 0+Tol && fabs(dy) <=0+Tol)){
            proj_ptr[4*pp] = 0.5;
            proj_ptr[4*pp+1] = -0.5;
            proj_ptr[4*pp+2] = 1;
            proj_ptr[4*pp+3] = 0;
            proj_ptr[4*pp+4] = x_0[i];
            proj_ptr[4*pp+5] = -0.5;
            proj_ptr[4*pp+6] = 0;
            proj_ptr[4*pp+7] = -1;
            pp = pp + 2;
            }
        }
        else if(abs(x_0[i]+1) <= delta*h+Tol){
            proj_ptr[4*pp] = -1;
            proj_ptr[4*pp+1] = y_0[i];
            proj_ptr[4*pp+2] = -1;
            proj_ptr[4*pp+3] = 0;
            pp = pp +1;
        }
        else if(abs(x_0[i]-1) <= delta*h+Tol){
            proj_ptr[4*pp] = 1;
            proj_ptr[4*pp+1] = y_0[i];
            proj_ptr[4*pp+2] = 1;
            proj_ptr[4*pp+3] = 0;
            pp = pp +1;
        }
        else if(abs(y_0[i]+1) <= delta*h+Tol){
            proj_ptr[4*pp] = x_0[i];
            proj_ptr[4*pp+1] = -1;
            proj_ptr[4*pp+2] = 0;
            proj_ptr[4*pp+3] = -1;
            pp = pp +1;
        }
        else if(abs(y_0[i]-1) <= delta*h+Tol){
            proj_ptr[4*pp] = x_0[i];
            proj_ptr[4*pp+1] = 1;
            proj_ptr[4*pp+2] = 0;
            proj_ptr[4*pp+3] = 1;
            pp = pp +1;
        }
        else if((x_0[i]+0.5) <= delta*h+Tol && x_0[i] > -0.5-Tol && (y_0[i] >= 0.5-Tol || y_0[i]<= -0.5+Tol)){
            proj_ptr[4*pp] = -0.5;
            proj_ptr[4*pp+1] = y_0[i];
            proj_ptr[4*pp+2] = -1;
            proj_ptr[4*pp+3] = 0;
            pp = pp +1;
        }
        else if((-x_0[i]+0.5) <= delta*h+Tol && x_0[i] < 0.5+Tol && (y_0[i] >= 0.5-Tol || y_0[i]<= -0.5+Tol)){
            proj_ptr[4*pp] = 0.5;
            proj_ptr[4*pp+1] = y_0[i];
            proj_ptr[4*pp+2] = 1;
            proj_ptr[4*pp+3] = 0;
            pp = pp +1;
        }
        else if((y_0[i]+0.5) <= delta*h+Tol && y_0[i] > -0.5-Tol && (x_0[i] >= 0.5-Tol || x_0[i]<= -0.5+Tol) ){
            proj_ptr[4*pp] = x_0[i];
            proj_ptr[4*pp+1] = -0.5;
            proj_ptr[4*pp+2] = 0;
            proj_ptr[4*pp+3] = -1;
            pp = pp +1;
        }
        else if((-y_0[i]+0.5) <= delta*h+Tol && y_0[i] < 0.5 +Tol&& (x_0[i] >= 0.5-Tol || x_0[i]<= -0.5+Tol)){
            proj_ptr[4*pp] = x_0[i];
            proj_ptr[4*pp+1] = 0.5;
            proj_ptr[4*pp+2] = 0;
            proj_ptr[4*pp+3] = 1;
            pp = pp +1;
        }
    }
    pdim_ptr[0] = (double)pp;
}

void BE(int s,int kdim, npArray v, npArray g,int UPDATE){
        auto v_buffer  = v.request();
        double *v_ptr = (double *)v_buffer.ptr; 
        auto g_buffer = g.request();
        double *g_ptr = (double *)g_buffer.ptr;
        
        for(int i=0;i<kdim;i++){
            v_ptr[i] = 0;
        }

        IPIV = new int[dim];
        int pdim = 0;

        for(int i=0;i<N_point;i++){
            rhs[i] = 0; 
            //u0[i] = u_exact(x_0[i],y_0[i],s*Dt);
        }
        for(int i=1;i<N_point;i++){

        if(abs(x_0[i]+1) <= delta*h+Tol && abs(y_0[i]-0.5)<=delta*h+Tol){  
          normal[0] = -1;
          normal[1] = 0;
          tanvec[0] = 0;
          tanvec[1] = 1;
          double dx = 1 + x_0[i];
          double dy = 0.5 - y_0[i];
          double x_proj1 = -1;
          double y_proj1 = y_0[i];
          double x_proj2 = x_0[i];
          double y_proj2 = 0.5;

          if(ifinhorizon(x_0[i],y_0[i],-1,0.5)){
            double cx = 8*(-pow(dy,3)/3-pow(sqrt(delta*delta*h*h-dx*dx),3)/3+dy*(dx*dx+dy*dy-delta*delta*h*h)/2)/pi/pow(delta*h,4);
            double cy = 8*(-pow(dx,3)/3-pow(sqrt(delta*delta*h*h-dy*dy),3)/3+dx*(dx*dx+dy*dy-delta*delta*h*h)/2)/pi/pow(delta*h,4);
            double cxx = ((dx*(2*dx*dx-delta*delta*h*h)*sqrt(delta*delta*h*h-dx*dx)-sqrt(delta*delta*h*h-dy*dy)*(delta*delta*h*h-2*dy*dy)*dy)/pow(delta*h,4)
                +atan(dx/sqrt(delta*delta*h*h-dx*dx))+atan(-sqrt(delta*delta*h*h-dy*dy)/dy)+pi)/pi
                +8*dy*(pow(dx,3)+pow(sqrt(delta*delta*h*h-dy*dy),3))/3/pi/pow(delta*h,4);
            double cyy = ((dy*(2*dy*dy-delta*delta*h*h)*sqrt(delta*delta*h*h-dy*dy)-sqrt(delta*delta*h*h-dx*dx)*(delta*delta*h*h-2*dx*dx)*dx)/pow(delta*h,4)
                +atan(dy/sqrt(delta*delta*h*h-dy*dy))+atan(-sqrt(delta*delta*h*h-dx*dx)/dx)+pi)/pi
                +8*dx*(pow(dy,3)+pow(sqrt(delta*delta*h*h-dx*dx),3))/3/pi/pow(delta*h,4);
            double cxy = pow(dx*dx+dy*dy-delta*delta*h*h,2)/pi/pow(delta*h,4);
            //rhs[i] = (0.5*cxx-cx*dx)*Ffun(x_0[i],y_0[i],s*Dt)-cx*(g_ptr[2*pdim]*normal[0]+g_ptr[2*pdim+1]*normal[1])-cy*(g_ptr[2*pdim+2]*tanvec[0]+g_ptr[2*pdim+3]*tanvec[1])
            rhs[i] = (0.5*cxx-cx*dx)*Ffun(x_0[i],y_0[i],s*Dt)-cx*g_ptr[pdim]-cy*g_ptr[pdim+1]
            //rhs[i] = (0.5*cxx-cx*dx)*Ffun(x_0[i],y_0[i],s*Dt)-cx*Gfun(x_proj1,y_proj1,normal[0],normal[1],s*Dt)-cy*Gfun(x_proj2,y_proj2,tanvec[0],tanvec[1],s*Dt)
              -cxy*0.5*(dGfuny(x_proj1,y_proj1,normal[0],normal[1],s*Dt)-dGfunx(x_proj2,y_proj2,tanvec[0],tanvec[1],s*Dt));
            rhs[i] += (0.5*cxx-cx*dx)*u0[i]/Dt;
          }
          else{
            double cx = -16*pow(sqrt(delta*delta*h*h-dx*dx),3)/3/pi/pow(delta*h,4);
            double cy = -16*pow(sqrt(delta*delta*h*h-dy*dy),3)/3/pi/pow(delta*h,4);
            double cxx = 2*((dx*(2*dx*dx-delta*delta*h*h)*sqrt(delta*delta*h*h-dx*dx)+dy*(5*delta*delta*h*h-2*dy*dy)*sqrt(delta*delta*h*h-dy*dy)/3)/pow(delta*h,4)
                +atan(dx/sqrt(delta*delta*h*h-dx*dx))+atan(dy/sqrt(delta*delta*h*h-dx*dx)))/pi; 
            double cyy = 2*((dy*(2*dy*dy-delta*delta*h*h)*sqrt(delta*delta*h*h-dy*dy)+dx*(5*delta*delta*h*h-2*dx*dx)*sqrt(delta*delta*h*h-dx*dx)/3)/pow(delta*h,4)
                +atan(dx/sqrt(delta*delta*h*h-dx*dx))+atan(dy/sqrt(delta*delta*h*h-dx*dx)))/pi; 
            //rhs[i] = (0.5*cxx-cx*dx)*Ffun(x_0[i],y_0[i],s*Dt)-cx*(g_ptr[2*pdim]*normal[0]+g_ptr[2*pdim+1]*normal[1])-cy*(g_ptr[2*pdim+2]*tanvec[0]+g_ptr[2*pdim+3]*tanvec[1]);
            rhs[i] = (0.5*cxx-cx*dx)*Ffun(x_0[i],y_0[i],s*Dt)-cx*g_ptr[pdim]-cy*g_ptr[pdim+1];
            //rhs[i] = (0.5*cxx-cx*dx)*Ffun(x_0[i],y_0[i],s*Dt)-cx*Gfun(x_proj1,y_proj1,normal[0],normal[1],s*Dt)-cy*Gfun(x_proj2,y_proj2,tanvec[0],tanvec[1],s*Dt);
            rhs[i] += (0.5*cxx-cx*dx)*u0[i]/Dt;
          }
          pdim = pdim+2;
        }
        else if(abs(x_0[i]+1) <= delta*h+Tol && abs(y_0[i]+0.5)<=delta*h+Tol){
          normal[0] = -1;
          normal[1] = 0;
          tanvec[0] = 0;
          tanvec[1] = -1;
          double dx = 1 + x_0[i];
          double dy = 0.5 + y_0[i];
          double x_proj1 = -1;
          double y_proj1 = y_0[i];
          double x_proj2 = x_0[i];
          double y_proj2 = -0.5;

          if(ifinhorizon(x_0[i],y_0[i],-1,-0.5)){
            double cx = 8*(-pow(dy,3)/3-pow(sqrt(delta*delta*h*h-dx*dx),3)/3+dy*(dx*dx+dy*dy-delta*delta*h*h)/2)/pi/pow(delta*h,4);
            double cy = 8*(-pow(dx,3)/3-pow(sqrt(delta*delta*h*h-dy*dy),3)/3+dx*(dx*dx+dy*dy-delta*delta*h*h)/2)/pi/pow(delta*h,4);
            double cxx = ((dx*(2*dx*dx-delta*delta*h*h)*sqrt(delta*delta*h*h-dx*dx)-sqrt(delta*delta*h*h-dy*dy)*(delta*delta*h*h-2*dy*dy)*dy)/pow(delta*h,4)
                +atan(dx/sqrt(delta*delta*h*h-dx*dx))+atan(-sqrt(delta*delta*h*h-dy*dy)/dy)+pi)/pi
                +8*dy*(pow(dx,3)+pow(sqrt(delta*delta*h*h-dy*dy),3))/3/pi/pow(delta*h,4);
            double cyy = ((dy*(2*dy*dy-delta*delta*h*h)*sqrt(delta*delta*h*h-dy*dy)-sqrt(delta*delta*h*h-dx*dx)*(delta*delta*h*h-2*dx*dx)*dx)/pow(delta*h,4)
                +atan(dy/sqrt(delta*delta*h*h-dy*dy))+atan(-sqrt(delta*delta*h*h-dx*dx)/dx)+pi)/pi
                +8*dx*(pow(dy,3)+pow(sqrt(delta*delta*h*h-dx*dx),3))/3/pi/pow(delta*h,4);
            double cxy = pow(dx*dx+dy*dy-delta*delta*h*h,2)/pi/pow(delta*h,4);
            //rhs[i] = (0.5*cxx-cx*dx)*Ffun(x_0[i],y_0[i],s*Dt)-cx*(g_ptr[2*pdim]*normal[0]+g_ptr[2*pdim+1]*normal[1])-cy*(g_ptr[2*pdim+2]*tanvec[0]+g_ptr[2*pdim+3]*tanvec[1])
            rhs[i] = (0.5*cxx-cx*dx)*Ffun(x_0[i],y_0[i],s*Dt)-cx*g_ptr[pdim]-cy*g_ptr[pdim+1]
            //rhs[i] = (0.5*cxx-cx*dx)*Ffun(x_0[i],y_0[i],s*Dt)-cx*Gfun(x_proj1,y_proj1,normal[0],normal[1],s*Dt)-cy*Gfun(x_proj2,y_proj2,tanvec[0],tanvec[1],s*Dt)
              -cxy*0.5*(-dGfuny(x_proj1,y_proj1,normal[0],normal[1],s*Dt)-dGfunx(x_proj2,y_proj2,tanvec[0],tanvec[1],s*Dt));
            rhs[i] += (0.5*cxx-cx*dx)*u0[i]/Dt;
          }
          else{
            double cx = -16*pow(sqrt(delta*delta*h*h-dx*dx),3)/3/pi/pow(delta*h,4);
            double cy = -16*pow(sqrt(delta*delta*h*h-dy*dy),3)/3/pi/pow(delta*h,4);
            double cxx = 2*((dx*(2*dx*dx-delta*delta*h*h)*sqrt(delta*delta*h*h-dx*dx)+dy*(5*delta*delta*h*h-2*dy*dy)*sqrt(delta*delta*h*h-dy*dy)/3)/pow(delta*h,4)
                +atan(dx/sqrt(delta*delta*h*h-dx*dx))+atan(dy/sqrt(delta*delta*h*h-dx*dx)))/pi; 
            double cyy = 2*((dy*(2*dy*dy-delta*delta*h*h)*sqrt(delta*delta*h*h-dy*dy)+dx*(5*delta*delta*h*h-2*dx*dx)*sqrt(delta*delta*h*h-dx*dx)/3)/pow(delta*h,4)
                +atan(dx/sqrt(delta*delta*h*h-dx*dx))+atan(dy/sqrt(delta*delta*h*h-dx*dx)))/pi; 
            //rhs[i] = (0.5*cxx-cx*dx)*Ffun(x_0[i],y_0[i],s*Dt)-cx*(g_ptr[2*pdim]*normal[0]+g_ptr[2*pdim+1]*normal[1])-cy*(g_ptr[2*pdim+2]*tanvec[0]+g_ptr[2*pdim+3]*tanvec[1]);
            rhs[i] = (0.5*cxx-cx*dx)*Ffun(x_0[i],y_0[i],s*Dt)-cx*g_ptr[pdim]-cy*g_ptr[pdim+1];
            //rhs[i] = (0.5*cxx-cx*dx)*Ffun(x_0[i],y_0[i],s*Dt)-cx*Gfun(x_proj1,y_proj1,normal[0],normal[1],s*Dt)-cy*Gfun(x_proj2,y_proj2,tanvec[0],tanvec[1],s*Dt);
            rhs[i] += (0.5*cxx-cx*dx)*u0[i]/Dt;
          }
          pdim = pdim+2;
        }
        else if(abs(x_0[i]+0.5) <= delta*h+Tol && abs(y_0[i]-1)<=delta*h+Tol){
          normal[0] = -1;
          normal[1] = 0;
          tanvec[0] = 0;
          tanvec[1] = 1;
          double dx = 0.5 + x_0[i];
          double dy = 1 - y_0[i];
          double x_proj1 = -0.5;
          double y_proj1 = y_0[i];
          double x_proj2 = x_0[i];
          double y_proj2 = 1;

          if(ifinhorizon(x_0[i],y_0[i],-0.5,1)){
            double cx = 8*(-pow(dy,3)/3-pow(sqrt(delta*delta*h*h-dx*dx),3)/3+dy*(dx*dx+dy*dy-delta*delta*h*h)/2)/pi/pow(delta*h,4);
            double cy = 8*(-pow(dx,3)/3-pow(sqrt(delta*delta*h*h-dy*dy),3)/3+dx*(dx*dx+dy*dy-delta*delta*h*h)/2)/pi/pow(delta*h,4);
            double cxx = ((dx*(2*dx*dx-delta*delta*h*h)*sqrt(delta*delta*h*h-dx*dx)-sqrt(delta*delta*h*h-dy*dy)*(delta*delta*h*h-2*dy*dy)*dy)/pow(delta*h,4)
                +atan(dx/sqrt(delta*delta*h*h-dx*dx))+atan(-sqrt(delta*delta*h*h-dy*dy)/dy)+pi)/pi
                +8*dy*(pow(dx,3)+pow(sqrt(delta*delta*h*h-dy*dy),3))/3/pi/pow(delta*h,4);
            double cyy = ((dy*(2*dy*dy-delta*delta*h*h)*sqrt(delta*delta*h*h-dy*dy)-sqrt(delta*delta*h*h-dx*dx)*(delta*delta*h*h-2*dx*dx)*dx)/pow(delta*h,4)
                +atan(dy/sqrt(delta*delta*h*h-dy*dy))+atan(-sqrt(delta*delta*h*h-dx*dx)/dx)+pi)/pi
                +8*dx*(pow(dy,3)+pow(sqrt(delta*delta*h*h-dx*dx),3))/3/pi/pow(delta*h,4);
            double cxy = pow(dx*dx+dy*dy-delta*delta*h*h,2)/pi/pow(delta*h,4);
            //rhs[i] = (0.5*cxx-cx*dx)*Ffun(x_0[i],y_0[i],s*Dt)-cx*(g_ptr[2*pdim]*normal[0]+g_ptr[2*pdim+1]*normal[1])-cy*(g_ptr[2*pdim+2]*tanvec[0]+g_ptr[2*pdim+3]*tanvec[1])
            rhs[i] = (0.5*cxx-cx*dx)*Ffun(x_0[i],y_0[i],s*Dt)-cx*g_ptr[pdim]-cy*g_ptr[pdim+1]
            //rhs[i] = (0.5*cxx-cx*dx)*Ffun(x_0[i],y_0[i],s*Dt)-cx*Gfun(x_proj1,y_proj1,normal[0],normal[1],s*Dt)-cy*Gfun(x_proj2,y_proj2,tanvec[0],tanvec[1],s*Dt)
              -cxy*0.5*(dGfuny(x_proj1,y_proj1,normal[0],normal[1],s*Dt)-dGfunx(x_proj2,y_proj2,tanvec[0],tanvec[1],s*Dt));
            rhs[i] += (0.5*cxx-cx*dx)*u0[i]/Dt;
          }
          else{
            double cx = -16*pow(sqrt(delta*delta*h*h-dx*dx),3)/3/pi/pow(delta*h,4);
            double cy = -16*pow(sqrt(delta*delta*h*h-dy*dy),3)/3/pi/pow(delta*h,4);
            double cxx = 2*((dx*(2*dx*dx-delta*delta*h*h)*sqrt(delta*delta*h*h-dx*dx)+dy*(5*delta*delta*h*h-2*dy*dy)*sqrt(delta*delta*h*h-dy*dy)/3)/pow(delta*h,4)
                +atan(dx/sqrt(delta*delta*h*h-dx*dx))+atan(dy/sqrt(delta*delta*h*h-dx*dx)))/pi; 
            double cyy = 2*((dy*(2*dy*dy-delta*delta*h*h)*sqrt(delta*delta*h*h-dy*dy)+dx*(5*delta*delta*h*h-2*dx*dx)*sqrt(delta*delta*h*h-dx*dx)/3)/pow(delta*h,4)
                +atan(dx/sqrt(delta*delta*h*h-dx*dx))+atan(dy/sqrt(delta*delta*h*h-dx*dx)))/pi; 
            //rhs[i] = (0.5*cxx-cx*dx)*Ffun(x_0[i],y_0[i],s*Dt)-cx*(g_ptr[2*pdim]*normal[0]+g_ptr[2*pdim+1]*normal[1])-cy*(g_ptr[2*pdim+2]*tanvec[0]+g_ptr[2*pdim+3]*tanvec[1]);
            rhs[i] = (0.5*cxx-cx*dx)*Ffun(x_0[i],y_0[i],s*Dt)-cx*g_ptr[pdim]-cy*g_ptr[pdim+1];
            //rhs[i] = (0.5*cxx-cx*dx)*Ffun(x_0[i],y_0[i],s*Dt)-cx*Gfun(x_proj1,y_proj1,normal[0],normal[1],s*Dt)-cy*Gfun(x_proj2,y_proj2,tanvec[0],tanvec[1],s*Dt);
            rhs[i] += (0.5*cxx-cx*dx)*u0[i]/Dt;
          }
          pdim = pdim+2;
        }
        else if(abs(x_0[i]-0.5) <= delta*h+Tol && abs(y_0[i]-1)<=delta*h+Tol){
          normal[0] = 1;
          normal[1] = 0;
          tanvec[0] = 0;
          tanvec[1] = 1;
          double dx = 0.5 - x_0[i];
          double dy = 1 - y_0[i];
          double x_proj1 = 0.5;
          double y_proj1 = y_0[i];
          double x_proj2 = x_0[i];
          double y_proj2 = 1;

          if(ifinhorizon(x_0[i],y_0[i],0.5,1)){
            double cx = 8*(-pow(dy,3)/3-pow(sqrt(delta*delta*h*h-dx*dx),3)/3+dy*(dx*dx+dy*dy-delta*delta*h*h)/2)/pi/pow(delta*h,4);
            double cy = 8*(-pow(dx,3)/3-pow(sqrt(delta*delta*h*h-dy*dy),3)/3+dx*(dx*dx+dy*dy-delta*delta*h*h)/2)/pi/pow(delta*h,4);
            double cxx = ((dx*(2*dx*dx-delta*delta*h*h)*sqrt(delta*delta*h*h-dx*dx)-sqrt(delta*delta*h*h-dy*dy)*(delta*delta*h*h-2*dy*dy)*dy)/pow(delta*h,4)
                +atan(dx/sqrt(delta*delta*h*h-dx*dx))+atan(-sqrt(delta*delta*h*h-dy*dy)/dy)+pi)/pi
                +8*dy*(pow(dx,3)+pow(sqrt(delta*delta*h*h-dy*dy),3))/3/pi/pow(delta*h,4);
            double cyy = ((dy*(2*dy*dy-delta*delta*h*h)*sqrt(delta*delta*h*h-dy*dy)-sqrt(delta*delta*h*h-dx*dx)*(delta*delta*h*h-2*dx*dx)*dx)/pow(delta*h,4)
                +atan(dy/sqrt(delta*delta*h*h-dy*dy))+atan(-sqrt(delta*delta*h*h-dx*dx)/dx)+pi)/pi
                +8*dx*(pow(dy,3)+pow(sqrt(delta*delta*h*h-dx*dx),3))/3/pi/pow(delta*h,4);
            double cxy = pow(dx*dx+dy*dy-delta*delta*h*h,2)/pi/pow(delta*h,4);
            //rhs[i] = (0.5*cxx-cx*dx)*Ffun(x_0[i],y_0[i],s*Dt)-cx*(g_ptr[2*pdim]*normal[0]+g_ptr[2*pdim+1]*normal[1])-cy*(g_ptr[2*pdim+2]*tanvec[0]+g_ptr[2*pdim+3]*tanvec[1])
            rhs[i] = (0.5*cxx-cx*dx)*Ffun(x_0[i],y_0[i],s*Dt)-cx*g_ptr[pdim]-cy*g_ptr[pdim+1]
            //rhs[i] = (0.5*cxx-cx*dx)*Ffun(x_0[i],y_0[i],s*Dt)-cx*Gfun(x_proj1,y_proj1,normal[0],normal[1],s*Dt)-cy*Gfun(x_proj2,y_proj2,tanvec[0],tanvec[1],s*Dt)
              -cxy*0.5*(dGfuny(x_proj1,y_proj1,normal[0],normal[1],s*Dt)+dGfunx(x_proj2,y_proj2,tanvec[0],tanvec[1],s*Dt));
            rhs[i] += (0.5*cxx-cx*dx)*u0[i]/Dt;
          }
          else{
            double cx = -16*pow(sqrt(delta*delta*h*h-dx*dx),3)/3/pi/pow(delta*h,4);
            double cy = -16*pow(sqrt(delta*delta*h*h-dy*dy),3)/3/pi/pow(delta*h,4);
            double cxx = 2*((dx*(2*dx*dx-delta*delta*h*h)*sqrt(delta*delta*h*h-dx*dx)+dy*(5*delta*delta*h*h-2*dy*dy)*sqrt(delta*delta*h*h-dy*dy)/3)/pow(delta*h,4)
                +atan(dx/sqrt(delta*delta*h*h-dx*dx))+atan(dy/sqrt(delta*delta*h*h-dx*dx)))/pi; 
            double cyy = 2*((dy*(2*dy*dy-delta*delta*h*h)*sqrt(delta*delta*h*h-dy*dy)+dx*(5*delta*delta*h*h-2*dx*dx)*sqrt(delta*delta*h*h-dx*dx)/3)/pow(delta*h,4)
                +atan(dx/sqrt(delta*delta*h*h-dx*dx))+atan(dy/sqrt(delta*delta*h*h-dx*dx)))/pi; 
            //rhs[i] = (0.5*cxx-cx*dx)*Ffun(x_0[i],y_0[i],s*Dt)-cx*(g_ptr[2*pdim]*normal[0]+g_ptr[2*pdim+1]*normal[1])-cy*(g_ptr[2*pdim+2]*tanvec[0]+g_ptr[2*pdim+3]*tanvec[1]);
            rhs[i] = (0.5*cxx-cx*dx)*Ffun(x_0[i],y_0[i],s*Dt)-cx*g_ptr[pdim]-cy*g_ptr[pdim+1];
            //rhs[i] = (0.5*cxx-cx*dx)*Ffun(x_0[i],y_0[i],s*Dt)-cx*Gfun(x_proj1,y_proj1,normal[0],normal[1],s*Dt)-cy*Gfun(x_proj2,y_proj2,tanvec[0],tanvec[1],s*Dt);
            rhs[i] += (0.5*cxx-cx*dx)*u0[i]/Dt;
          }
          pdim = pdim+2;

        }
        else if(abs(x_0[i]-1) <= delta*h+Tol && abs(y_0[i]-0.5)<=delta*h+Tol){
          normal[0] = 1;
          normal[1] = 0;
          tanvec[0] = 0;
          tanvec[1] = 1;
          double dx = 1 - x_0[i];
          double dy = 0.5 - y_0[i];
          double x_proj1 = 1;
          double y_proj1 = y_0[i];
          double x_proj2 = x_0[i];
          double y_proj2 = 0.5;

          if(ifinhorizon(x_0[i],y_0[i],1,0.5)){
            double cx = 8*(-pow(dy,3)/3-pow(sqrt(delta*delta*h*h-dx*dx),3)/3+dy*(dx*dx+dy*dy-delta*delta*h*h)/2)/pi/pow(delta*h,4);
            double cy = 8*(-pow(dx,3)/3-pow(sqrt(delta*delta*h*h-dy*dy),3)/3+dx*(dx*dx+dy*dy-delta*delta*h*h)/2)/pi/pow(delta*h,4);
            double cxx = ((dx*(2*dx*dx-delta*delta*h*h)*sqrt(delta*delta*h*h-dx*dx)-sqrt(delta*delta*h*h-dy*dy)*(delta*delta*h*h-2*dy*dy)*dy)/pow(delta*h,4)
                +atan(dx/sqrt(delta*delta*h*h-dx*dx))+atan(-sqrt(delta*delta*h*h-dy*dy)/dy)+pi)/pi
                +8*dy*(pow(dx,3)+pow(sqrt(delta*delta*h*h-dy*dy),3))/3/pi/pow(delta*h,4);
            double cyy = ((dy*(2*dy*dy-delta*delta*h*h)*sqrt(delta*delta*h*h-dy*dy)-sqrt(delta*delta*h*h-dx*dx)*(delta*delta*h*h-2*dx*dx)*dx)/pow(delta*h,4)
                +atan(dy/sqrt(delta*delta*h*h-dy*dy))+atan(-sqrt(delta*delta*h*h-dx*dx)/dx)+pi)/pi
                +8*dx*(pow(dy,3)+pow(sqrt(delta*delta*h*h-dx*dx),3))/3/pi/pow(delta*h,4);
            double cxy = pow(dx*dx+dy*dy-delta*delta*h*h,2)/pi/pow(delta*h,4);
            //rhs[i] = (0.5*cxx-cx*dx)*Ffun(x_0[i],y_0[i],s*Dt)-cx*(g_ptr[2*pdim]*normal[0]+g_ptr[2*pdim+1]*normal[1])-cy*(g_ptr[2*pdim+2]*tanvec[0]+g_ptr[2*pdim+3]*tanvec[1])
            rhs[i] = (0.5*cxx-cx*dx)*Ffun(x_0[i],y_0[i],s*Dt)-cx*g_ptr[pdim]-cy*g_ptr[pdim+1]
            //rhs[i] = (0.5*cxx-cx*dx)*Ffun(x_0[i],y_0[i],s*Dt)-cx*Gfun(x_proj1,y_proj1,normal[0],normal[1],s*Dt)-cy*Gfun(x_proj2,y_proj2,tanvec[0],tanvec[1],s*Dt)
              -cxy*0.5*(dGfuny(x_proj1,y_proj1,normal[0],normal[1],s*Dt)+dGfunx(x_proj2,y_proj2,tanvec[0],tanvec[1],s*Dt));
            rhs[i] += (0.5*cxx-cx*dx)*u0[i]/Dt;
          }
          else{
            double cx = -16*pow(sqrt(delta*delta*h*h-dx*dx),3)/3/pi/pow(delta*h,4);
            double cy = -16*pow(sqrt(delta*delta*h*h-dy*dy),3)/3/pi/pow(delta*h,4);
            double cxx = 2*((dx*(2*dx*dx-delta*delta*h*h)*sqrt(delta*delta*h*h-dx*dx)+dy*(5*delta*delta*h*h-2*dy*dy)*sqrt(delta*delta*h*h-dy*dy)/3)/pow(delta*h,4)
                +atan(dx/sqrt(delta*delta*h*h-dx*dx))+atan(dy/sqrt(delta*delta*h*h-dx*dx)))/pi; 
            double cyy = 2*((dy*(2*dy*dy-delta*delta*h*h)*sqrt(delta*delta*h*h-dy*dy)+dx*(5*delta*delta*h*h-2*dx*dx)*sqrt(delta*delta*h*h-dx*dx)/3)/pow(delta*h,4)
                +atan(dx/sqrt(delta*delta*h*h-dx*dx))+atan(dy/sqrt(delta*delta*h*h-dx*dx)))/pi; 
            //rhs[i] = (0.5*cxx-cx*dx)*Ffun(x_0[i],y_0[i],s*Dt)-cx*(g_ptr[2*pdim]*normal[0]+g_ptr[2*pdim+1]*normal[1])-cy*(g_ptr[2*pdim+2]*tanvec[0]+g_ptr[2*pdim+3]*tanvec[1]);
            rhs[i] = (0.5*cxx-cx*dx)*Ffun(x_0[i],y_0[i],s*Dt)-cx*g_ptr[pdim]-cy*g_ptr[pdim+1];
            //rhs[i] = (0.5*cxx-cx*dx)*Ffun(x_0[i],y_0[i],s*Dt)-cx*Gfun(x_proj1,y_proj1,normal[0],normal[1],s*Dt)-cy*Gfun(x_proj2,y_proj2,tanvec[0],tanvec[1],s*Dt);
            rhs[i] += (0.5*cxx-cx*dx)*u0[i]/Dt;
          }
          pdim=pdim+2;
        }
        else if(abs(x_0[i]-1) <= delta*h+Tol && abs(y_0[i]+0.5)<=delta*h+Tol){
          normal[0] = 1;
          normal[1] = 0;
          tanvec[0] = 0;
          tanvec[1] = -1;
          double dx = 1 - x_0[i];
          double dy = 0.5 + y_0[i];
          double x_proj1 = 1;
          double y_proj1 = y_0[i];
          double x_proj2 = x_0[i];
          double y_proj2 = -0.5;

          if(ifinhorizon(x_0[i],y_0[i],1,-0.5)){
            double cx = 8*(-pow(dy,3)/3-pow(sqrt(delta*delta*h*h-dx*dx),3)/3+dy*(dx*dx+dy*dy-delta*delta*h*h)/2)/pi/pow(delta*h,4);
            double cy = 8*(-pow(dx,3)/3-pow(sqrt(delta*delta*h*h-dy*dy),3)/3+dx*(dx*dx+dy*dy-delta*delta*h*h)/2)/pi/pow(delta*h,4);
            double cxx = ((dx*(2*dx*dx-delta*delta*h*h)*sqrt(delta*delta*h*h-dx*dx)-sqrt(delta*delta*h*h-dy*dy)*(delta*delta*h*h-2*dy*dy)*dy)/pow(delta*h,4)
                +atan(dx/sqrt(delta*delta*h*h-dx*dx))+atan(-sqrt(delta*delta*h*h-dy*dy)/dy)+pi)/pi
                +8*dy*(pow(dx,3)+pow(sqrt(delta*delta*h*h-dy*dy),3))/3/pi/pow(delta*h,4);
            double cyy = ((dy*(2*dy*dy-delta*delta*h*h)*sqrt(delta*delta*h*h-dy*dy)-sqrt(delta*delta*h*h-dx*dx)*(delta*delta*h*h-2*dx*dx)*dx)/pow(delta*h,4)
                +atan(dy/sqrt(delta*delta*h*h-dy*dy))+atan(-sqrt(delta*delta*h*h-dx*dx)/dx)+pi)/pi
                +8*dx*(pow(dy,3)+pow(sqrt(delta*delta*h*h-dx*dx),3))/3/pi/pow(delta*h,4);
            double cxy = pow(dx*dx+dy*dy-delta*delta*h*h,2)/pi/pow(delta*h,4);
            //rhs[i] = (0.5*cxx-cx*dx)*Ffun(x_0[i],y_0[i],s*Dt)-cx*(g_ptr[2*pdim]*normal[0]+g_ptr[2*pdim+1]*normal[1])-cy*(g_ptr[2*pdim+2]*tanvec[0]+g_ptr[2*pdim+3]*tanvec[1])
            rhs[i] = (0.5*cxx-cx*dx)*Ffun(x_0[i],y_0[i],s*Dt)-cx*g_ptr[pdim]-cy*g_ptr[pdim+1]
            //rhs[i] = (0.5*cxx-cx*dx)*Ffun(x_0[i],y_0[i],s*Dt)-cx*Gfun(x_proj1,y_proj1,normal[0],normal[1],s*Dt)-cy*Gfun(x_proj2,y_proj2,tanvec[0],tanvec[1],s*Dt)
              -cxy*0.5*(-dGfuny(x_proj1,y_proj1,normal[0],normal[1],s*Dt)+dGfunx(x_proj2,y_proj2,tanvec[0],tanvec[1],s*Dt));
            rhs[i] += (0.5*cxx-cx*dx)*u0[i]/Dt;
          }
          else{
            double cx = -16*pow(sqrt(delta*delta*h*h-dx*dx),3)/3/pi/pow(delta*h,4);
            double cy = -16*pow(sqrt(delta*delta*h*h-dy*dy),3)/3/pi/pow(delta*h,4);
            double cxx = 2*((dx*(2*dx*dx-delta*delta*h*h)*sqrt(delta*delta*h*h-dx*dx)+dy*(5*delta*delta*h*h-2*dy*dy)*sqrt(delta*delta*h*h-dy*dy)/3)/pow(delta*h,4)
                +atan(dx/sqrt(delta*delta*h*h-dx*dx))+atan(dy/sqrt(delta*delta*h*h-dx*dx)))/pi; 
            double cyy = 2*((dy*(2*dy*dy-delta*delta*h*h)*sqrt(delta*delta*h*h-dy*dy)+dx*(5*delta*delta*h*h-2*dx*dx)*sqrt(delta*delta*h*h-dx*dx)/3)/pow(delta*h,4)
                +atan(dx/sqrt(delta*delta*h*h-dx*dx))+atan(dy/sqrt(delta*delta*h*h-dx*dx)))/pi; 
            //rhs[i] = (0.5*cxx-cx*dx)*Ffun(x_0[i],y_0[i],s*Dt)-cx*(g_ptr[2*pdim]*normal[0]+g_ptr[2*pdim+1]*normal[1])-cy*(g_ptr[2*pdim+2]*tanvec[0]+g_ptr[2*pdim+3]*tanvec[1]);
            rhs[i] = (0.5*cxx-cx*dx)*Ffun(x_0[i],y_0[i],s*Dt)-cx*g_ptr[pdim]-cy*g_ptr[pdim+1];
            //rhs[i] = (0.5*cxx-cx*dx)*Ffun(x_0[i],y_0[i],s*Dt)-cx*Gfun(x_proj1,y_proj1,normal[0],normal[1],s*Dt)-cy*Gfun(x_proj2,y_proj2,tanvec[0],tanvec[1],s*Dt);
            rhs[i] += (0.5*cxx-cx*dx)*u0[i]/Dt;
          }
          pdim = pdim+2;
        }
        else if(abs(x_0[i]+0.5) <= delta*h+Tol && abs(y_0[i]+1)<=delta*h+Tol){
          normal[0] = -1;
          normal[1] = 0;
          tanvec[0] = 0;
          tanvec[1] = -1;
          double dx = 0.5 + x_0[i];
          double dy = 1 + y_0[i];
          double x_proj1 = -0.5;
          double y_proj1 = y_0[i];
          double x_proj2 = x_0[i];
          double y_proj2 = -1;

          if(ifinhorizon(x_0[i],y_0[i],-0.5,-1)){
            double cx = 8*(-pow(dy,3)/3-pow(sqrt(delta*delta*h*h-dx*dx),3)/3+dy*(dx*dx+dy*dy-delta*delta*h*h)/2)/pi/pow(delta*h,4);
            double cy = 8*(-pow(dx,3)/3-pow(sqrt(delta*delta*h*h-dy*dy),3)/3+dx*(dx*dx+dy*dy-delta*delta*h*h)/2)/pi/pow(delta*h,4);
            double cxx = ((dx*(2*dx*dx-delta*delta*h*h)*sqrt(delta*delta*h*h-dx*dx)-sqrt(delta*delta*h*h-dy*dy)*(delta*delta*h*h-2*dy*dy)*dy)/pow(delta*h,4)
                +atan(dx/sqrt(delta*delta*h*h-dx*dx))+atan(-sqrt(delta*delta*h*h-dy*dy)/dy)+pi)/pi
                +8*dy*(pow(dx,3)+pow(sqrt(delta*delta*h*h-dy*dy),3))/3/pi/pow(delta*h,4);
            double cyy = ((dy*(2*dy*dy-delta*delta*h*h)*sqrt(delta*delta*h*h-dy*dy)-sqrt(delta*delta*h*h-dx*dx)*(delta*delta*h*h-2*dx*dx)*dx)/pow(delta*h,4)
                +atan(dy/sqrt(delta*delta*h*h-dy*dy))+atan(-sqrt(delta*delta*h*h-dx*dx)/dx)+pi)/pi
                +8*dx*(pow(dy,3)+pow(sqrt(delta*delta*h*h-dx*dx),3))/3/pi/pow(delta*h,4);
            double cxy = pow(dx*dx+dy*dy-delta*delta*h*h,2)/pi/pow(delta*h,4);
            //rhs[i] = (0.5*cxx-cx*dx)*Ffun(x_0[i],y_0[i],s*Dt)-cx*(g_ptr[2*pdim]*normal[0]+g_ptr[2*pdim+1]*normal[1])-cy*(g_ptr[2*pdim+2]*tanvec[0]+g_ptr[2*pdim+3]*tanvec[1])
            rhs[i] = (0.5*cxx-cx*dx)*Ffun(x_0[i],y_0[i],s*Dt)-cx*g_ptr[pdim]-cy*g_ptr[pdim+1]
            //rhs[i] = (0.5*cxx-cx*dx)*Ffun(x_0[i],y_0[i],s*Dt)-cx*Gfun(x_proj1,y_proj1,normal[0],normal[1],s*Dt)-cy*Gfun(x_proj2,y_proj2,tanvec[0],tanvec[1],s*Dt)
              -cxy*0.5*(-dGfuny(x_proj1,y_proj1,normal[0],normal[1],s*Dt)-dGfunx(x_proj2,y_proj2,tanvec[0],tanvec[1],s*Dt));
            rhs[i] += (0.5*cxx-cx*dx)*u0[i]/Dt;
          }
          else{
            double cx = -16*pow(sqrt(delta*delta*h*h-dx*dx),3)/3/pi/pow(delta*h,4);
            double cy = -16*pow(sqrt(delta*delta*h*h-dy*dy),3)/3/pi/pow(delta*h,4);
            double cxx = 2*((dx*(2*dx*dx-delta*delta*h*h)*sqrt(delta*delta*h*h-dx*dx)+dy*(5*delta*delta*h*h-2*dy*dy)*sqrt(delta*delta*h*h-dy*dy)/3)/pow(delta*h,4)
                +atan(dx/sqrt(delta*delta*h*h-dx*dx))+atan(dy/sqrt(delta*delta*h*h-dx*dx)))/pi; 
            double cyy = 2*((dy*(2*dy*dy-delta*delta*h*h)*sqrt(delta*delta*h*h-dy*dy)+dx*(5*delta*delta*h*h-2*dx*dx)*sqrt(delta*delta*h*h-dx*dx)/3)/pow(delta*h,4)
                +atan(dx/sqrt(delta*delta*h*h-dx*dx))+atan(dy/sqrt(delta*delta*h*h-dx*dx)))/pi; 
            //rhs[i] = (0.5*cxx-cx*dx)*Ffun(x_0[i],y_0[i],s*Dt)-cx*(g_ptr[2*pdim]*normal[0]+g_ptr[2*pdim+1]*normal[1])-cy*(g_ptr[2*pdim+2]*tanvec[0]+g_ptr[2*pdim+3]*tanvec[1]);
            rhs[i] = (0.5*cxx-cx*dx)*Ffun(x_0[i],y_0[i],s*Dt)-cx*g_ptr[pdim]-cy*g_ptr[pdim+1];
            //rhs[i] = (0.5*cxx-cx*dx)*Ffun(x_0[i],y_0[i],s*Dt)-cx*Gfun(x_proj1,y_proj1,normal[0],normal[1],s*Dt)-cy*Gfun(x_proj2,y_proj2,tanvec[0],tanvec[1],s*Dt);
            rhs[i] += (0.5*cxx-cx*dx)*u0[i]/Dt;
          }
          pdim = pdim+2;
        }
        else if(abs(x_0[i]-0.5) <= delta*h+Tol && abs(y_0[i]+1)<=delta*h+Tol){
          normal[0] = 1;
          normal[1] = 0;
          tanvec[0] = 0;
          tanvec[1] = -1;
          double dx = 0.5 - x_0[i];
          double dy = 1 + y_0[i];
          double x_proj1 = 0.5;
          double y_proj1 = y_0[i];
          double x_proj2 = x_0[i];
          double y_proj2 = -1;

          if(ifinhorizon(x_0[i],y_0[i],0.5,-1)){
            double cx = 8*(-pow(dy,3)/3-pow(sqrt(delta*delta*h*h-dx*dx),3)/3+dy*(dx*dx+dy*dy-delta*delta*h*h)/2)/pi/pow(delta*h,4);
            double cy = 8*(-pow(dx,3)/3-pow(sqrt(delta*delta*h*h-dy*dy),3)/3+dx*(dx*dx+dy*dy-delta*delta*h*h)/2)/pi/pow(delta*h,4);
            double cxx = ((dx*(2*dx*dx-delta*delta*h*h)*sqrt(delta*delta*h*h-dx*dx)-sqrt(delta*delta*h*h-dy*dy)*(delta*delta*h*h-2*dy*dy)*dy)/pow(delta*h,4)
                +atan(dx/sqrt(delta*delta*h*h-dx*dx))+atan(-sqrt(delta*delta*h*h-dy*dy)/dy)+pi)/pi
                +8*dy*(pow(dx,3)+pow(sqrt(delta*delta*h*h-dy*dy),3))/3/pi/pow(delta*h,4);
            double cyy = ((dy*(2*dy*dy-delta*delta*h*h)*sqrt(delta*delta*h*h-dy*dy)-sqrt(delta*delta*h*h-dx*dx)*(delta*delta*h*h-2*dx*dx)*dx)/pow(delta*h,4)
                +atan(dy/sqrt(delta*delta*h*h-dy*dy))+atan(-sqrt(delta*delta*h*h-dx*dx)/dx)+pi)/pi
                +8*dx*(pow(dy,3)+pow(sqrt(delta*delta*h*h-dx*dx),3))/3/pi/pow(delta*h,4);
            double cxy = pow(dx*dx+dy*dy-delta*delta*h*h,2)/pi/pow(delta*h,4);
            //rhs[i] = (0.5*cxx-cx*dx)*Ffun(x_0[i],y_0[i],s*Dt)-cx*(g_ptr[2*pdim]*normal[0]+g_ptr[2*pdim+1]*normal[1])-cy*(g_ptr[2*pdim+2]*tanvec[0]+g_ptr[2*pdim+3]*tanvec[1])
            rhs[i] = (0.5*cxx-cx*dx)*Ffun(x_0[i],y_0[i],s*Dt)-cx*g_ptr[pdim]-cy*g_ptr[pdim+1]
            //rhs[i] = (0.5*cxx-cx*dx)*Ffun(x_0[i],y_0[i],s*Dt)-cx*Gfun(x_proj1,y_proj1,normal[0],normal[1],s*Dt)-cy*Gfun(x_proj2,y_proj2,tanvec[0],tanvec[1],s*Dt)
              -cxy*0.5*(-dGfuny(x_proj1,y_proj1,normal[0],normal[1],s*Dt)+dGfunx(x_proj2,y_proj2,tanvec[0],tanvec[1],s*Dt));
            rhs[i] += (0.5*cxx-cx*dx)*u0[i]/Dt;
          }
          else{
            double cx = -16*pow(sqrt(delta*delta*h*h-dx*dx),3)/3/pi/pow(delta*h,4);
            double cy = -16*pow(sqrt(delta*delta*h*h-dy*dy),3)/3/pi/pow(delta*h,4);
            double cxx = 2*((dx*(2*dx*dx-delta*delta*h*h)*sqrt(delta*delta*h*h-dx*dx)+dy*(5*delta*delta*h*h-2*dy*dy)*sqrt(delta*delta*h*h-dy*dy)/3)/pow(delta*h,4)
                +atan(dx/sqrt(delta*delta*h*h-dx*dx))+atan(dy/sqrt(delta*delta*h*h-dx*dx)))/pi; 
            double cyy = 2*((dy*(2*dy*dy-delta*delta*h*h)*sqrt(delta*delta*h*h-dy*dy)+dx*(5*delta*delta*h*h-2*dx*dx)*sqrt(delta*delta*h*h-dx*dx)/3)/pow(delta*h,4)
                +atan(dx/sqrt(delta*delta*h*h-dx*dx))+atan(dy/sqrt(delta*delta*h*h-dx*dx)))/pi; 
            //rhs[i] = (0.5*cxx-cx*dx)*Ffun(x_0[i],y_0[i],s*Dt)-cx*(g_ptr[2*pdim]*normal[0]+g_ptr[2*pdim+1]*normal[1])-cy*(g_ptr[2*pdim+2]*tanvec[0]+g_ptr[2*pdim+3]*tanvec[1]);
            rhs[i] = (0.5*cxx-cx*dx)*Ffun(x_0[i],y_0[i],s*Dt)-cx*g_ptr[pdim]-cy*g_ptr[pdim+1];
            //rhs[i] = (0.5*cxx-cx*dx)*Ffun(x_0[i],y_0[i],s*Dt)-cx*Gfun(x_proj1,y_proj1,normal[0],normal[1],s*Dt)-cy*Gfun(x_proj2,y_proj2,tanvec[0],tanvec[1],s*Dt);
            rhs[i] += (0.5*cxx-cx*dx)*u0[i]/Dt;
          }
          pdim = pdim+2;
        }
        else if(ifinhorizon(x_0[i],y_0[i],-0.5,0.5)){
            normal[0] = -1;
            normal[1] = 0;
            tanvec[0] = 0;
            tanvec[1] = 1;
          double dx = 0.5 + x_0[i];
          double dy = 0.5 - y_0[i];
          if(dx > 0 && dy > 0){
            double x_proj1 = -0.5;
            double y_proj1 = 0.5;
            double x_proj2 = -0.5;
            double y_proj2 = 0.5;
            double cx = 8*(-1*pow(sqrt(delta*delta*h*h-dx*dx),3)/3+dy*(delta*delta*h*h-dy*dy-dx*dx)/2+pow(dy,3)/3)/pi/pow(delta*h,4);
            double cy = 8*(-1*pow(sqrt(delta*delta*h*h-dy*dy),3)/3+dx*(delta*delta*h*h-dy*dy-dx*dx)/2+pow(dx,3)/3)/pi/pow(delta*h,4);
            double cxx = (dx*(2*dx*dx-delta*delta*h*h)*sqrt(delta*delta*h*h-dx*dx)/pow(delta*h,4)+dy*(2*dy*dy-delta*delta*h*h)*sqrt(delta*delta*h*h-dy*dy)/pow(delta*h,4)
                +atan(dx/sqrt(delta*delta*h*h-dx*dx))+atan(dy/sqrt(delta*delta*h*h-dy*dy)+2*pi))/pi + 8*dy*(pow(sqrt(delta*delta*h*h-dy*dy),3)-pow(dx,3))/3/pi/pow(delta*h,4);  
            double cyy = (dx*(2*dx*dx-delta*delta*h*h)*sqrt(delta*delta*h*h-dx*dx)/pow(delta*h,4)+dy*(2*dy*dy-delta*delta*h*h)*sqrt(delta*delta*h*h-dy*dy)/pow(delta*h,4)
                +atan(dx/sqrt(delta*delta*h*h-dx*dx))+atan(dy/sqrt(delta*delta*h*h-dy*dy)+2*pi))/pi + 8*dx*(pow(sqrt(delta*delta*h*h-dx*dx),3)-pow(dy,3))/3/pi/pow(delta*h,4);  
            double cxy = -1*pow(delta*delta*h*h-dx*dx-dy*dy,2)/pi/pow(delta*h,4);
            //rhs[i] = (0.5*cxx-cx*dx)*Ffun(x_0[i],y_0[i],s*Dt)-cx*(g_ptr[2*pdim]*normal[0]+g_ptr[2*pdim+1]*normal[1])-cy*(g_ptr[2*pdim+2]*tanvec[0]+g_ptr[2*pdim+3]*tanvec[1])
            rhs[i] = (0.5*cxx-cx*dx)*Ffun(x_0[i],y_0[i],s*Dt)-cx*g_ptr[pdim]-cy*g_ptr[pdim+1]
            //rhs[i] = (0.5*cxx-dx*cx)*Ffun(x_0[i],y_0[i],s*Dt) - cx*Gfun(x_proj1,y_proj1,normal[0],normal[1],s*Dt)-cy*Gfun(x_proj2,y_proj2,tanvec[0],tanvec[1],s*Dt)
            +(cy*dx+cx*dy-cxy)*0.5*(dGfuny(x_proj1,y_proj1,normal[0],normal[1],s*Dt)-dGfunx(x_proj2,y_proj2,tanvec[0],tanvec[1],s*Dt));
            rhs[i] += (0.5*cxx-cx*dx)*u0[i]/Dt;
          }
          if(dx >= 0-Tol && dy <= 0+Tol){
            double x_proj1 = -0.5;
            double y_proj1 = y_0[i];
            double x_proj2 = -0.5;
            double y_proj2 = 0.5;
            double cx = 8*(-1*pow(sqrt(delta*delta*h*h-dx*dx),3)/3+(delta*delta*h*h-dx*dx)*dy/2)/pi/pow(delta*h,4);
            double cy = 8*((dy*dy-delta*delta*h*h)*(delta*h-dx)/2+pow(delta*h,3)/6-pow(dx,3)/6)/pi/pow(delta*h,4);
            double cxx = (dx*(2*dx*dx-delta*delta*h*h)*sqrt(delta*delta*h*h-dx*dx)/pow(delta*h,4)+atan(dx/sqrt(delta*delta*h*h-dx*dx)+3*pi/2))/pi
              +8*dy*(pow(delta*delta*h*h,3)-pow(dx,3))/3/pi/pow(delta*h,4);
            double cyy = (dx*(5*delta*delta*h*h-2*dx*dx)*sqrt(delta*delta*h*h-dx*dx)/3/pow(delta*h,4)+atan(dx/sqrt(delta*delta*h*h-dx*dx))+3*pi/2)/pi
              +8*pow(dy,3)*(delta*h-dx)/3/pi/pow(delta*h,4);
            double cxy = (2*(dy*dy-delta*delta*h*h)*(delta*delta*h*h-dx*dx)+pow(delta*h,4)-pow(dx,4))/pi/pow(delta*h,4);
            //rhs[i] = (0.5*cxx-cx*dx)*Ffun(x_0[i],y_0[i],s*Dt)-cx*(g_ptr[2*pdim]*normal[0]+g_ptr[2*pdim+1]*normal[1])-cy*(g_ptr[2*pdim+2]*tanvec[0]+g_ptr[2*pdim+3]*tanvec[1])
            rhs[i] = (0.5*cxx-cx*dx)*Ffun(x_0[i],y_0[i],s*Dt)-cx*g_ptr[pdim]-cy*g_ptr[pdim+1]
            //rhs[i] = (0.5*cxx-dx*cx)*Ffun(x_0[i],y_0[i],s*Dt) - cx*Gfun(x_proj1,y_proj1,normal[0],normal[1],s*Dt)-cy*Gfun(x_proj2,y_proj2,tanvec[0],tanvec[1],s*Dt)
            +(cy*dx-cxy)*0.5*(dGfuny(x_proj1,y_proj1,normal[0],normal[1],s*Dt)-dGfunx(x_proj2,y_proj2,tanvec[0],tanvec[1],s*Dt));
            rhs[i] += (0.5*cxx-cx*dx)*u0[i]/Dt;
          }
          if(dx <= 0+Tol && dy >=0-Tol && !(fabs(dx)<= 0+Tol && fabs(dy) <=0+Tol)){
            double x_proj1 = -0.5;
            double y_proj1 = 0.5;
            double x_proj2 = x_0[i];
            double y_proj2 = 0.5;
            double cx = 8*((dx*dx-delta*delta*h*h)*(delta*h-dy)/2+pow(delta*h,3)/6-pow(dy,3)/6)/pi/pow(delta*h,4); 
            double cy = 8*(-1*pow(sqrt(delta*delta*h*h-dy*dy),3)/3+(delta*delta*h*h-dy*dy)*dx/2)/pi/pow(delta*h,4);
            double cxx = (dy*(5*delta*delta*h*h-2*dy*dy)*sqrt(delta*delta*h*h-dy*dy)/3/pow(delta*h,4)+atan(dy/sqrt(delta*delta*h*h-dy*dy))+3*pi/2)/pi
              +8*pow(dx,3)*(delta*h-dy)/3/pi/pow(delta*h,4);
            double cyy = (dy*(2*dy*dy-delta*delta*h*h)*sqrt(delta*delta*h*h-dy*dy)/pow(delta*h,4)+atan(dy/sqrt(delta*delta*h*h-dy*dy)+3*pi/2))/pi
              +8*dx*(pow(delta*delta*h*h,3)-pow(dy,3))/3/pi/pow(delta*h,4);
            double cxy = (2*(dx*dx-delta*delta*h*h)*(delta*delta*h*h-dy*dy)+pow(delta*h,4)-pow(dy,4))/pi/pow(delta*h,4);
            //rhs[i] = (0.5*cxx-cx*dx)*Ffun(x_0[i],y_0[i],s*Dt)-cx*(g_ptr[2*pdim]*normal[0]+g_ptr[2*pdim+1]*normal[1])-cy*(g_ptr[2*pdim+2]*tanvec[0]+g_ptr[2*pdim+3]*tanvec[1])
            rhs[i] = (0.5*cxx-cx*dx)*Ffun(x_0[i],y_0[i],s*Dt)-cx*g_ptr[pdim]-cy*g_ptr[pdim+1]
            //rhs[i] = (0.5*cxx-dx*cx)*Ffun(x_0[i],y_0[i],s*Dt) - cx*Gfun(x_proj1,y_proj1,normal[0],normal[1],s*Dt)-cy*Gfun(x_proj2,y_proj2,tanvec[0],tanvec[1],s*Dt)
            +(cx*dy-cxy)*0.5*(dGfuny(x_proj1,y_proj1,normal[0],normal[1],s*Dt)-dGfunx(x_proj2,y_proj2,tanvec[0],tanvec[1],s*Dt));
            rhs[i] += (0.5*cxx-cx*dx)*u0[i]/Dt;
          }
          pdim = pdim+2;
        }
        else if(ifinhorizon(x_0[i],y_0[i],0.5,0.5)){
            normal[0] = 1;
            normal[1] = 0;
            tanvec[0] = 0;
            tanvec[1] = 1;
          double dx = 0.5 - x_0[i];
          double dy = 0.5 - y_0[i];
          if(dx > 0 && dy > 0){
            double x_proj1 = 0.5;
            double y_proj1 = 0.5;
            double x_proj2 = 0.5;
            double y_proj2 = 0.5;
            double cx = 8*(-1*pow(sqrt(delta*delta*h*h-dx*dx),3)/3+dy*(delta*delta*h*h-dy*dy-dx*dx)/2+pow(dy,3)/3)/pi/pow(delta*h,4);
            double cy = 8*(-1*pow(sqrt(delta*delta*h*h-dy*dy),3)/3+dx*(delta*delta*h*h-dy*dy-dx*dx)/2+pow(dx,3)/3)/pi/pow(delta*h,4);
            double cxx = (dx*(2*dx*dx-delta*delta*h*h)*sqrt(delta*delta*h*h-dx*dx)/pow(delta*h,4)+dy*(2*dy*dy-delta*delta*h*h)*sqrt(delta*delta*h*h-dy*dy)/pow(delta*h,4)
                +atan(dx/sqrt(delta*delta*h*h-dx*dx))+atan(dy/sqrt(delta*delta*h*h-dy*dy)+2*pi))/pi + 8*dy*(pow(sqrt(delta*delta*h*h-dy*dy),3)-pow(dx,3))/3/pi/pow(delta*h,4);  
            double cyy = (dx*(2*dx*dx-delta*delta*h*h)*sqrt(delta*delta*h*h-dx*dx)/pow(delta*h,4)+dy*(2*dy*dy-delta*delta*h*h)*sqrt(delta*delta*h*h-dy*dy)/pow(delta*h,4)
                +atan(dx/sqrt(delta*delta*h*h-dx*dx))+atan(dy/sqrt(delta*delta*h*h-dy*dy)+2*pi))/pi + 8*dx*(pow(sqrt(delta*delta*h*h-dx*dx),3)-pow(dy,3))/3/pi/pow(delta*h,4);  
            double cxy = -1*pow(delta*delta*h*h-dx*dx-dy*dy,2)/pi/pow(delta*h,4);
            //rhs[i] = (0.5*cxx-cx*dx)*Ffun(x_0[i],y_0[i],s*Dt)-cx*(g_ptr[2*pdim]*normal[0]+g_ptr[2*pdim+1]*normal[1])-cy*(g_ptr[2*pdim+2]*tanvec[0]+g_ptr[2*pdim+3]*tanvec[1])
            rhs[i] = (0.5*cxx-cx*dx)*Ffun(x_0[i],y_0[i],s*Dt)-cx*g_ptr[pdim]-cy*g_ptr[pdim+1]
            //rhs[i] = (0.5*cxx-dx*cx)*Ffun(x_0[i],y_0[i],s*Dt) - cx*Gfun(x_proj1,y_proj1,normal[0],normal[1],s*Dt)-cy*Gfun(x_proj2,y_proj2,tanvec[0],tanvec[1],s*Dt)
            +(cy*dx+cx*dy-cxy)*0.5*(dGfuny(x_proj1,y_proj1,normal[0],normal[1],s*Dt)+dGfunx(x_proj2,y_proj2,tanvec[0],tanvec[1],s*Dt));
            rhs[i] += (0.5*cxx-cx*dx)*u0[i]/Dt;
          }
          if(dx >= 0-Tol && dy <= 0+Tol){
            double x_proj1 = 0.5;
            double y_proj1 = y_0[i];
            double x_proj2 = 0.5;
            double y_proj2 = 0.5;
            double cx = 8*(-1*pow(sqrt(delta*delta*h*h-dx*dx),3)/3+(delta*delta*h*h-dx*dx)*dy/2)/pi/pow(delta*h,4);
            double cy = 8*((dy*dy-delta*delta*h*h)*(delta*h-dx)/2+pow(delta*h,3)/6-pow(dx,3)/6)/pi/pow(delta*h,4);
            double cxx = (dx*(2*dx*dx-delta*delta*h*h)*sqrt(delta*delta*h*h-dx*dx)/pow(delta*h,4)+atan(dx/sqrt(delta*delta*h*h-dx*dx)+3*pi/2))/pi
              +8*dy*(pow(delta*delta*h*h,3)-pow(dx,3))/3/pi/pow(delta*h,4);
            double cyy = (dx*(5*delta*delta*h*h-2*dx*dx)*sqrt(delta*delta*h*h-dx*dx)/3/pow(delta*h,4)+atan(dx/sqrt(delta*delta*h*h-dx*dx))+3*pi/2)/pi
              +8*pow(dy,3)*(delta*h-dx)/3/pi/pow(delta*h,4);
            double cxy = (2*(dy*dy-delta*delta*h*h)*(delta*delta*h*h-dx*dx)+pow(delta*h,4)-pow(dx,4))/pi/pow(delta*h,4);
            //rhs[i] = (0.5*cxx-cx*dx)*Ffun(x_0[i],y_0[i],s*Dt)-cx*(g_ptr[2*pdim]*normal[0]+g_ptr[2*pdim+1]*normal[1])-cy*(g_ptr[2*pdim+2]*tanvec[0]+g_ptr[2*pdim+3]*tanvec[1])
            rhs[i] = (0.5*cxx-cx*dx)*Ffun(x_0[i],y_0[i],s*Dt)-cx*g_ptr[pdim]-cy*g_ptr[pdim+1]
            //rhs[i] = (0.5*cxx-dx*cx)*Ffun(x_0[i],y_0[i],s*Dt) - cx*Gfun(x_proj1,y_proj1,normal[0],normal[1],s*Dt)-cy*Gfun(x_proj2,y_proj2,tanvec[0],tanvec[1],s*Dt)
            +(cy*dx-cxy)*0.5*(dGfuny(x_proj1,y_proj1,normal[0],normal[1],s*Dt)+dGfunx(x_proj2,y_proj2,tanvec[0],tanvec[1],s*Dt));
            rhs[i] += (0.5*cxx-cx*dx)*u0[i]/Dt;
          }
          if(dx <= 0+Tol && dy >=0-Tol && !(fabs(dx)<= 0+Tol && fabs(dy) <=0+Tol)){
            double x_proj1 = 0.5;
            double y_proj1 = 0.5;
            double x_proj2 = x_0[i];
            double y_proj2 = 0.5;
            double cx = 8*((dx*dx-delta*delta*h*h)*(delta*h-dy)/2+pow(delta*h,3)/6-pow(dy,3)/6)/pi/pow(delta*h,4); 
            double cy = 8*(-1*pow(sqrt(delta*delta*h*h-dy*dy),3)/3+(delta*delta*h*h-dy*dy)*dx/2)/pi/pow(delta*h,4);
            double cxx = (dy*(5*delta*delta*h*h-2*dy*dy)*sqrt(delta*delta*h*h-dy*dy)/3/pow(delta*h,4)+atan(dy/sqrt(delta*delta*h*h-dy*dy))+3*pi/2)/pi
              +8*pow(dx,3)*(delta*h-dy)/3/pi/pow(delta*h,4);
            double cyy = (dy*(2*dy*dy-delta*delta*h*h)*sqrt(delta*delta*h*h-dy*dy)/pow(delta*h,4)+atan(dy/sqrt(delta*delta*h*h-dy*dy)+3*pi/2))/pi
              +8*dx*(pow(delta*delta*h*h,3)-pow(dy,3))/3/pi/pow(delta*h,4);
            double cxy = (2*(dx*dx-delta*delta*h*h)*(delta*delta*h*h-dy*dy)+pow(delta*h,4)-pow(dy,4))/pi/pow(delta*h,4);
            //rhs[i] = (0.5*cxx-cx*dx)*Ffun(x_0[i],y_0[i],s*Dt)-cx*(g_ptr[2*pdim]*normal[0]+g_ptr[2*pdim+1]*normal[1])-cy*(g_ptr[2*pdim+2]*tanvec[0]+g_ptr[2*pdim+3]*tanvec[1])
            rhs[i] = (0.5*cxx-cx*dx)*Ffun(x_0[i],y_0[i],s*Dt)-cx*g_ptr[pdim]-cy*g_ptr[pdim+1]
            //rhs[i] = (0.5*cxx-dx*cx)*Ffun(x_0[i],y_0[i],s*Dt) - cx*Gfun(x_proj1,y_proj1,normal[0],normal[1],s*Dt)-cy*Gfun(x_proj2,y_proj2,tanvec[0],tanvec[1],s*Dt)
            +(cx*dy-cxy)*0.5*(dGfuny(x_proj1,y_proj1,normal[0],normal[1],s*Dt)+dGfunx(x_proj2,y_proj2,tanvec[0],tanvec[1],s*Dt));
            rhs[i] += (0.5*cxx-cx*dx)*u0[i]/Dt;
          }
          pdim = pdim+2;
        }
        else if(ifinhorizon(x_0[i],y_0[i],-0.5,-0.5)){
            normal[0] = -1;
            normal[1] = 0;
            tanvec[0] = 0;
            tanvec[1] = -1;
          double dx = 0.5 + x_0[i];
          double dy = 0.5 + y_0[i];
          if(dx > 0 && dy > 0){
            double x_proj1 = -0.5;
            double y_proj1 = -0.5;
            double x_proj2 = -0.5;
            double y_proj2 = -0.5;
            double cx = 8*(-1*pow(sqrt(delta*delta*h*h-dx*dx),3)/3+dy*(delta*delta*h*h-dy*dy-dx*dx)/2+pow(dy,3)/3)/pi/pow(delta*h,4);
            double cy = 8*(-1*pow(sqrt(delta*delta*h*h-dy*dy),3)/3+dx*(delta*delta*h*h-dy*dy-dx*dx)/2+pow(dx,3)/3)/pi/pow(delta*h,4);
            double cxx = (dx*(2*dx*dx-delta*delta*h*h)*sqrt(delta*delta*h*h-dx*dx)/pow(delta*h,4)+dy*(2*dy*dy-delta*delta*h*h)*sqrt(delta*delta*h*h-dy*dy)/pow(delta*h,4)
                +atan(dx/sqrt(delta*delta*h*h-dx*dx))+atan(dy/sqrt(delta*delta*h*h-dy*dy)+2*pi))/pi + 8*dy*(pow(sqrt(delta*delta*h*h-dy*dy),3)-pow(dx,3))/3/pi/pow(delta*h,4);  
            double cyy = (dx*(2*dx*dx-delta*delta*h*h)*sqrt(delta*delta*h*h-dx*dx)/pow(delta*h,4)+dy*(2*dy*dy-delta*delta*h*h)*sqrt(delta*delta*h*h-dy*dy)/pow(delta*h,4)
                +atan(dx/sqrt(delta*delta*h*h-dx*dx))+atan(dy/sqrt(delta*delta*h*h-dy*dy)+2*pi))/pi + 8*dx*(pow(sqrt(delta*delta*h*h-dx*dx),3)-pow(dy,3))/3/pi/pow(delta*h,4);  
            double cxy = -1*pow(delta*delta*h*h-dx*dx-dy*dy,2)/pi/pow(delta*h,4);
            //rhs[i] = (0.5*cxx-cx*dx)*Ffun(x_0[i],y_0[i],s*Dt)-cx*(g_ptr[2*pdim]*normal[0]+g_ptr[2*pdim+1]*normal[1])-cy*(g_ptr[2*pdim+2]*tanvec[0]+g_ptr[2*pdim+3]*tanvec[1])
            rhs[i] = (0.5*cxx-cx*dx)*Ffun(x_0[i],y_0[i],s*Dt)-cx*g_ptr[pdim]-cy*g_ptr[pdim+1]
            //rhs[i] = (0.5*cxx-dx*cx)*Ffun(x_0[i],y_0[i],s*Dt) - cx*Gfun(x_proj1,y_proj1,normal[0],normal[1],s*Dt)-cy*Gfun(x_proj2,y_proj2,tanvec[0],tanvec[1],s*Dt)
            +(cy*dx+cx*dy-cxy)*0.5*(-dGfuny(x_proj1,y_proj1,normal[0],normal[1],s*Dt)-dGfunx(x_proj2,y_proj2,tanvec[0],tanvec[1],s*Dt));
            rhs[i] += (0.5*cxx-cx*dx)*u0[i]/Dt;
          }
          if(dx >= 0-Tol && dy <= 0+Tol){
            double x_proj1 = -0.5;
            double y_proj1 = y_0[i];
            double x_proj2 = -0.5;
            double y_proj2 = -0.5;
            double cx = 8*(-1*pow(sqrt(delta*delta*h*h-dx*dx),3)/3+(delta*delta*h*h-dx*dx)*dy/2)/pi/pow(delta*h,4);
            double cy = 8*((dy*dy-delta*delta*h*h)*(delta*h-dx)/2+pow(delta*h,3)/6-pow(dx,3)/6)/pi/pow(delta*h,4);
            double cxx = (dx*(2*dx*dx-delta*delta*h*h)*sqrt(delta*delta*h*h-dx*dx)/pow(delta*h,4)+atan(dx/sqrt(delta*delta*h*h-dx*dx)+3*pi/2))/pi
              +8*dy*(pow(delta*delta*h*h,3)-pow(dx,3))/3/pi/pow(delta*h,4);
            double cyy = (dx*(5*delta*delta*h*h-2*dx*dx)*sqrt(delta*delta*h*h-dx*dx)/3/pow(delta*h,4)+atan(dx/sqrt(delta*delta*h*h-dx*dx))+3*pi/2)/pi
              +8*pow(dy,3)*(delta*h-dx)/3/pi/pow(delta*h,4);
            double cxy = (2*(dy*dy-delta*delta*h*h)*(delta*delta*h*h-dx*dx)+pow(delta*h,4)-pow(dx,4))/pi/pow(delta*h,4);
            //rhs[i] = (0.5*cxx-cx*dx)*Ffun(x_0[i],y_0[i],s*Dt)-cx*(g_ptr[2*pdim]*normal[0]+g_ptr[2*pdim+1]*normal[1])-cy*(g_ptr[2*pdim+2]*tanvec[0]+g_ptr[2*pdim+3]*tanvec[1])
            rhs[i] = (0.5*cxx-cx*dx)*Ffun(x_0[i],y_0[i],s*Dt)-cx*g_ptr[pdim]-cy*g_ptr[pdim+1]
            //rhs[i] = (0.5*cxx-dx*cx)*Ffun(x_0[i],y_0[i],s*Dt) - cx*Gfun(x_proj1,y_proj1,normal[0],normal[1],s*Dt)-cy*Gfun(x_proj2,y_proj2,tanvec[0],tanvec[1],s*Dt)
            +(cy*dx-cxy)*0.5*(-dGfuny(x_proj1,y_proj1,normal[0],normal[1],s*Dt)-dGfunx(x_proj2,y_proj2,tanvec[0],tanvec[1],s*Dt));
            rhs[i] += (0.5*cxx-cx*dx)*u0[i]/Dt;
          }
          if(dx <= 0+Tol && dy >=0-Tol && !(fabs(dx)<= 0+Tol && fabs(dy) <=0+Tol)){
            double x_proj1 = -0.5;
            double y_proj1 = -0.5;
            double x_proj2 = x_0[i];
            double y_proj2 = -0.5;
            double cx = 8*((dx*dx-delta*delta*h*h)*(delta*h-dy)/2+pow(delta*h,3)/6-pow(dy,3)/6)/pi/pow(delta*h,4); 
            double cy = 8*(-1*pow(sqrt(delta*delta*h*h-dy*dy),3)/3+(delta*delta*h*h-dy*dy)*dx/2)/pi/pow(delta*h,4);
            double cxx = (dy*(5*delta*delta*h*h-2*dy*dy)*sqrt(delta*delta*h*h-dy*dy)/3/pow(delta*h,4)+atan(dy/sqrt(delta*delta*h*h-dy*dy))+3*pi/2)/pi
              +8*pow(dx,3)*(delta*h-dy)/3/pi/pow(delta*h,4);
            double cyy = (dy*(2*dy*dy-delta*delta*h*h)*sqrt(delta*delta*h*h-dy*dy)/pow(delta*h,4)+atan(dy/sqrt(delta*delta*h*h-dy*dy)+3*pi/2))/pi
              +8*dx*(pow(delta*delta*h*h,3)-pow(dy,3))/3/pi/pow(delta*h,4);
            double cxy = (2*(dx*dx-delta*delta*h*h)*(delta*delta*h*h-dy*dy)+pow(delta*h,4)-pow(dy,4))/pi/pow(delta*h,4);
            //rhs[i] = (0.5*cxx-cx*dx)*Ffun(x_0[i],y_0[i],s*Dt)-cx*(g_ptr[2*pdim]*normal[0]+g_ptr[2*pdim+1]*normal[1])-cy*(g_ptr[2*pdim+2]*tanvec[0]+g_ptr[2*pdim+3]*tanvec[1])
            rhs[i] = (0.5*cxx-cx*dx)*Ffun(x_0[i],y_0[i],s*Dt)-cx*g_ptr[pdim]-cy*g_ptr[pdim+1]
            //rhs[i] = (0.5*cxx-dx*cx)*Ffun(x_0[i],y_0[i],s*Dt) - cx*Gfun(x_proj1,y_proj1,normal[0],normal[1],s*Dt)-cy*Gfun(x_proj2,y_proj2,tanvec[0],tanvec[1],s*Dt)
            +(cx*dy-cxy)*0.5*(-dGfuny(x_proj1,y_proj1,normal[0],normal[1],s*Dt)-dGfunx(x_proj2,y_proj2,tanvec[0],tanvec[1],s*Dt));
            rhs[i] += (0.5*cxx-cx*dx)*u0[i]/Dt;
          }
          pdim = pdim+2;
        }
        else if(ifinhorizon(x_0[i],y_0[i],0.5,-0.5)){
            normal[0] = 1;
            normal[1] = 0;
            tanvec[0] = 0;
            tanvec[1] = -1;
          double dx = 0.5 - x_0[i];
          double dy = 0.5 + y_0[i];
          if(dx > 0 && dy > 0){
            double x_proj1 = 0.5;
            double y_proj1 = -0.5;
            double x_proj2 = 0.5;
            double y_proj2 = -0.5;
            double cx = 8*(-1*pow(sqrt(delta*delta*h*h-dx*dx),3)/3+dy*(delta*delta*h*h-dy*dy-dx*dx)/2+pow(dy,3)/3)/pi/pow(delta*h,4);
            double cy = 8*(-1*pow(sqrt(delta*delta*h*h-dy*dy),3)/3+dx*(delta*delta*h*h-dy*dy-dx*dx)/2+pow(dx,3)/3)/pi/pow(delta*h,4);
            double cxx = (dx*(2*dx*dx-delta*delta*h*h)*sqrt(delta*delta*h*h-dx*dx)/pow(delta*h,4)+dy*(2*dy*dy-delta*delta*h*h)*sqrt(delta*delta*h*h-dy*dy)/pow(delta*h,4)
                +atan(dx/sqrt(delta*delta*h*h-dx*dx))+atan(dy/sqrt(delta*delta*h*h-dy*dy)+2*pi))/pi + 8*dy*(pow(sqrt(delta*delta*h*h-dy*dy),3)-pow(dx,3))/3/pi/pow(delta*h,4);  
            double cyy = (dx*(2*dx*dx-delta*delta*h*h)*sqrt(delta*delta*h*h-dx*dx)/pow(delta*h,4)+dy*(2*dy*dy-delta*delta*h*h)*sqrt(delta*delta*h*h-dy*dy)/pow(delta*h,4)
                +atan(dx/sqrt(delta*delta*h*h-dx*dx))+atan(dy/sqrt(delta*delta*h*h-dy*dy)+2*pi))/pi + 8*dx*(pow(sqrt(delta*delta*h*h-dx*dx),3)-pow(dy,3))/3/pi/pow(delta*h,4);  
            double cxy = -1*pow(delta*delta*h*h-dx*dx-dy*dy,2)/pi/pow(delta*h,4);
            //rhs[i] = (0.5*cxx-cx*dx)*Ffun(x_0[i],y_0[i],s*Dt)-cx*(g_ptr[2*pdim]*normal[0]+g_ptr[2*pdim+1]*normal[1])-cy*(g_ptr[2*pdim+2]*tanvec[0]+g_ptr[2*pdim+3]*tanvec[1])
            rhs[i] = (0.5*cxx-cx*dx)*Ffun(x_0[i],y_0[i],s*Dt)-cx*g_ptr[pdim]-cy*g_ptr[pdim+1]
            //rhs[i] = (0.5*cxx-dx*cx)*Ffun(x_0[i],y_0[i],s*Dt) - cx*Gfun(x_proj1,y_proj1,normal[0],normal[1],s*Dt)-cy*Gfun(x_proj2,y_proj2,tanvec[0],tanvec[1],s*Dt)
            +(cy*dx+cx*dy-cxy)*0.5*(-dGfuny(x_proj1,y_proj1,normal[0],normal[1],s*Dt)+dGfunx(x_proj2,y_proj2,tanvec[0],tanvec[1],s*Dt));
            rhs[i] += (0.5*cxx-cx*dx)*u0[i]/Dt;
          }
          if(dx >= 0-Tol && dy <= 0+Tol){
            double x_proj1 = 0.5;
            double y_proj1 = y_0[i];
            double x_proj2 = 0.5;
            double y_proj2 = -0.5;
            double cx = 8*(-1*pow(sqrt(delta*delta*h*h-dx*dx),3)/3+(delta*delta*h*h-dx*dx)*dy/2)/pi/pow(delta*h,4);
            double cy = 8*((dy*dy-delta*delta*h*h)*(delta*h-dx)/2+pow(delta*h,3)/6-pow(dx,3)/6)/pi/pow(delta*h,4);
            double cxx = (dx*(2*dx*dx-delta*delta*h*h)*sqrt(delta*delta*h*h-dx*dx)/pow(delta*h,4)+atan(dx/sqrt(delta*delta*h*h-dx*dx)+3*pi/2))/pi
              +8*dy*(pow(delta*delta*h*h,3)-pow(dx,3))/3/pi/pow(delta*h,4);
            double cyy = (dx*(5*delta*delta*h*h-2*dx*dx)*sqrt(delta*delta*h*h-dx*dx)/3/pow(delta*h,4)+atan(dx/sqrt(delta*delta*h*h-dx*dx))+3*pi/2)/pi
              +8*pow(dy,3)*(delta*h-dx)/3/pi/pow(delta*h,4);
            double cxy = (2*(dy*dy-delta*delta*h*h)*(delta*delta*h*h-dx*dx)+pow(delta*h,4)-pow(dx,4))/pi/pow(delta*h,4);
            //rhs[i] = (0.5*cxx-cx*dx)*Ffun(x_0[i],y_0[i],s*Dt)-cx*(g_ptr[2*pdim]*normal[0]+g_ptr[2*pdim+1]*normal[1])-cy*(g_ptr[2*pdim+2]*tanvec[0]+g_ptr[2*pdim+3]*tanvec[1])
            rhs[i] = (0.5*cxx-cx*dx)*Ffun(x_0[i],y_0[i],s*Dt)-cx*g_ptr[pdim]-cy*g_ptr[pdim+1]
            //rhs[i] = (0.5*cxx-dx*cx)*Ffun(x_0[i],y_0[i],s*Dt) - cx*Gfun(x_proj1,y_proj1,normal[0],normal[1],s*Dt)-cy*Gfun(x_proj2,y_proj2,tanvec[0],tanvec[1],s*Dt)
            +(cy*dx-cxy)*0.5*(-dGfuny(x_proj1,y_proj1,normal[0],normal[1],s*Dt)+dGfunx(x_proj2,y_proj2,tanvec[0],tanvec[1],s*Dt));
            rhs[i] += (0.5*cxx-cx*dx)*u0[i]/Dt;
          }
          if(dx <= 0+Tol && dy >=0-Tol && !(fabs(dx)<= 0+Tol && fabs(dy) <=0+Tol)){
            double x_proj1 = 0.5;
            double y_proj1 = -0.5;
            double x_proj2 = x_0[i];
            double y_proj2 = -0.5;
            double cx = 8*((dx*dx-delta*delta*h*h)*(delta*h-dy)/2+pow(delta*h,3)/6-pow(dy,3)/6)/pi/pow(delta*h,4); 
            double cy = 8*(-1*pow(sqrt(delta*delta*h*h-dy*dy),3)/3+(delta*delta*h*h-dy*dy)*dx/2)/pi/pow(delta*h,4);
            double cxx = (dy*(5*delta*delta*h*h-2*dy*dy)*sqrt(delta*delta*h*h-dy*dy)/3/pow(delta*h,4)+atan(dy/sqrt(delta*delta*h*h-dy*dy))+3*pi/2)/pi
              +8*pow(dx,3)*(delta*h-dy)/3/pi/pow(delta*h,4);
            double cyy = (dy*(2*dy*dy-delta*delta*h*h)*sqrt(delta*delta*h*h-dy*dy)/pow(delta*h,4)+atan(dy/sqrt(delta*delta*h*h-dy*dy)+3*pi/2))/pi
              +8*dx*(pow(delta*delta*h*h,3)-pow(dy,3))/3/pi/pow(delta*h,4);
            double cxy = (2*(dx*dx-delta*delta*h*h)*(delta*delta*h*h-dy*dy)+pow(delta*h,4)-pow(dy,4))/pi/pow(delta*h,4);
            //rhs[i] = (0.5*cxx-cx*dx)*Ffun(x_0[i],y_0[i],s*Dt)-cx*(g_ptr[2*pdim]*normal[0]+g_ptr[2*pdim+1]*normal[1])-cy*(g_ptr[2*pdim+2]*tanvec[0]+g_ptr[2*pdim+3]*tanvec[1])
            rhs[i] = (0.5*cxx-cx*dx)*Ffun(x_0[i],y_0[i],s*Dt)-cx*g_ptr[pdim]-cy*g_ptr[pdim+1]
            //rhs[i] = (0.5*cxx-dx*cx)*Ffun(x_0[i],y_0[i],s*Dt) - cx*Gfun(x_proj1,y_proj1,normal[0],normal[1],s*Dt)-cy*Gfun(x_proj2,y_proj2,tanvec[0],tanvec[1],s*Dt)
            +(cx*dy-cxy)*0.5*(-dGfuny(x_proj1,y_proj1,normal[0],normal[1],s*Dt)+dGfunx(x_proj2,y_proj2,tanvec[0],tanvec[1],s*Dt));
            rhs[i] += (0.5*cxx-cx*dx)*u0[i]/Dt;
          }
          pdim = pdim+2;
        }
        //straight line cases
        else if(abs(x_0[i]+1) <= delta*h+Tol){ 
          normal[0] = -1;
          normal[1] = 0;
          double dist = x_0[i]+1;
          double cx = -16*pow(sqrt(delta*delta*h*h-dist*dist),3)/3/pi/pow(delta*h,4);
          double cxx = (2*dist*(2*dist*dist-delta*h*delta*h)*sqrt(delta*delta*h*h-dist*dist)/pow(delta*h,4)+2*atan(dist/sqrt(delta*delta*h*h-dist*dist))
                +pi)/pi;
          double cyy = (2*dist*(5*delta*delta*h*h-2*dist*dist)*sqrt(delta*delta*h*h-dist*dist)/3/pow(delta*h,4)+2*atan(dist/sqrt(delta*delta*h*h-dist*dist))+pi)/pi;
          double x_proj = -1;
          double y_proj = y_0[i];
          
          //rhs[i] = (0.5*cxx-dist*cx)*Ffun(x_0[i],y_0[i],s*Dt)-cx*(g_ptr[2*pdim]*normal[0]+g_ptr[2*pdim+1]*normal[1]);
          rhs[i] = (0.5*cxx-dist*cx)*Ffun(x_0[i],y_0[i],s*Dt)-cx*g_ptr[pdim];
          //rhs[i] = (0.5*cxx-dist*cx)*Ffun(x_0[i],y_0[i],s*Dt)-cx*Gfun(x_proj,y_proj,normal[0],normal[1],s*Dt);
            rhs[i] += (0.5*cxx-cx*dist)*u0[i]/Dt;
          //rhs[i] = u0[i];
          pdim = pdim+1;
        }
        else if(abs(x_0[i]-1) <= delta*h+Tol){ 
          normal[0] = 1;
          normal[1] = 0;
          double dist = -x_0[i]+1;
          double cx = -16*pow(sqrt(delta*delta*h*h-dist*dist),3)/3/pi/pow(delta*h,4);
          double cxx = (2*dist*(2*dist*dist-delta*h*delta*h)*sqrt(delta*delta*h*h-dist*dist)/pow(delta*h,4)+2*atan(dist/sqrt(delta*delta*h*h-dist*dist))
                +pi)/pi;
          double cyy = (2*dist*(5*delta*delta*h*h-2*dist*dist)*sqrt(delta*delta*h*h-dist*dist)/3/pow(delta*h,4)+2*atan(dist/sqrt(delta*delta*h*h-dist*dist))+pi)/pi;
          double x_proj = 1;
          double y_proj = y_0[i];
          
          //rhs[i] = (0.5*cxx-dist*cx)*Ffun(x_0[i],y_0[i],s*Dt)-cx*(g_ptr[2*pdim]*normal[0]+g_ptr[2*pdim+1]*normal[1]);
          rhs[i] = (0.5*cxx-dist*cx)*Ffun(x_0[i],y_0[i],s*Dt)-cx*g_ptr[pdim];
          //rhs[i] = (0.5*cxx-dist*cx)*Ffun(x_0[i],y_0[i],s*Dt)-cx*Gfun(x_proj,y_proj,normal[0],normal[1],s*Dt);
            rhs[i] += (0.5*cxx-cx*dist)*u0[i]/Dt;
          pdim = pdim+1;
          //rhs[i] = u0[i];
        }
        else if(abs(y_0[i]+1) <= delta*h+Tol){ 
          normal[0] = 0;
          normal[1] = -1;
          double dist = y_0[i]+1;
          double cx = -16*pow(sqrt(delta*delta*h*h-dist*dist),3)/3/pi/pow(delta*h,4);
          double cxx = (2*dist*(2*dist*dist-delta*h*delta*h)*sqrt(delta*delta*h*h-dist*dist)/pow(delta*h,4)+2*atan(dist/sqrt(delta*delta*h*h-dist*dist))
                +pi)/pi;
          double cyy = (2*dist*(5*delta*delta*h*h-2*dist*dist)*sqrt(delta*delta*h*h-dist*dist)/3/pow(delta*h,4)+2*atan(dist/sqrt(delta*delta*h*h-dist*dist))+pi)/pi;
          double x_proj = x_0[i];
          double y_proj = -1;
          
          //rhs[i] = (0.5*cxx-dist*cx)*Ffun(x_0[i],y_0[i],s*Dt)-cx*(g_ptr[2*pdim]*normal[0]+g_ptr[2*pdim+1]*normal[1]);
          rhs[i] = (0.5*cxx-dist*cx)*Ffun(x_0[i],y_0[i],s*Dt)-cx*g_ptr[pdim];
          //rhs[i] = (0.5*cxx-dist*cx)*Ffun(x_0[i],y_0[i],s*Dt)-cx*Gfun(x_proj,y_proj,normal[0],normal[1],s*Dt);
            rhs[i] += (0.5*cxx-cx*dist)*u0[i]/Dt;
            pdim = pdim + 1;
        }
        else if(abs(y_0[i]-1) <= delta*h+Tol){ 
          normal[0] = 0;
          normal[1] = 1;
          double dist = -y_0[i]+1;
          double cx = -16*pow(sqrt(delta*delta*h*h-dist*dist),3)/3/pi/pow(delta*h,4);
          double cxx = (2*dist*(2*dist*dist-delta*h*delta*h)*sqrt(delta*delta*h*h-dist*dist)/pow(delta*h,4)+2*atan(dist/sqrt(delta*delta*h*h-dist*dist))
                +pi)/pi;
          double cyy = (2*dist*(5*delta*delta*h*h-2*dist*dist)*sqrt(delta*delta*h*h-dist*dist)/3/pow(delta*h,4)+2*atan(dist/sqrt(delta*delta*h*h-dist*dist))+pi)/pi;
          double x_proj = x_0[i];
          double y_proj = 1;
          
          //rhs[i] = (0.5*cxx-dist*cx)*Ffun(x_0[i],y_0[i],s*Dt)-cx*(g_ptr[2*pdim]*normal[0]+g_ptr[2*pdim+1]*normal[1]);
          rhs[i] = (0.5*cxx-dist*cx)*Ffun(x_0[i],y_0[i],s*Dt)-cx*g_ptr[pdim];
          //rhs[i] = (0.5*cxx-dist*cx)*Ffun(x_0[i],y_0[i],s*Dt)-cx*Gfun(x_proj,y_proj,normal[0],normal[1],s*Dt);
            rhs[i] += (0.5*cxx-cx*dist)*u0[i]/Dt;
            pdim = pdim+1;
        }
        else if((x_0[i]+0.5) <= delta*h+Tol && x_0[i] > -0.5-Tol && (y_0[i] >= 0.5-Tol || y_0[i]<= -0.5+Tol)){ 
          normal[0] = -1;
          normal[1] = 0;
          double dist = x_0[i]+0.5;
          double cx = -16*pow(sqrt(delta*delta*h*h-dist*dist),3)/3/pi/pow(delta*h,4);
          double cxx = (2*dist*(2*dist*dist-delta*h*delta*h)*sqrt(delta*delta*h*h-dist*dist)/pow(delta*h,4)+2*atan(dist/sqrt(delta*delta*h*h-dist*dist))
                +pi)/pi;
          double cyy = (2*dist*(5*delta*delta*h*h-2*dist*dist)*sqrt(delta*delta*h*h-dist*dist)/3/pow(delta*h,4)+2*atan(dist/sqrt(delta*delta*h*h-dist*dist))+pi)/pi;
          double x_proj = -0.5;
          double y_proj = y_0[i];
          
          //rhs[i] = (0.5*cxx-dist*cx)*Ffun(x_0[i],y_0[i],s*Dt)-cx*(g_ptr[2*pdim]*normal[0]+g_ptr[2*pdim+1]*normal[1]);
          rhs[i] = (0.5*cxx-dist*cx)*Ffun(x_0[i],y_0[i],s*Dt)-cx*g_ptr[pdim];
          //rhs[i] = (0.5*cxx-dist*cx)*Ffun(x_0[i],y_0[i],s*Dt)-cx*Gfun(x_proj,y_proj,normal[0],normal[1],s*Dt);
            rhs[i] += (0.5*cxx-cx*dist)*u0[i]/Dt;
            pdim = pdim+1;
        }
        else if((-x_0[i]+0.5) <= delta*h+Tol && x_0[i] < 0.5 + Tol && (y_0[i] >= 0.5-Tol || y_0[i]<= -0.5+Tol)){ 
          normal[0] = 1;
          normal[1] = 0;
          double dist = -x_0[i]+0.5;
          double cx = -16*pow(sqrt(delta*delta*h*h-dist*dist),3)/3/pi/pow(delta*h,4);
          double cxx = (2*dist*(2*dist*dist-delta*h*delta*h)*sqrt(delta*delta*h*h-dist*dist)/pow(delta*h,4)+2*atan(dist/sqrt(delta*delta*h*h-dist*dist))
                +pi)/pi;
          double cyy = (2*dist*(5*delta*delta*h*h-2*dist*dist)*sqrt(delta*delta*h*h-dist*dist)/3/pow(delta*h,4)+2*atan(dist/sqrt(delta*delta*h*h-dist*dist))+pi)/pi;
          double x_proj = 0.5;
          double y_proj = y_0[i];
          
          //rhs[i] = (0.5*cxx-dist*cx)*Ffun(x_0[i],y_0[i],s*Dt)-cx*(g_ptr[2*pdim]*normal[0]+g_ptr[2*pdim+1]*normal[1]);
          rhs[i] = (0.5*cxx-dist*cx)*Ffun(x_0[i],y_0[i],s*Dt)-cx*g_ptr[pdim];
          //rhs[i] = (0.5*cxx-dist*cx)*Ffun(x_0[i],y_0[i],s*Dt)-cx*Gfun(x_proj,y_proj,normal[0],normal[1],s*Dt);
            rhs[i] += (0.5*cxx-cx*dist)*u0[i]/Dt;
            pdim = pdim+1;
        }
        else if((y_0[i]+0.5) <= delta*h+Tol && y_0[i] > -0.5-Tol && (x_0[i] >= 0.5-Tol || x_0[i]<= -0.5+Tol)){ 
          normal[0] = 0;
          normal[1] = -1;
          double dist = y_0[i]+0.5;
          double cx = -16*pow(sqrt(delta*delta*h*h-dist*dist),3)/3/pi/pow(delta*h,4);
          double cxx = (2*dist*(2*dist*dist-delta*h*delta*h)*sqrt(delta*delta*h*h-dist*dist)/pow(delta*h,4)+2*atan(dist/sqrt(delta*delta*h*h-dist*dist))
                +pi)/pi;
          double cyy = (2*dist*(5*delta*delta*h*h-2*dist*dist)*sqrt(delta*delta*h*h-dist*dist)/3/pow(delta*h,4)+2*atan(dist/sqrt(delta*delta*h*h-dist*dist))+pi)/pi;
          double x_proj = x_0[i];
          double y_proj = -0.5;
          
          //rhs[i] = (0.5*cxx-dist*cx)*Ffun(x_0[i],y_0[i],s*Dt)-cx*(g_ptr[2*pdim]*normal[0]+g_ptr[2*pdim+1]*normal[1]);
          rhs[i] = (0.5*cxx-dist*cx)*Ffun(x_0[i],y_0[i],s*Dt)-cx*g_ptr[pdim];
          //rhs[i] = (0.5*cxx-dist*cx)*Ffun(x_0[i],y_0[i],s*Dt)-cx*Gfun(x_proj,y_proj,normal[0],normal[1],s*Dt);
            rhs[i] += (0.5*cxx-cx*dist)*u0[i]/Dt;
            pdim = pdim+1;
        }
        else if((-y_0[i]+0.5) <= delta*h+Tol && y_0[i] < 0.5+Tol && (x_0[i] >= 0.5-Tol || x_0[i]<= -0.5+Tol)){ 
          normal[0] = 0;
          normal[1] = 1;
          double dist = -y_0[i]+0.5;
          double cx = -16*pow(sqrt(delta*delta*h*h-dist*dist),3)/3/pi/pow(delta*h,4);
          double cxx = (2*dist*(2*dist*dist-delta*h*delta*h)*sqrt(delta*delta*h*h-dist*dist)/pow(delta*h,4)+2*atan(dist/sqrt(delta*delta*h*h-dist*dist))
                +pi)/pi;
          double cyy = (2*dist*(5*delta*delta*h*h-2*dist*dist)*sqrt(delta*delta*h*h-dist*dist)/3/pow(delta*h,4)+2*atan(dist/sqrt(delta*delta*h*h-dist*dist))+pi)/pi;
          double x_proj = x_0[i];
          double y_proj = 0.5;
          
          //rhs[i] = (0.5*cxx-dist*cx)*Ffun(x_0[i],y_0[i],s*Dt)-cx*(g_ptr[2*pdim]*normal[0]+g_ptr[2*pdim+1]*normal[1]);
          rhs[i] = (0.5*cxx-dist*cx)*Ffun(x_0[i],y_0[i],s*Dt)-cx*g_ptr[pdim];
          //rhs[i] = (0.5*cxx-dist*cx)*Ffun(x_0[i],y_0[i],s*Dt)-cx*Gfun(x_proj,y_proj,normal[0],normal[1],s*Dt);
            rhs[i] += (0.5*cxx-cx*dist)*u0[i]/Dt;
            pdim = pdim+1;
        }
        else{
          rhs[i] = Ffun(x_0[i],y_0[i],s*Dt);  
          rhs[i] += u0[i]/Dt;
        }
        }

        rhs[0] = u_exact(x_0[0],y_0[0],s*Dt);
        if(UPDATE){
            for(int i=0;i<dim;i++){
                //cout<<rhs[i]<<endl;
            }
        }
        for(int i=0;i<dim*dim;i++){
            stiffK[i] = stiffKT[i];
        }
        dgetrf_(&dim,&dim,stiffK,&dim,IPIV,&info);
        dgetrs_(&C,&dim,&nrhs,stiffK,&dim, IPIV,rhs,&dim,&info);
        
        if(UPDATE)
        {
          for(int i=0;i<dim;i++){
            u0[i] = rhs[i];
          }
          /*ofstream myfile;
          myfile.open("cornermesh.txt");
          for(int i=0;i<dim;i++){
            myfile<<x_0[i]<<" ,"<<y_0[i]<<","<<u0[i]-u_exact(x_0[i],y_0[i],s*Dt)<<endl;
          }
          myfile.close(); */
        
        }

      for(int i=0;i<N_point;i++){
        du0[i] = 0;
      }
    
    for(int i=0;i<kdim;i++){
        for(int s=0;s<bdynei[i].size();s++){
            v_ptr[i] += rhs[bdynei[i][s]]*localbeta[i][s*6];
        }
    }



  for(int i=0;i<N_point;i++){
    for(int j=0;j<N_point;j++){
      //du0[i] += stiffK[i*dim+j]*u0[j];  
      //cout<<stiffK[i*dim+j]<<" ";
    }
    //cout<<endl;
  } 
  double error = 0;
  double errinf = 0 ;

  for(int i=0;i<N_point;i++){
    error += pow(rhs[i]-u_exact(x_0[i],y_0[i],s*Dt),2); 
    //cout<<i<<"   "<<du0[i]<<"   "<<x_0[i]<<"   "<<y_0[i]<<endl;
  }

  error /= (double)N_point;
  error = sqrt(error);
  int maxposition = -1;

  for(int i=0;i<N_point;i++){
    if(fabs(rhs[i]-u_exact(x_0[i],y_0[i],s*Dt)) > errinf){
      errinf = fabs(rhs[i]-u_exact(x_0[i],y_0[i],s*Dt));
      maxposition = i;
    }
  } 
  //cout<<x_0[maxposition]<<"   "<<y_0[maxposition]<<endl;
    
 /* ofstream myfile;
  myfile.open("nonlocalerrorquad40.txt",std::ios_base::app);
 
  myfile<<" L^2 error:  "<<error<<" L_inf error: "<<errinf<<endl;
  myfile.close();
  */
  cout<<" L^2 error:  "<<error<<" L_inf error: "<<errinf<<endl;

}
PYBIND11_MODULE(SIGNATURE, m)
{
    m.def("Preprocess",&Preprocess,"");
    m.def("getBdycoord",&getBdycoord,"");
    m.def("projBdy",&projBdy,"");
    m.def("BE",&BE,"");
}
"""




#u_D = Expression('t*(x[0]*x[0]+x[1]*x[1]+x[0]*x[1])', degree=2, t = 0,domain = mesh)
u_D = Expression('t*t*sin(x[0])*cos(x[1])', degree=2, t = 0,domain = mesh)
#u_D = Expression('t*(sin(0.1*x[0]))',degree = 2, t = 0, domain = mesh)
#u_D = Expression('x[0]',degree = 2, t = 0, domain = mesh)


def boundary(x, on_boundary):
    return on_boundary 
 



n = FacetNormal(mesh)
V = FunctionSpace(mesh,'P', 1)
V_vec = VectorFunctionSpace(mesh,'DG',0)
d = mesh.geometry().dim()
def getNodeCoordinates(V,nFields):
    dofxFlat = V.tabulate_dof_coordinates()
    dofx = dofxFlat.reshape((-1,d))
    nNodes = dofx.shape[0]//nFields
    nodex = zeros((nNodes,d))
    for i in range(0,nNodes):
        nodex[i,:] = dofx[i,:]
    return nodex, nNodes

Nodex, nNodes = getNodeCoordinates(V,1)

class BdryOfInterestver(SubDomain):
    def inside(self,x,on_boundary):
        return ((near(x[0],-1) and x[1] > -0.5-DOLFIN_EPS and x[1] < 0.5+DOLFIN_EPS )
            or(near(x[0],-0.5) and((x[1] > 0.5-DOLFIN_EPS and x[1] < 1+DOLFIN_EPS) or(x[1] >-1-DOLFIN_EPS and x[1] < -0.5+DOLFIN_EPS) ))
            or(near(x[0],0.5) and((x[1] > 0.5-DOLFIN_EPS and x[1] < 1+DOLFIN_EPS) or(x[1] >-1-DOLFIN_EPS and x[1] < -0.5+DOLFIN_EPS) ))
            or(near(x[0],1) and x[1] > -0.5-DOLFIN_EPS and x[1] < 0.5+DOLFIN_EPS )) and on_boundary

class BdryOfInteresthor(SubDomain):
    def inside(self,x,on_boundary):
        return ((near(x[1],-1) and x[0] > -0.5-DOLFIN_EPS and x[0] < 0.5+DOLFIN_EPS )
            or(near(x[1],-0.5) and((x[0] > 0.5-DOLFIN_EPS and x[0] < 1+DOLFIN_EPS) or(x[0] >-1-DOLFIN_EPS and x[0] < -0.5+DOLFIN_EPS) ))
            or(near(x[1],0.5) and((x[0] > 0.5-DOLFIN_EPS and x[0] < 1+DOLFIN_EPS) or(x[0] >-1-DOLFIN_EPS and x[0] < -0.5+DOLFIN_EPS) ))
            or(near(x[1],1) and x[0] > -0.5-DOLFIN_EPS and x[0] < 0.5+DOLFIN_EPS )) and on_boundary

class AntiBdryver(SubDomain):
    def inside(self,x,on_boundary):
        return not ((near(x[0],-1) and x[1] > -0.5-DOLFIN_EPS and x[1] < 0.5+DOLFIN_EPS )
            or(near(x[0],-0.5) and((x[1] > 0.5-DOLFIN_EPS and x[1] < 1+DOLFIN_EPS) or(x[1] >-1-DOLFIN_EPS and x[1] < -0.5+DOLFIN_EPS) ))
            or(near(x[0],0.5) and((x[1] > 0.5-DOLFIN_EPS and x[1] < 1+DOLFIN_EPS) or(x[1] >-1-DOLFIN_EPS and x[1] < -0.5+DOLFIN_EPS) ))
            or(near(x[0],1) and x[1] > -0.5-DOLFIN_EPS and x[1] < 0.5+DOLFIN_EPS )) 
            
class AntiBdryhor(SubDomain):
    def inside(self,x,on_boundary):
        return not ((near(x[1],-1) and x[0] > -0.5-DOLFIN_EPS and x[0] < 0.5+DOLFIN_EPS )
            or(near(x[1],-0.5) and((x[0] > 0.5-DOLFIN_EPS and x[0] < 1+DOLFIN_EPS) or(x[0] >-1-DOLFIN_EPS and x[0] < -0.5+DOLFIN_EPS) ))
            or(near(x[1],0.5) and((x[0] > 0.5-DOLFIN_EPS and x[0] < 1+DOLFIN_EPS) or(x[0] >-1-DOLFIN_EPS and x[0] < -0.5+DOLFIN_EPS) ))
            or(near(x[1],1) and x[0] > -0.5-DOLFIN_EPS and x[0] < 0.5+DOLFIN_EPS )) 
 
class BdryOfInterest(SubDomain):
    def inside(self,x,on_boundary):
        return ((near(x[0],-1) and x[1] > -0.5-DOLFIN_EPS and x[1] < 0.5+DOLFIN_EPS )
            or(near(x[0],-0.5) and((x[1] > 0.5-DOLFIN_EPS and x[1] < 1+DOLFIN_EPS) or(x[1] >-1-DOLFIN_EPS and x[1] < -0.5+DOLFIN_EPS) ))
            or(near(x[0],0.5) and((x[1] > 0.5-DOLFIN_EPS and x[1] < 1+DOLFIN_EPS) or(x[1] >-1-DOLFIN_EPS and x[1] < -0.5+DOLFIN_EPS) ))
            or(near(x[0],1) and x[1] > -0.5-DOLFIN_EPS and x[1] < 0.5+DOLFIN_EPS ) 
            or (near(x[1],-1) and x[0] > -0.5-DOLFIN_EPS and x[0] < 0.5+DOLFIN_EPS )
            or(near(x[1],-0.5) and((x[0] > 0.5-DOLFIN_EPS and x[0] < 1+DOLFIN_EPS) or(x[0] >-1-DOLFIN_EPS and x[0] < -0.5+DOLFIN_EPS) ))
            or(near(x[1],0.5) and((x[0] > 0.5-DOLFIN_EPS and x[0] < 1+DOLFIN_EPS) or(x[0] >-1-DOLFIN_EPS and x[0] < -0.5+DOLFIN_EPS) ))
            or(near(x[1],1) and x[0] > -0.5-DOLFIN_EPS and x[0] < 0.5+DOLFIN_EPS )) and on_boundary
#class BdryOfInterest(SubDomain):
#    def inside(self,x,on_boundary):
#        return  ((near(x[0],-1) and x[1] > -0.5-DOLFIN_EPS and x[1] < 0.5+DOLFIN_EPS )  
#            or (near(x[0],-0.5) and((x[1] > 0.5-DOLFIN_EPS and x[1] < 1+DOLFIN_EPS) or(x[1] >-1-DOLFIN_EPS and x[1] < -0.5+DOLFIN_EPS) ))
#            or (near(x[0],0.5) and((x[1] > 0.5-DOLFIN_EPS and x[1] < 1+DOLFIN_EPS) or(x[1] >-1-DOLFIN_EPS and x[1] < -0.5+DOLFIN_EPS) ))
#            or (near(x[0],1) and x[1] > -0.5-DOLFIN_EPS and x[1] < 0.5+DOLFIN_EPS )) and on_boundary 

class AntiBdry(SubDomain):
    def inside(self,x,on_boundary):
        return  not ((near(x[0],-1) and x[1] > -0.5-DOLFIN_EPS and x[1] < 0.5+DOLFIN_EPS )  
            or (near(x[0],-0.5) and((x[1] > 0.5-DOLFIN_EPS and x[1] < 1+DOLFIN_EPS) or(x[1] >-1-DOLFIN_EPS and x[1] < -0.5+DOLFIN_EPS) ))
            or (near(x[0],0.5) and((x[1] > 0.5-DOLFIN_EPS and x[1] < 1+DOLFIN_EPS) or(x[1] >-1-DOLFIN_EPS and x[1] < -0.5+DOLFIN_EPS) ))
            or (near(x[0],1) and x[1] > -0.5-DOLFIN_EPS and x[1] < 0.5+DOLFIN_EPS ) 
            or (near(x[1],-1) and x[0] > -0.5-DOLFIN_EPS and x[0] < 0.5+DOLFIN_EPS )
            or(near(x[1],-0.5) and((x[0] > 0.5-DOLFIN_EPS and x[0] < 1+DOLFIN_EPS) or(x[0] >-1-DOLFIN_EPS and x[0] < -0.5+DOLFIN_EPS) ))
            or(near(x[1],0.5) and((x[0] > 0.5-DOLFIN_EPS and x[0] < 1+DOLFIN_EPS) or(x[0] >-1-DOLFIN_EPS and x[0] < -0.5+DOLFIN_EPS) ))
            or(near(x[1],1) and x[0] > -0.5-DOLFIN_EPS and x[0] < 0.5+DOLFIN_EPS ))

#class BdryOfInterest1(SubDomain):
#    def inside(self,x,on_boundary):
#        return ((near(x[1],-1) and x[0] > -0.5-DOLFIN_EPS and x[0] < 0.5+DOLFIN_EPS )
#            or(near(x[1],-0.5) and((x[0] > 0.5-DOLFIN_EPS and x[0] < 1+DOLFIN_EPS) or(x[0] >-1-DOLFIN_EPS and x[0] < -0.5+DOLFIN_EPS) ))
#            or(near(x[1],0.5) and((x[0] > 0.5-DOLFIN_EPS and x[0] < 1+DOLFIN_EPS) or(x[0] >-1-DOLFIN_EPS and x[0] < -0.5+DOLFIN_EPS) ))
#            or(near(x[1],1) and x[0] > -0.5-DOLFIN_EPS and x[0] < 0.5+DOLFIN_EPS )) and on_boundary 
#class AntiBdry1(SubDomain):
#    def inside(self,x,on_boundary):
#        return not ((near(x[1],-1) and x[0] > -0.5-DOLFIN_EPS and x[0] < 0.5+DOLFIN_EPS )
#            or(near(x[1],-0.5) and((x[0] > 0.5-DOLFIN_EPS and x[0] < 1+DOLFIN_EPS) or(x[0] >-1-DOLFIN_EPS and x[0] < -0.5+DOLFIN_EPS) ))
#            or(near(x[1],0.5) and((x[0] > 0.5-DOLFIN_EPS and x[0] < 1+DOLFIN_EPS) or(x[0] >-1-DOLFIN_EPS and x[0] < -0.5+DOLFIN_EPS) ))
#            or(near(x[1],1) and x[0] > -0.5-DOLFIN_EPS and x[0] < 0.5+DOLFIN_EPS )) 




antiBCsver = [DirichletBC(V,Constant(0.0),AntiBdryver()),]
antiBCshor = [DirichletBC(V,Constant(0.0),AntiBdryhor()),]
antiBCs = [DirichletBC(V,Constant(0.0),AntiBdry()),]
k = 0
Bdy = zeros(nNodes)

for i in range(nNodes):
    if((near(Nodex[i,0],-1) and Nodex[i,1] > -0.5-DOLFIN_EPS and Nodex[i,1] < 0.5+DOLFIN_EPS )
    or(near(Nodex[i,0],-0.5) and((Nodex[i,1] > 0.5-DOLFIN_EPS and Nodex[i,1] < 1+DOLFIN_EPS) or(Nodex[i,1] >-1-DOLFIN_EPS and Nodex[i,1] < -0.5+DOLFIN_EPS) ))
    or(near(Nodex[i,0],0.5) and((Nodex[i,1] > 0.5-DOLFIN_EPS and Nodex[i,1] < 1+DOLFIN_EPS) or(Nodex[i,1] >-1-DOLFIN_EPS and Nodex[i,1] < -0.5+DOLFIN_EPS) ))
    or(near(Nodex[i,0],1) and Nodex[i,1] > -0.5-DOLFIN_EPS and Nodex[i,1] < 0.5+DOLFIN_EPS )
    or(near(Nodex[i,1],-1) and Nodex[i,0] > -0.5-DOLFIN_EPS and Nodex[i,0] < 0.5+DOLFIN_EPS )
    or(near(Nodex[i,1],-0.5) and((Nodex[i,0] > 0.5-DOLFIN_EPS and Nodex[i,0] < 1+DOLFIN_EPS) or(Nodex[i,0] >-1-DOLFIN_EPS and Nodex[i,0] < -0.5+DOLFIN_EPS) ))
    or(near(Nodex[i,1],0.5) and((Nodex[i,0] > 0.5-DOLFIN_EPS and Nodex[i,0] < 1+DOLFIN_EPS) or(Nodex[i,0] >-1-DOLFIN_EPS and Nodex[i,0] < -0.5+DOLFIN_EPS) ))
    or(near(Nodex[i,1],1) and Nodex[i,0] > -0.5-DOLFIN_EPS and Nodex[i,0] < 0.5+DOLFIN_EPS )):
        Bdy[k] = i
        k = k+1

localBdycoord = zeros(2*k)

for i in range(0,k):
    localBdycoord[2*i] = Nodex[int(Bdy[i]),0]
    localBdycoord[2*i+1] = Nodex[int(Bdy[i]),1]



bc = DirichletBC(V, u_D,boundary)
VscalarE = FiniteElement("Lagrange",mesh.ufl_cell(),1)
Vscalar = FunctionSpace(mesh,VscalarE)
ubc = Function(Vscalar)

BC = DirichletBC(V,ubc,"on_boundary")

u_n = interpolate(u_D,V)

u = TrialFunction(V)
v = TestFunction(V)
f = Constant(0)

#f = Expression('x[0]*x[0]+x[1]*x[1]+x[0]*x[1]-4*t',degree = 2, t = 0)
f = Expression('2*t*sin(x[0])*cos(x[1])+2*t*t*sin(x[0])*cos(x[1])',degree = 2, t = 0)
#f = Expression('2*sin(x[0])*cos(x[1])',degree = 2, t = 0)
#f = Expression('sin(0.1*x[0]) + 0.01*t*sin(0.1*x[0])',degree = 2, t =0)
#Du = Expression(('4*x[0]*x[0]*x[0]','4*alpha*x[1]*x[1]*x[1]'), degree = 2,alpha = alpha)
        
F = u*v*dx + dt*dot(grad(u),grad(v))*dx - (u_n + dt*f)*v*dx
a,L = lhs(F), rhs(F)

def res(u,v):
    return (u*v/dt - u_n*v/dt + inner(grad(u),grad(v)) - f*v)*dx

u_current = Function(V)
u_lastIter = Function(V)

#FLUX_BDRYver = 1
#FLUX_BDRYhor = 2
#COMPLEMENT_FLUX_BDRY = 3
#boundaryMarkers = MeshFunction("size_t",mesh,mesh.topology().dim()-1,
#                              COMPLEMENT_FLUX_BDRY)
#BdryOfInterestver().mark(boundaryMarkers,FLUX_BDRYver)
#BdryOfInteresthor().mark(boundaryMarkers,FLUX_BDRYhor)
#ds_marked = ds(subdomain_data=boundaryMarkers)

qn = TrialFunction(V)
#n = FacetNormal(mesh)
#consistencyTermver = inner(grad(u_current),n)*v*ds_marked(COMPLEMENT_FLUX_BDRY) + inner(grad(u_current),n)*v*ds_marked(FLUX_BDRYhor)
#consistencyTermhor = inner(grad(u_current),n)*v*ds_marked(COMPLEMENT_FLUX_BDRY) + inner(grad(u_current),n)*v*ds_marked(FLUX_BDRYver)
#A=assemble(consistencyTermver)
#np.savetxt("verrhs.txt",A)

#file = File("boundaryMarkers.pvd")
#file << boundaryMarkers

#FBdryver = qn*v*ds_marked(FLUX_BDRYver) - res(u_current,v) + consistencyTermver
#FBdryhor = qn*v*ds_marked(FLUX_BDRYhor) - res(u_current,v) + consistencyTermhor
 
FLUX_BDRY = 1
COMPLEMENT_FLUX_BDRY = 2
boundaryMarkers = MeshFunction("size_t",mesh,mesh.topology().dim()-1,
                               COMPLEMENT_FLUX_BDRY)
BdryOfInterest().mark(boundaryMarkers,FLUX_BDRY)
ds_marked = ds(subdomain_data=boundaryMarkers)

n = FacetNormal(mesh)
consistencyTerm = inner(grad(u_current),n)*v*ds_marked(COMPLEMENT_FLUX_BDRY)
     
FBdry = qn*v*ds_marked(FLUX_BDRY) - res(u_current,v) + consistencyTerm





def solveFor_qn_h(FBdry, BCs):
    aBdry = lhs(FBdry)
    LBdry = rhs(FBdry)
    ABdry = assemble(aBdry,keep_diagonal=True)
    bBdry = assemble(LBdry)
    [BC.apply(ABdry,bBdry) for BC in BCs]
    qn_h = Function(V)
    solve(ABdry,qn_h.vector(),bBdry)
    return qn_h
        
def fluxErr(qn):
    err = inner(grad(u_D),n) - qn
    return math.sqrt(assemble(err*err*ds_marked(FLUX_BDRY)))
        
ext_module = compile_cpp_code(cpp_code, libs=["lapack"])
ext_module.Preprocess()
ext_module.getBdycoord(k,localBdycoord)

projcoord = zeros(8*N*N)
projdim = zeros(1)

ext_module.projBdy(projdim,projcoord)

u = Function(V)
vout = zeros(k)
gin = zeros(int(projdim[0]))
gin_last = zeros(int(projdim[0]))


MAX_SUBITERS = 200
SUBITER_TOL = 1e-8
SUB_TOTAL = 0

t = 0
for i in range(int(projdim[0])):
    #gin[i] = 1*projcoord[4*i+2] + Rbeta*projcoord[4*i]
    gin[i] = 0.0

for s in range(num_steps):
    t += dt 
    u_D.t = t
    f.t = t
    print("------- Time step "+str(s+1)
          +" , t = "+str(t)+"-------\n")
    
    for subIters in range(0, 1):
        SUB_TOTAL = SUB_TOTAL+1

        ext_module.BE(s+1,k,vout,gin,1); 
        
        ubc.assign(interpolate(u_D,V))
        uBCarray = ubc.vector().get_local()
        for i in range(k):
            uBCarray[int(Bdy[i])] = vout[i]
        
        ubc.vector().set_local(uBCarray)
        
        solve(a == L, u_current, BC)
        

        #qn_hver = solveFor_qn_h(FBdryver, antiBCsver)
        #qn_hhor = solveFor_qn_h(FBdryhor, antiBCshor)
        #qn_h = solveFor_qn_h(FBdry,antiBCs)
        #err1 = fluxErr(qn_h)
        #print(qn_h)

        #FLUX_BDRY = 1
        #COMPLEMENT_FLUX_BDRY = 2
        #boundaryMarkers = MeshFunction("size_t",mesh,mesh.topology().dim()-1,
        #                       COMPLEMENT_FLUX_BDRY)
        #BdryOfInterest1().mark(boundaryMarkers,FLUX_BDRY)
        #ds_marked = ds(subdomain_data=boundaryMarkers)

        #qn = TrialFunction(V)
        #n = FacetNormal(mesh)
        #consistencyTerm = inner(grad(u_current),n)*v*ds_marked(COMPLEMENT_FLUX_BDRY)
     
        #FBdry = qn*v*ds_marked(FLUX_BDRY) - res(u_current,v) + consistencyTerm


        #qn_h1 = solveFor_qn_h(FBdry, antiBCs1)
        
        #print(qn_h1)

        #err2 = fluxErr(qn_h1)
        
        
        #print("Error in consistent flux on boundary of interest: "
            #+str(err1))


        gradu = project(grad(u_current),V_vec)
        #if(subIters == 0):
        for i in range(int(projdim[0])):
            gin[i] = gradu(projcoord[4*i],projcoord[4*i+1])[0]*projcoord[4*i+2]+  gradu(projcoord[4*i],projcoord[4*i+1])[1]*projcoord[4*i+3]+Rbeta*(u_current(projcoord[4*i],projcoord[4*i+1]))   
            #gin[i] = t*t*cos(projcoord[4*i])*cos(projcoord[4*i+1])*projcoord[4*i+2]-t*t*sin(projcoord[4*i])*sin(projcoord[4*i+1])*projcoord[4*i+3]+Rbeta*t*t*sin(projcoord[4*i])*cos(projcoord[4*i+1])
                #gin[i] = projcoord[4*i+1]
                #print(gin[i])
                #gin[i] = gradu(projcoord[2*i],projcoord[2*i+1])[0]*projcoord[2*i] + gradu(projcoord[2*i],projcoord[2*i+1])[1]*projcoord[2*i+1]
                #gin[i] = gin[i] + Rbeta*(u_current(projcoord[2*i],projcoord[2*i+1]))
                #gin[i] = t*t*2*(projcoord[2*i]*projcoord[2*i]+projcoord[2*i+1]*projcoord[2*i+1]) + 1/N_ELEM
                #gin[i] = t*(projcoord[2*i]*cos(projcoord[2*i])*cos(projcoord[2*i+1])-projcoord[2*i+1]*sin(projcoord[2*i])*sin(projcoord[2*i+1]))
        #else:
            #for i in range(int(projdim[0])):
                #if((near(projcoord[2*i],-1) and projcoord[2*i+1] > -0.5-DOLFIN_EPS and projcoord[2*i+1] < 0.5+DOLFIN_EPS )  
                #    or (near(projcoord[2*i],-0.5) and((projcoord[2*i+1] > 0.5-DOLFIN_EPS and projcoord[2*i+1] < 1+DOLFIN_EPS) 
                #        or(projcoord[2*i+1] >-1-DOLFIN_EPS and projcoord[2*i+1] < -0.5+DOLFIN_EPS) ))
                #    or (near(projcoord[2*i],0.5) and((projcoord[2*i+1] > 0.5-DOLFIN_EPS and projcoord[2*i+1] < 1+DOLFIN_EPS) 
                 #       or(projcoord[2*i+1] >-1-DOLFIN_EPS and projcoord[2*i+1] < -0.5+DOLFIN_EPS) ))
                 #   or (near(projcoord[2*i],1) and projcoord[2*i+1] > -0.5-DOLFIN_EPS and projcoord[2*i+1] < 0.5+DOLFIN_EPS )): 
                #gin[i] = -1*qn_h(projcoord[2*i],projcoord[2*i+1]) + Rbeta*(u_current(projcoord[2*i],projcoord[2*i+1]))
                #gin[i] = 0.1*(-1*qn_h(projcoord[2*i],projcoord[2*i+1]) + Rbeta*(u_current(projcoord[2*i],projcoord[2*i+1])))+0.9*gin_last[i]
                #gin[i] = (-1*qn_h(projcoord[2*i],projcoord[2*i+1]))
                #gin[i] = 0.1*(-1*qn_h(projcoord[2*i],projcoord[2*i+1]))+0.9*gin_last[i]
                #else:
                #    gin[i] = 0.01*(-1*qn_h1(projcoord[2*i],projcoord[2*i+1]))+0.99*gin_last[i]
                #gin[i] = gradu(projcoord[2*i],projcoord[2*i+1])[0]*projcoord[2*i] + gradu(projcoord[2*i],projcoord[2*i+1])[1]*projcoord[2*i+1]
                #gin[i] = gin[i] + Rbeta*(u_current(projcoord[2*i],projcoord[2*i+1]))
                #gin[i] = t*t*2*(projcoord[2*i]*projcoord[2*i]+projcoord[2*i+1]*projcoord[2*i+1]) + 1/N_ELEM
                #gin[i] = t*(projcoord[2*i]*cos(projcoord[2*i])*cos(projcoord[2*i+1])-projcoord[2*i+1]*sin(projcoord[2*i])*sin(projcoord[2*i+1]))
        
        for i in range(int(projdim[0])):
            gin_last[i] = gin[i]
        
        diff = Function(V) 
        diff.assign(u_current - u_lastIter)
        
        u_lastIter.assign(u_current)

        normDiff = math.sqrt(assemble(((diff)**2)*dx))
        normLastIter = math.sqrt(assemble((u_lastIter**2)*dx))
        rel = normDiff/normLastIter

        print("Relative change: "+str(rel))

        if(rel < SUBITER_TOL):
            break

   
    #ext_module.BE(s+1,k,vout,gin,1)
    u_n.assign(u_current)


    u_e = interpolate(u_D,V)
    #gradu = interpolate(Du,V_vec)
    
    #gradu1 = project(grad(u),V_vec)
    
    #derror = np.abs(gradu.vector().get_local()-gradu1.vector().get_local())
    errorinf = np.abs(u_e.vector().get_local()-u_current.vector().get_local()).max()
    error2 = math.sqrt(assemble(((u_current-u_e)**2)*dx))
    
    print('t = %.8f: errorinf=%.10f\n' %(t,errorinf))
    print('t = %.8f: error2=%.10f\n' %(t,error2))

SUB_TOTAL = SUB_TOTAL/num_steps
print('Number of subiteration: %.10f\n' %(SUB_TOTAL))

#for i in range(0,nNodes):
#    print('%.8f, %.8f, %.8f\n' %(Nodex[i,0],Nodex[i,1],u_current.vector().get_local()[i]),file=open("cornerlocal.txt","a"))

