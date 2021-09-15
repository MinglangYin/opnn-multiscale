import sys
from dolfin import *
from mshr import *
import numpy as np
import ufl
from numpy import zeros
from numpy import linalg as LA
from numpy.linalg import norm as npNorm
import math


T = 1.0
N_ELEM = 10
N_steps = int(N_ELEM*N_ELEM/10)
dt = T/N_steps
Rbeta = 0.2*N_ELEM

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

    Type Householder(vvector<Type>& v){//x的Householder变换,返回beta
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
  return t*sin(x)*cos(y);//cos(myB*pi*x)*sin(myB*pi*y);//t*cos(myB*pi*x)*sin(myB*pi*y);
    //return x*x+t;
}



double Ffun(double x,double y,double t){
  return sin(x)*cos(y)+2*t*sin(x)*cos(y);//4*Bulk*myB*pi*myB*pi*cos(myB*pi*x)*sin(myB*pi*y)/5;//4*Bulk*myB*pi*myB*pi*t*cos(myB*pi*x)*sin(myB*pi*y)/5;
    //return -1;
}

double Gfun(double x, double y,double t){
  return 2*x*x;//-4.*Bulk*myB*pi*sin(myB*pi*x)*sin(myB*pi*y)/5.;//-4.*Bulk*myB*pi*t*sin(myB*pi*x)*sin(myB*pi*y)/5.;
}


////////////////////////////////////////////////////////////////////////////


//////////////////////////////////////////////////////////////////////////////





int N="""+str(N_ELEM)+""";
const double delta=3.9;
double h = 1./N;            // lattice constant, all the nodes are evenly distributed 
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
double* circlebdy;
int dim;


double diff = 100;
const double totalT="""+str(T)+""";
const int nstep = N*N/10;//time-step 
const double Dt = totalT/nstep;//0.0001; // time-stepsize 
const double rbeta = """+str(Rbeta)+""";

int* IPIV;
int info;
int nrhs = 1;
char C='T';
char D = 'A';
int N_point = 0;

void Preprocess(){
  for(int i=0;i < 2*N+1; i++){
    for(int j=0;j < 2*N+1;j++){
      if((-1+i*h)*(-1+i*h) + (-1+j*h)*(-1+j*h) <= 1){
        N_point = N_point+1;
      }
    }
  }
  x_0 = new double[N_point];
  y_0 = new double[N_point];
  u0 = new double[N_point];
  u = new double[N_point];
  rhs = new double[N_point];
  stiffKT = new double[N_point*N_point];
  stiffK = new double[N_point*N_point];
  dim = N_point;


  int s = 0;
  for(int i=0;i < 2*N+1; i++){
    for(int j=0;j < 2*N+1;j++){
      if((-1+i*h)*(-1+i*h) + (-1+j*h)*(-1+j*h) <= 1){
        x_0[s] = -1+j*h;
        y_0[s] = -1+i*h;
        s = s+1;
      }
    }
  }

vector< vector<int> > nei(N_point); //neighborlist
  
  for(int i=0; i<N_point;i++){
    u0[i] = u_exact(x_0[i],y_0[i],0);
  }

  for(int i=0;i<N_point;i++){
    for(int j=0;j<N_point;j++){
      if((x_0[j]-x_0[i])*(x_0[j]-x_0[i]) + (y_0[j]-y_0[i])*(y_0[j]-y_0[i]) < delta*delta*h*h){
        nei[i].push_back(j);
      }  
    }
  }

/*  for(int i=0;i<N_point;i++){
    for(int j=0;j!=nei[i].size();j++){
      cout<<nei[i][j]<<" ";
    }
    cout<<endl;
  }
*/
  for(int i=0;i<N_point*N_point;i++){
        stiffKT[i] = 0;
      }
  
  for(int i=1;i<N_point;i++){
        double* P = new double[nei[i].size()*6];
        double* M = new double[6*6];
        double* D = new double[nei[i].size()]; 
        double* Beta = new double[nei[i].size()*6];
        if(x_0[i]*x_0[i] + y_0[i]*y_0[i] < (1-delta*h)*(1-delta*h)){
          for(int s=0;s!=nei[i].size();s++){
            P[s*6] = 1;
            P[s*6+1] = x_0[nei[i][s]]-x_0[i];
            P[s*6+2] = y_0[nei[i][s]]-y_0[i];
            P[s*6+3] = pow(x_0[nei[i][s]]-x_0[i],2);
            P[s*6+4] = pow(y_0[nei[i][s]]-y_0[i],2);
            P[s*6+5] = (x_0[nei[i][s]]-x_0[i])*(y_0[nei[i][s]]-y_0[i]);
            D[s] =  weight(sqrt(pow(P[s*6+1],2)+pow(P[s*6+2],2)),delta*h);
          }
        }
        else{
          normal[0] = x_0[i]/sqrt(x_0[i]*x_0[i]+y_0[i]*y_0[i]);
          normal[1] = y_0[i]/sqrt(x_0[i]*x_0[i]+y_0[i]*y_0[i]);
          tanvec[0] = -y_0[i]/sqrt(x_0[i]*x_0[i]+y_0[i]*y_0[i]);
          tanvec[1] = x_0[i]/sqrt(x_0[i]*x_0[i]+y_0[i]*y_0[i]);

          for(int s=0;s!=nei[i].size();s++){
            P[s*6] = 1;
            P[s*6+1] = (x_0[nei[i][s]] - x_0[i])*normal[0] + (y_0[nei[i][s]]-y_0[i])*normal[1]; 
            P[s*6+2] = (x_0[nei[i][s]] - x_0[i])*tanvec[0] + (y_0[nei[i][s]]-y_0[i])*tanvec[1];
            P[s*6+3] = pow((x_0[nei[i][s]] - x_0[i])*normal[0] + (y_0[nei[i][s]]-y_0[i])*normal[1],2);
            P[s*6+4] = pow((x_0[nei[i][s]] - x_0[i])*tanvec[0] + (y_0[nei[i][s]]-y_0[i])*tanvec[1],2);
            P[s*6+5] =((x_0[nei[i][s]] - x_0[i])*normal[0] + (y_0[nei[i][s]]-y_0[i])*normal[1])*((x_0[nei[i][s]] - x_0[i])*tanvec[0] + (y_0[nei[i][s]]-y_0[i])*tanvec[1]);
            D[s] =  weight(sqrt(pow(P[s*6+1],2)+pow(P[s*6+2],2)),delta*h);
          }
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

        


        if(x_0[i]*x_0[i] + y_0[i]*y_0[i] < (1-delta*h)*(1-delta*h)){
                for(int s=0;s<nei[i].size();s++){
                    stiffKT[i*dim+nei[i][s]] -= Beta[s*6+3]*2 + Beta[s*6+4]*2;
                }
                stiffKT[i*dim+i] += 1./Dt;
        }
        else{
          normal[0] = x_0[i]/sqrt(x_0[i]*x_0[i]+y_0[i]*y_0[i]);
          normal[1] = y_0[i]/sqrt(x_0[i]*x_0[i]+y_0[i]*y_0[i]);
          tanvec[0] = -y_0[i]/sqrt(x_0[i]*x_0[i]+y_0[i]*y_0[i]);
          tanvec[1] = x_0[i]/sqrt(x_0[i]*x_0[i]+y_0[i]*y_0[i]);
          double x1 = x_0[i]*(1+x_0[i]*x_0[i]+y_0[i]*y_0[i]-delta*h*delta*h)/2/(x_0[i]*x_0[i]+y_0[i]*y_0[i]);
          double y1 = y_0[i]*(1+x_0[i]*x_0[i]+y_0[i]*y_0[i]-delta*h*delta*h)/2/(x_0[i]*x_0[i]+y_0[i]*y_0[i]);
          double x0 = x_0[i]*normal[0] + y_0[i]*normal[1];

          double d1 =  sqrt(x1*x1+y1*y1);
          double dist = (x1-x_0[i])*normal[0] + (y1-y_0[i])*normal[1];
          double theta = acos(d1);
          double d_proj = 1-x0;
          double x_proj = x_0[i]/sqrt(x_0[i]*x_0[i]+y_0[i]*y_0[i]);  
          double y_proj = y_0[i]/sqrt(x_0[i]*x_0[i]+y_0[i]*y_0[i]);  

          double c1 = -16*(pow(sqrt(pow(delta*h,2)-pow(dist,2)),3))/3/pi/pow(delta*h,4) 
            +8*(2*sin(theta)/3-2*pow(d1,3)*tan(theta)/3-x0*(theta-d1*d1*tan(theta)))/pi/pow(delta*h,4);
          double c2 = (pi+2*dist*(2*dist*dist-delta*delta*h*h)*sqrt(delta*delta*h*h-dist*dist)/pow(delta*h,4)
                          +2*atan(dist/sqrt(delta*delta*h*h-dist*dist)))/pi
                    +8*((2*theta+sin(2*theta))/8 - 4*x0*sin(theta)/3+x0*x0*theta-(pow(d1,4)/2-4*x0*pow(d1,3)/3+x0*x0*d1*d1)*tan(theta))/pi/pow(delta*h,4);
        
          double c3 = (pi+2*dist*(5*delta*delta*h*h-2*dist*dist)*sqrt(delta*delta*h*h-dist*dist)/3/pow(delta*h,4)
                          +2*atan(dist/sqrt(delta*delta*h*h-dist*dist)))/pi + 8*((2*theta-sin(2*theta))/8 - pow(d1,4)*pow(tan(theta),3)/6)/pi/pow(delta*h,4);
          
          for(int s=0;s<nei[i].size();s++){
            stiffKT[i*dim+nei[i][s]] -= Beta[s*6+1]*c1 + Beta[s*6+3]*c2 + Beta[s*6+4]*c3+Beta[s*6+4]*(0.5*(c2-c3)-d_proj*c1)*2;
            //stiffKT[i*dim+nei[i][s]] -= Beta[s*6+1]*c1 + Beta[s*6+3]*c2 + Beta[s*6+4]*c3+(0.5*(c2-c3)-d_proj*c1)*(
            //Beta[s*6+1]*(3*(2*x0*x0*sin(delta*h/x0)-2*x0*delta*h)/pow(delta*h,3))
            //+Beta[s*6+3]*(3*(3*x0*x0*delta*h+0.5*x0*x0*x0*sin(2*delta*h/x0)-4*x0*x0*x0*sin(delta*h/x0))/pow(delta*h,3))
            //+Beta[s*6+5]*(3*(delta*h-0.5*x0*sin(2*delta*h/x0))/pow(delta*h,3)));
          }
          stiffKT[i*dim+i] += (0.5*c2-d_proj*c1)/Dt; 
            
            vector<int> projnei;
            for(int j=0;j<N_point;j++){
                if((x_0[j]-x_proj)*(x_0[j]-x_proj) + (y_0[j]-y_proj)*(y_0[j]-y_proj) < delta*delta*h*h){
                    projnei.push_back(j);    
                }    
            }
            double* P1 = new double[projnei.size()*6];
            double* M1 = new double[6*6];
            double* D1 = new double[projnei.size()]; 
            double* Beta1 = new double[projnei.size()*6];
            
            for(int s=0;s!=projnei.size();s++){
                P1[s*6]  = 1;
                P1[s*6+1] = x_0[projnei[s]] - x_proj;
                P1[s*6+2] = y_0[projnei[s]] - y_proj;   
                P1[s*6+3] = pow(x_0[projnei[s]] - x_proj,2);
                P1[s*6+4] = pow(y_0[projnei[s]] - y_proj,2);
                P1[s*6+5] = (x_0[projnei[s]] - x_proj)*(y_0[projnei[s]] - y_proj);
                D1[s] =  weight(sqrt(pow(P1[s*6+1],2)+pow(P1[s*6+2],2)),delta*h);
            }
            
            for(int m=0;m<6;m++){
                for(int n=0;n<6;n++){
                      M1[m*6+n] = 0;
                      for(int s=0;s<projnei.size();s++){
                        M1[m*6+n] += P1[s*6+m]*D1[s]*P1[s*6+n];
                     }
                }
            }
            
            inverse(M1,6);

            for(int m=0;m<projnei.size();m++){
                for(int n=0;n<6;n++){
                    Beta1[m*6+n] = 0;
                  for(int s=0;s<6;s++){
                      Beta1[m*6+n] += D1[m]*P1[m*6+s]*M1[s*6+n];
                  }
              }
          }

          for(int j=0;j<projnei.size();j++){
                //stiffKT[i*dim+projnei[j]] += (-c1+(0.5*(c2-c3)-d_proj*c1)/x0)*rbeta*Beta1[j*6];      
                stiffKT[i*dim+projnei[j]] += -c1*rbeta*Beta1[j*6];      
            }
          
            delete[] P1;
            delete[] D1;
            delete[] M1;
            delete[] Beta1;
        }
        delete[] P;
        delete[] D;
        delete[] M;
        delete[] Beta;
    }
  //for(int i=0;i<dim;i++){
  //  stiffKT[i] = 0;
  //}
  stiffKT[0] = 1;
  for(int i=0;i<dim;i++){
    for(int j=0;j<dim;j++){
      //cout<<stiffKT[i*dim+j]<<" ";
    }
   // cout<<endl;
  }
}

void getBdycoord(int kdim, npArray coord){
    auto coord_buffer = coord.request();
    double *coord_ptr = (double *)coord_buffer.ptr; 

    circlebdy = new double[2*kdim];
    for(int i=0;i<kdim;i++){
        circlebdy[2*i] = coord_ptr[2*i];
        circlebdy[2*i+1] = coord_ptr[2*i+1];
    }

  /*bdynei.resize(kdim);
  for(int i=0;i<kdim;i++){
    for(int j=0;j<N_point;j++){
        if(pow(x_0[j]-circlebdy[2*i],2)+pow(y_0[j]-circlebdy[2*i+1],2) < delta*delta*h*h){
            bdynei[i].push_back(j);
        }
    }
  }

  circlebeta = new double*[kdim];
  for(int i=0;i<kdim;i++){
        double* P = new double[bdynei[i].size()*6];
        double* M = new double[6*6];
        double* D = new double[bdynei[i].size()]; 
        circlebeta[i] = new double[bdynei[i].size()*6];
          
        normal[0] = circlebdy[2*i]/sqrt(circlebdy[2*i]*circlebdy[2*i]+circlebdy[2*i+1]*circlebdy[2*i+1]);
        normal[1] = circlebdy[2*i+1]/sqrt(circlebdy[2*i]*circlebdy[2*i]+circlebdy[2*i+1]*circlebdy[2*i+1]);
        tanvec[0] = -circlebdy[2*i+1]/sqrt(circlebdy[2*i]*circlebdy[2*i]+circlebdy[2*i+1]*circlebdy[2*i+1]);
        tanvec[1] = circlebdy[2*i]/sqrt(circlebdy[2*i]*circlebdy[2*i]+circlebdy[2*i+1]*circlebdy[2*i+1]);
        
        for(int s=0;s!=bdynei[i].size();s++){
            P[s*6] = 1;
            P[s*6+1] = (x_0[bdynei[i][s]]-circlebdy[2*i])*normal[0]+(y_0[bdynei[i][s]]-circlebdy[2*i+1])*normal[1];
            P[s*6+2] = (x_0[bdynei[i][s]]-circlebdy[2*i])*tanvec[0]+(y_0[bdynei[i][s]]-circlebdy[2*i+1])*tanvec[1];
            P[s*6+3] = pow((x_0[bdynei[i][s]]-circlebdy[2*i])*normal[0]+(y_0[bdynei[i][s]]-circlebdy[2*i+1])*normal[1],2);
            P[s*6+4] = pow((x_0[bdynei[i][s]]-circlebdy[2*i])*tanvec[0]+(y_0[bdynei[i][s]]-circlebdy[2*i+1])*tanvec[1],2);
            P[s*6+5] = (x_0[bdynei[i][s]]-circlebdy[2*i])*(y_0[bdynei[i][s]]-circlebdy[2*i+1]);
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
                  circlebeta[i][m*6+n] = 0;
                  for(int s=0;s<6;s++){
                      circlebeta[i][m*6+n] += D[m]*P[m*6+s]*M[s*6+n];
                  }
              }
          }
              
  }  
*/
}


void projBdy(npArray pdim, npArray proj){
    auto pdim_buffer = pdim.request();
    double *pdim_ptr = (double *)pdim_buffer.ptr;
    auto proj_buffer = proj.request();
    double *proj_ptr = (double *)proj_buffer.ptr;
    
    int pp = 0;
    for(int i=1;i<N_point;i++){
         if(x_0[i]*x_0[i] + y_0[i]*y_0[i] >= (1-delta*h)*(1-delta*h)){
            proj_ptr[2*pp] = x_0[i]/sqrt(x_0[i]*x_0[i]+y_0[i]*y_0[i]);
            proj_ptr[2*pp+1] = y_0[i]/sqrt(x_0[i]*x_0[i]+y_0[i]*y_0[i]);
            pp = pp+1;
        }
    }
    pdim_ptr[0] = (double)pp;
}



void BE(int s,int kdim, npArray v, npArray g, int UPDATE){

  auto v_buffer  = v.request();
  double *v_ptr = (double *)v_buffer.ptr; 
  auto g_buffer = g.request();
  double *g_ptr = (double *)g_buffer.ptr;
    
  for(int i=0;i<kdim;i++){
    v_ptr[i] = 0;
  }

  IPIV = new int[dim];
  
  for(int i=0;i<N_point;i++){
    rhs[i] = 0;
  }

    int pdim = 0;
  for(int i=1;i<N_point;i++){
      if(x_0[i]*x_0[i] + y_0[i]*y_0[i] < (1-delta*h)*(1-delta*h)){
        rhs[i] = Ffun(x_0[i],y_0[i],s*Dt) + u0[i]/Dt;
      }
      else{
          normal[0] = x_0[i]/sqrt(x_0[i]*x_0[i]+y_0[i]*y_0[i]);
          normal[1] = y_0[i]/sqrt(x_0[i]*x_0[i]+y_0[i]*y_0[i]);
          tanvec[0] = -y_0[i]/sqrt(x_0[i]*x_0[i]+y_0[i]*y_0[i]);
          tanvec[1] = x_0[i]/sqrt(x_0[i]*x_0[i]+y_0[i]*y_0[i]);
          double x1 = x_0[i]*(1+x_0[i]*x_0[i]+y_0[i]*y_0[i]-delta*h*delta*h)/2/(x_0[i]*x_0[i]+y_0[i]*y_0[i]);
          double y1 = y_0[i]*(1+x_0[i]*x_0[i]+y_0[i]*y_0[i]-delta*h*delta*h)/2/(x_0[i]*x_0[i]+y_0[i]*y_0[i]);
          double x0 = x_0[i]*normal[0] + y_0[i]*normal[1];

          double d1 =  sqrt(x1*x1+y1*y1);
          double dist = (x1-x_0[i])*normal[0] + (y1-y_0[i])*normal[1];
          double theta = acos(d1);
          double d_proj = 1-x0;
          double x_proj = x_0[i]/sqrt(x_0[i]*x_0[i]+y_0[i]*y_0[i]);  
          double y_proj = y_0[i]/sqrt(x_0[i]*x_0[i]+y_0[i]*y_0[i]);  

          double c1 = -16*(pow(sqrt(pow(delta*h,2)-pow(dist,2)),3))/3/pi/pow(delta*h,4) 
            +8*(2*sin(theta)/3-2*pow(d1,3)*tan(theta)/3-x0*(theta-d1*d1*tan(theta)))/pi/pow(delta*h,4);
          double c2 = (pi+2*dist*(2*dist*dist-delta*delta*h*h)*sqrt(delta*delta*h*h-dist*dist)/pow(delta*h,4)
                          +2*atan(dist/sqrt(delta*delta*h*h-dist*dist)))/pi
                    +8*((2*theta+sin(2*theta))/8 - 4*x0*sin(theta)/3+x0*x0*theta-(pow(d1,4)/2-4*x0*pow(d1,3)/3+x0*x0*d1*d1)*tan(theta))/pi/pow(delta*h,4);
      
          double c3 = (pi+2*dist*(5*delta*delta*h*h-2*dist*dist)*sqrt(delta*delta*h*h-dist*dist)/3/pow(delta*h,4)
                +2*atan(dist/sqrt(delta*delta*h*h-dist*dist)))/pi + 8*((2*theta-sin(2*theta))/8 - pow(d1,4)*pow(tan(theta),3)/6)/pi/pow(delta*h,4);
          
          //rhs[i] = (0.5*c2-d_proj*c1)*(Ffun(x_0[i],y_0[i],s*Dt)+u0[i]/Dt)-c1*g_ptr[pdim] + (0.5*(c2-c3)-d_proj*c1)*g_ptr[pdim]/x0;
          rhs[i] = (0.5*c2-d_proj*c1)*(Ffun(x_0[i],y_0[i],s*Dt)+u0[i]/Dt)-c1*g_ptr[pdim];
          pdim = pdim+1;
      }
  }
 

  rhs[0] = u_exact(x_0[0],y_0[0],s*Dt);
  

  for(int i=0;i<N_point;i++){
    //cout<<rhs[i]<<endl;
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
  }
    
  vector< vector<int> > bdynei(kdim);
  for(int i=0;i<kdim;i++){
    for(int j=0;j<N_point;j++){
        if(pow(x_0[j]-circlebdy[2*i],2)+pow(y_0[j]-circlebdy[2*i+1],2) < delta*delta*h*h){
            bdynei[i].push_back(j);
        }
    }
  }
    
  for(int i=0;i<kdim;i++){
        double* P = new double[bdynei[i].size()*6];
        double* M = new double[6*6];
        double* D = new double[bdynei[i].size()]; 
        double* Beta = new double[bdynei[i].size()*6];
          
        normal[0] = circlebdy[2*i]/sqrt(circlebdy[2*i]*circlebdy[2*i]+circlebdy[2*i+1]*circlebdy[2*i+1]);
        normal[1] = circlebdy[2*i+1]/sqrt(circlebdy[2*i]*circlebdy[2*i]+circlebdy[2*i+1]*circlebdy[2*i+1]);
        tanvec[0] = -circlebdy[2*i+1]/sqrt(circlebdy[2*i]*circlebdy[2*i]+circlebdy[2*i+1]*circlebdy[2*i+1]);
        tanvec[1] = circlebdy[2*i]/sqrt(circlebdy[2*i]*circlebdy[2*i]+circlebdy[2*i+1]*circlebdy[2*i+1]);
        
        for(int s=0;s!=bdynei[i].size();s++){
            P[s*6] = 1;
            P[s*6+1] = (x_0[bdynei[i][s]]-circlebdy[2*i])*normal[0]+(y_0[bdynei[i][s]]-circlebdy[2*i+1])*normal[1];
            P[s*6+2] = (x_0[bdynei[i][s]]-circlebdy[2*i])*tanvec[0]+(y_0[bdynei[i][s]]-circlebdy[2*i+1])*tanvec[1];
            P[s*6+3] = pow((x_0[bdynei[i][s]]-circlebdy[2*i])*normal[0]+(y_0[bdynei[i][s]]-circlebdy[2*i+1])*normal[1],2);
            P[s*6+4] = pow((x_0[bdynei[i][s]]-circlebdy[2*i])*tanvec[0]+(y_0[bdynei[i][s]]-circlebdy[2*i+1])*tanvec[1],2);
            P[s*6+5] = (x_0[bdynei[i][s]]-circlebdy[2*i])*(y_0[bdynei[i][s]]-circlebdy[2*i+1]);
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
                  Beta[m*6+n] = 0;
                  for(int s=0;s<6;s++){
                      Beta[m*6+n] += D[m]*P[m*6+s]*M[s*6+n];
                  }
              }
          }
          
          double x1 = circlebdy[2*i]*(1+pow(circlebdy[2*i],2)+pow(circlebdy[2*i+1],2)-delta*h*delta*h)/2/(pow(circlebdy[2*i],2)+pow(circlebdy[2*i+1],2));
          double y1 = circlebdy[2*i+1]*(1+pow(circlebdy[2*i],2)+pow(circlebdy[2*i+1],2)-delta*h*delta*h)/2/(pow(circlebdy[2*i],2)+pow(circlebdy[2*i+1],2));
          double x0 = circlebdy[2*i]*normal[0] + circlebdy[2*i+1]*normal[1];

          double d1 =  sqrt(x1*x1+y1*y1);
          double dist = (x1-circlebdy[2*i])*normal[0] + (y1-circlebdy[2*i+1])*normal[1];
          double theta = acos(d1);
          double d_proj = 1-x0;

          double c1 = -16*(pow(sqrt(pow(delta*h,2)-pow(dist,2)),3))/3/pi/pow(delta*h,4) 
            +8*(2*sin(theta)/3-2*pow(d1,3)*tan(theta)/3-x0*(theta-d1*d1*tan(theta)))/pi/pow(delta*h,4);
          double c2 = (pi+2*dist*(2*dist*dist-delta*delta*h*h)*sqrt(delta*delta*h*h-dist*dist)/pow(delta*h,4)
                          +2*atan(dist/sqrt(delta*delta*h*h-dist*dist)))/pi
                    +8*((2*theta+sin(2*theta))/8 - 4*x0*sin(theta)/3+x0*x0*theta-(pow(d1,4)/2-4*x0*pow(d1,3)/3+x0*x0*d1*d1)*tan(theta))/pi/pow(delta*h,4);
        
          double c3 = (pi+2*dist*(5*delta*delta*h*h-2*dist*dist)*sqrt(delta*delta*h*h-dist*dist)/3/pow(delta*h,4)
                          +2*atan(dist/sqrt(delta*delta*h*h-dist*dist)))/pi + 8*((2*theta-sin(2*theta))/8 - pow(d1,4)*pow(tan(theta),3)/6)/pi/pow(delta*h,4);
        
          for(int s=0;s<bdynei[i].size();s++){
            v_ptr[i] += rhs[bdynei[i][s]]*Beta[s*6]; 
          }

    
  }  



  double error = 0;
  double errinf = 0 ;

  for(int i=0;i<N_point;i++){
    error += pow(u0[i]-u_exact(x_0[i],y_0[i],s*Dt),2); 
  }

  error /= N_point;
  error = sqrt(error);
  
  for(int i=0;i<N_point;i++){
    if(fabs(u0[i]-u_exact(x_0[i],y_0[i],s*Dt)) > errinf){
      errinf = fabs(u0[i]-u_exact(x_0[i],y_0[i],s*Dt));
    }
  } 

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


RADIUS = 1
ENLARGE = 1.2

N_REF = 4

c = Rectangle(Point(-2,-2),Point(2,2)) - Circle(Point(0,0),RADIUS)

mesh = generate_mesh(c,N_ELEM)

class Hole(SubDomain):
    def inside(self, x, on_boundary):
        r = sqrt(x[0]**2 + x[1]**2)
        return r < ENLARGE*RADIUS
    def snap(self, x):
        r = sqrt(x[0]**2 + x[1]**2)
        if(r < ENLARGE*RADIUS):
            x[0] = (RADIUS/r)*x[0]
            x[1] = (RADIUS/r)*x[1]


def py_snap_boundary(mesh, sub_domain):
    boundary = BoundaryMesh(mesh,"exterior")
    dim = mesh.geometry().dim()
    x = boundary.coordinates()
    for i in range(0,boundary.num_vertices()):
        sub_domain.snap(x[i,:])
    ALE.move(mesh,boundary)

#for i in range(0,N_REF):
    #mesh = refine(mesh)    
    #py_snap_boundary(mesh,Hole())
    #mesh.smooth(20)






n = FacetNormal(mesh)
V = FunctionSpace(mesh,"Lagrange",1)
V_vec = VectorFunctionSpace(mesh, 'DG',0)
u_D = Expression('t*sin(x[0])*cos(x[1])',degree = 2,t = 0,domain = mesh) 
#u_D = Expression('t*t+(x[0]*x[0]+x[1]*x[1])',degree = 2,t = 0,domain = mesh) 

GEOM_TOL = 1e3*DOLFIN_EPS

d = mesh.geometry().dim()
def getNodeCoordinates(V,nFields):
    dofxFlat = V.tabulate_dof_coordinates()
    dofx = dofxFlat.reshape((-1,d))
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
        return (not (near(x[0],-2) or near(x[0],2)
                     or near(x[1],-2) or near(x[1],2) or (x[0]*x[0]+x[1]*x[1])<RADIUS**2+GEOM_TOL))

class BdryOfInterest(SubDomain):
    def inside(self,x,on_boundary):
        return (x[0]*x[0]+x[1]*x[1] < RADIUS**2 + GEOM_TOL) and on_boundary 

class AntiBdry(SubDomain):
    def inside(self,x,on_boundary):
        return x[0]*x[0]+x[1]*x[1] > RADIUS**2 + GEOM_TOL


FLUX_BDRY = 1
COMPLEMENT_FLUX_BDRY = 2
boundaryMarkers = MeshFunction("size_t",mesh,mesh.topology().dim()-1,
                               COMPLEMENT_FLUX_BDRY)
BdryOfInterest().mark(boundaryMarkers,FLUX_BDRY)
ds_marked = ds(subdomain_data=boundaryMarkers)
Bdy = zeros((2*N_ELEM+1)*(2*N_ELEM+1)) 

k = 0
for i in range(nFluidNodes):
    if(fluidNodex[i,0]*fluidNodex[i,0] + fluidNodex[i,1]*fluidNodex[i,1] < RADIUS**2 + GEOM_TOL):
        Bdy[k] = i
        k = k+1
#        if(fluidNodex[i,0]*fluidNodex[i,0] + fluidNodex[i,1]*fluidNodex[i,1] >= RADIUS**2 + DOLFIN_EPS):
#            print(fluidNodex[i])
        

Bdycoord = zeros(2*k)

for i in range(0,k):
    Bdycoord[2*i] = fluidNodex[int(Bdy[i]),0]
    Bdycoord[2*i+1] = fluidNodex[int(Bdy[i]),1]


antiBCs = [DirichletBC(V,Constant(0.0),AntiBdry()),]
u_n = interpolate(u_D,V)

VscalarE = FiniteElement("Lagrange",mesh.ufl_cell(),1)
Vscalar = FunctionSpace(mesh,VscalarE)

ubc = Function(Vscalar)

bc = DirichletBC(V,ubc,"on_boundary")
BC = DirichletBC(V,ubc,AntiBdry())

u = TrialFunction(V)
v = TestFunction(V)

f = Expression('sin(x[0])*cos(x[1])+20*t*sin(x[0])*cos(x[1])',degree = 2, t = 0,domain = mesh)
#f = Expression('2*t-4',degree = 2, t = 0)

C_pen = Constant(1.0e1)
#f = Constant(-3)
#f = Expression('2*t*(x[0]*x[0]+x[1]*x[1])-4*t*t',t = 0, degree = 2, domain =mesh)

F = u*v*dx + 10*dt*dot(grad(u), grad(v))*dx - (u_n + dt*f)*v*dx #+ dt*(C_pen*N_ELEM)*inner(u-u_n,v)*ds(FLUX_BDRY)
a, L = lhs(F), rhs(F)

u_current = Function(V)
u_lastIter = Function(V)
def res(u,v):
    return (u*v/dt - u_n*v/dt + inner(grad(u),grad(v)) - f*v)*dx

qn = TrialFunction(V)
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

ext_module = compile_cpp_code(cpp_code, libs=["lapack"])
ext_module.Preprocess()
ext_module.getBdycoord(k,Bdycoord)

projcoord = zeros(8*N_ELEM*N_ELEM);
projdim = zeros(1);

ext_module.projBdy(projdim,projcoord);

#print(projcoord)



t = 0
vout = zeros(k)
gin = zeros(int(projdim[0]))
gin_last = zeros(int(projdim[0]))


for i in range(int(projdim[0])):
    #gin[i] = 2*(projcoord[2*i]*projcoord[2*i]+projcoord[2*i+1]*projcoord[2*i+1]) + Rbeta*(projcoord[2*i]*projcoord[2*i]+projcoord[2*i+1]*projcoord[2*i+1])
    #gin[i] = dt*(projcoord[2*i]*cos(projcoord[2*i])*cos(projcoord[2*i+1])-projcoord[2*i+1]*sin(projcoord[2*i])*sin(projcoord[2*i+1]))+Rbeta*dt*(sin(projcoord[2*i])*cos(projcoord[2*i+1]))
    #gin[i] = 2*projcoord[2*i]*projcoord[2*i]
    gin[i] = 0

SUB_TOTAL = 0;
tau1 = 1./2 
tau  = 0.1
Q_n = zeros(int(projdim[0]))
Q_last = zeros(int(projdim[0]))
Q_n1 = zeros(k)
Q_last1 = zeros(k)
taumin = -0.5
taumax = 0.95

MAX_SUBITERS = 100
SUBITER_TOL = 1e-8

SUB_TOTAL = 0
for timeStep in range(N_steps):
    t += dt
    u_D.t = t
    f.t = t
    print("------- Time step "+str(timeStep+1)
          +" , t = "+str(t)+"-------")
    for subIter in range(0,1):
        
        SUB_TOTAL = SUB_TOTAL + 1
        ext_module.BE(timeStep+1,k,vout,gin,1)

        ubc.assign(interpolate(u_D,V))
        uBCarray = ubc.vector().get_local()
        for i in range(k):
            uBCarray[int(Bdy[i])] = vout[i]

        ubc.vector().set_local(uBCarray)
        solve( a == L, u_current ,bc )


        
        #qn_h = solveFor_qn_h(FBdry, antiBCs)

        
        gradu = project(grad(u_current), V_vec)
        if(timeStep == 0):
            for i in range(int(projdim[0])):
                #gin[i] = -1*qn_h(projcoord[2*i],projcoord[2*i+1]) + Rbeta*(u_current(projcoord[2*i],projcoord[2*i+1]))
                #gin[i] = -1*qn_h(projcoord[2*i],projcoord[2*i+1]) + Rbeta*(u_current(projcoord[2*i],projcoord[2*i+1]))
                #gin[i] = (t)*(projcoord[2*i]*cos(projcoord[2*i])*cos(projcoord[2*i+1])-projcoord[2*i+1]*sin(projcoord[2*i])*sin(projcoord[2*i+1]))+Rbeta*(t)*(sin(projcoord[2*i])*cos(projcoord[2*i+1]))
                #gin[i] = (t+dt)*(projcoord[2*i]*cos(projcoord[2*i])*cos(projcoord[2*i+1])-projcoord[2*i+1]*sin(projcoord[2*i])*sin(projcoord[2*i+1]))+Rbeta*(u_current(projcoord[2*i],projcoord[2*i+1]))

                gin[i] = gradu(projcoord[2*i],projcoord[2*i+1])[0]*projcoord[2*i] + gradu(projcoord[2*i],projcoord[2*i+1])[1]*projcoord[2*i+1]+Rbeta*(u_current(projcoord[2*i],projcoord[2*i+1]))
                #gin[i] = gin[i] + Rbeta*(u_current(projcoord[2*i],projcoord[2*i+1]))
                #gin[i] = t*t*2*(projcoord[2*i]*projcoord[2*i]+projcoord[2*i+1]*projcoord[2*i+1]) + 1/N_ELEM
                #gin[i] = t*(projcoord[2*i]*cos(projcoord[2*i])*cos(projcoord[2*i+1])-projcoord[2*i+1]*sin(projcoord[2*i])*sin(projcoord[2*i+1]))
                #gin[i] = 2*projcoord[2*i]*projcoord[2*i]
                Q_n[i] = gin[i]
        else:
            for i in range(int(projdim[0])):
                gin[i] = gradu(projcoord[2*i],projcoord[2*i+1])[0]*projcoord[2*i] + gradu(projcoord[2*i],projcoord[2*i+1])[1]*projcoord[2*i+1]+Rbeta*(u_current(projcoord[2*i],projcoord[2*i+1]))
                #gin[i] = -1*qn_h(projcoord[2*i],projcoord[2*i+1]) + Rbeta*(2*u_current(projcoord[2*i],projcoord[2*i+1])-u_lastIter(projcoord[2*i],projcoord[2*i+1]))
                #gin[i] = (t)*(projcoord[2*i]*cos(projcoord[2*i])*cos(projcoord[2*i+1])-projcoord[2*i+1]*sin(projcoord[2*i])*sin(projcoord[2*i+1]))+Rbeta*(t)*(sin(projcoord[2*i])*cos(projcoord[2*i+1]))
                #gin[i] = (t+dt)*(projcoord[2*i]*cos(projcoord[2*i])*cos(projcoord[2*i+1])-projcoord[2*i+1]*sin(projcoord[2*i])*sin(projcoord[2*i+1]))+Rbeta*(2*u_current(projcoord[2*i],projcoord[2*i+1])-u_lastIter(projcoord[2*i],projcoord[2*i+1]))
                #gin[i] = 0.1*(-1*qn_h(projcoord[2*i],projcoord[2*i+1]) + Rbeta*(u_current(projcoord[2*i],projcoord[2*i+1])))+0.9*gin_last[i]
                #gin[i] = 0.1*(-1*qn_h(projcoord[2*i],projcoord[2*i+1]))+0.9*gin_last[i];
                #gin[i] = gin[i] + Rbeta*(u_current(projcoord[2*i],projcoord[2*i+1]))
                #gin[i] = t*t*2*(projcoord[2*i]*projcoord[2*i]+projcoord[2*i+1]*projcoord[2*i+1]) + 1/N_ELEM
                #gin[i] = t*(projcoord[2*i]*cos(projcoord[2*i])*cos(projcoord[2*i+1])-projcoord[2*i+1]*sin(projcoord[2*i])*sin(projcoord[2*i+1]))
                #gin[i] = 2*projcoord[2*i]*projcoord[2*i]
                Q_n[i] = gin[i] - gin_last[i]
        #gerror = 0
        
        #for i in range(int(projdim[0])):
           # if(abs(gin[i]-2*(projcoord[2*i]*projcoord[2*i]+projcoord[2*i+1]*projcoord[2*i+1]))>gerror):
           #     gerror = abs(gin[i]-2*(projcoord[2*i]*projcoord[2*i]+projcoord[2*i+1]*projcoord[2*i+1]))

        #print('gerror=%.3g' %(gerror))

            if(LA.norm(Q_last-Q_n) > 1E-10):
                tau = tau + (tau-1)*(np.dot(Q_last-Q_n,Q_n)/LA.norm(Q_last-Q_n)/LA.norm(Q_last-Q_n))
            if (tau < -0.5):
                tau = -0.5
            elif (tau > 0.95):
                tau = 0.95

            

            #for i in range(0,int(projdim[0])):
                #gin[i] = (1-tau)*gin[i] + tau*gin_last[i]


        for i in range(int(projdim[0])):
            #print(gin[i])
            gin_last[i] = gin[i]
            Q_last[i] = Q_n[i]
        
        #for i in range(k):
            #print(qn_h(fluidNodex[int(Bdy[i]),0],fluidNodex[int(Bdy[i]),1]))
        #print(qn_h.vector().get_local())
        #np.savetxt("flux",qn_h.vector().get_local())
        #print(fluidNodex[452])
        #print(fluidNodex[615])
        #print(fluidNodex[647])
        #print(fluidNodex[1639])



        diff = Function(V) 
        diff.assign(u_current - u_lastIter)
        
        u_lastIter.assign(u_current)

        normDiff = math.sqrt(assemble(((diff)**2)*dx))
        normLastIter = math.sqrt(assemble((u_lastIter**2)*dx))
        rel = normDiff/normLastIter

   
        print("Relative change: "+str(rel))

        if(rel < SUBITER_TOL):
            break

    #ext_module.BE(timeStep+1,k,vout,gin,1)
    u_n.assign(u_current)
    
    u_e = interpolate(u_D,V)

    error = np.abs(u_e.vector().get_local() - u_current.vector().get_local()).max()
    #error = errornorm(u_D,u_current,norm_type='linf',degree_rise=3,mesh = mesh)
    print('t = %.8f: L_inf error = %.4g' % (t, error))
    
    L2error = (u_current-u_e)**2*dx
    L2error = sqrt(abs(assemble(L2error)))
    #L2error = errornorm(u_D,u_current,norm_type='l2',degree_rise=3,mesh =mesh)
    print('t = %.8f: L_2 error = %.4g' % (t, L2error))

    #def fluxErr(qn):
        #err = inner(grad(u_D),n) - qn
        #return math.sqrt(assemble(err*err*ds_marked(FLUX_BDRY)))

    
    #print('t = %.8f: derror = %.3g' % (t, fluxErr(qn_h)))

SUB_TOTAL = SUB_TOTAL/N_steps
print('number of subiteraions: %.10f' %(SUB_TOTAL))
