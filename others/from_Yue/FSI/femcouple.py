from dolfin import *
import ufl
from numpy import zeros
from scipy.spatial import KDTree
from numpy.linalg import norm as npNorm

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


    //		Type arrmultiply(const vvector<C>& c){
    //			Type m(0);
    //			for(size_t i=0;i!=(*this).size();++i){
    //				m+=(*this)[i]*c[i];
    //			}
    //			return m;
    //		}

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
const double Bulk = 4e4;
const double rho = 100.;
const double myB=5.;
const double myA=sqrt(4*Bulk/5./rho)*myB*pi;

double nv(double bond, double horizon, double dx){
  if(bond <= horizon && bond > horizon - dx) return -1.*bond/2./dx + (horizon/2./dx + 1./2);
  else if(bond <= horizon - dx) return 1.;
  else return 0;
}
//basis funtions
//
double u_exactx(double x, double y,double t){
  return t*cos(myB*pi*x)*sin(myB*pi*y);
  //return cos(myA*t)*cos(myB*pi*x)*sin(myB*pi*y);
  // return x*x;
}

double u_exacty(double x, double y,double t){
  return -t*sin(myB*pi*x)*cos(myB*pi*y);
  //  return -cos(myA*t)*sin(myB*pi*x)*cos(myB*pi*y);
  //return -2*x*y;
}

double du_exactx(double x, double y,double t){
  return cos(myB*pi*x)*sin(myB*pi*y);
  //  return -myA*sin(myA*t)*cos(myB*pi*x)*sin(myB*pi*y);
}

double du_exacty(double x, double y,double t){
  return -sin(myB*pi*x)*cos(myB*pi*y);
  //  return myA*sin(myA*t)*sin(myB*pi*x)*cos(myB*pi*y);
}

double F_x(double x,double y,double t){ 
  return -4*Bulk*myB*pi*myB*pi*t*cos(myB*pi*x)*sin(myB*pi*y)/5;
  //  return -4*Bulk*cos(x)*sin(y)/5;
  //return 6*Bulk/5;
}
double F_y(double x,double y,double t){
  return 0;
  //  return 4*Bulk*sin(x)*cos(y)/5;
  //return 0;
}

double G_x(double x, double y,double t){
  return -4.*Bulk*myB*pi*t*sin(myB*pi*x)*sin(myB*pi*y)/5.;
  //  return -4.*Bulk*myB*pi*cos(myA*t)*sin(myB*pi*x)*sin(myB*pi*y)/5.;
  //  return 0;
  //   return -2*sin(x)*sin(y);
  //  return -4.*Bulk*sin(x)*sin(y)/5.;
}

double G_y(double x,double y,double t){
  //  return -4.*Bulk*sin(x)*sin(y)/5.;
  return 0;
}


vvector<double> phi(int m,vvector<double> x){
  vvector<double> y(2);
  if(m == 0){
    y[0] = x[0];
    y[1] = 0;
    return y;
  }
  else if(m == 1){
    y[0] = 0;
    y[1] = x[0];
    return y;
  }
  else if(m == 2){
    y[0] = x[1];
    y[1] = 0;
    return y;
  }
  else if(m == 3){
    y[0] = 0;
    y[1] = x[1];
    return y;
  }
  else if(m == 4){
    y[0] = x[0]*x[0];
    y[1] = 0;
    return y;
  }
  else if(m == 5){
    y[0] = 0;
    y[1] = x[0]*x[0];
    return y;
  }
  else if(m == 6){
    y[0] = x[1]*x[1];
    y[1] = 0;
    return y;
  }
  else if(m == 7){
    y[0] = 0;
    y[1] = x[1]*x[1];
    return y;
  }
  else if(m == 8){
    y[0] = x[0]*x[1];
    y[1] = 0;
    return y;
  }
  else{
    y[0] = 0;
    y[1] = x[0]*x[1];
    return y;
  }
}

const int N=10;
const double delta=3;
double h = 1./N;            // lattice constant, all the nodes are evenly distributed 
int k = floor(delta);
int dim = 2*(N+1+2*k)*(N+1+k);
int basedim = 10;      //linear basis funtions
double* x0 = new double [dim];    //initial configuration 
double* u = new double [dim];   //displacement in x direction and y direction 
double* u0 = new double [dim];
double* du0 = new double [dim];
double* rhs = new double [dim];
double* stiffK = new double [dim*dim];
double* stiffKT = new double[dim*dim];
double *gin = new double [2*N+2];
double *vout = new double [2*N+2];

vector< vector<int> > nei((N+1)*(N+1)); //neighborlist
vector< vector<double> > weight((N+1)*(N+1)); //quadrature weights

double diff = 100;
double error = 0;
const double c = 48.*Bulk/5/pi/pow(delta*h,3);
const double cg = 24./pi/pow(delta*h,3);
double dpi = 0;
const int nstep = 5;//time-step 
const double Dt = 0.0001; // time-stepsize 

int* IPIV = new int[dim];
int info;
int nrhs = 1;
char C='T';
char D = 'A';

void Preprocess()
{
  int cenx,ceny,cenx1,ceny1;
  vvector<double> xi(2),zeta(2),gamma(2), eta(2);

  //set initial configuration 
  for(int i=0;i<N+1+2*k;i++){
    for(int j=0;j<N+1+k;j++){
      x0[i*(N+1+k)+j] = (j-k)*h;
      x0[i*(N+1+k)+j+(N+1+2*k)*(N+1+k)] = (i-k)*h;
    }
  }


  for(int i=0;i<N+1+2*k;i++){
    for(int j=0;j<N+1+k;j++){
      u0[i*(N+1+k)+j]=u_exactx(x0[i*(N+1+k)+j],x0[i*(N+1+k)+j+(N+1+k)*(N+1+2*k)],0.);
      u0[i*(N+1+k)+j+(N+1+k)*(N+1+2*k)]=u_exacty(x0[i*(N+1+k)+j],x0[i*(N+1+k)+j+(N+1+k)*(N+1+2*k)],0.);
      du0[i*(N+1+k)+j]=du_exactx(x0[i*(N+1+k)+j],x0[i*(N+1+k)+j+(N+1+k)*(N+1+2*k)],0.);
      du0[i*(N+1+k)+j+(N+1+k)*(N+1+2*k)]=du_exacty(x0[i*(N+1+k)+j],x0[i*(N+1+k)+j+(N+1+k)*(N+1+2*k)],0.);
    }
  }



  //set neighborhood list for every particle 
  for(int i=0;i<N+1;i++){
    for(int j=0;j<N+1;j++){
      for(int s1=-k;s1<k+1;s1++){
        for(int s2=-k;s2<k+1;s2++){
          if((i+k+s1)*(N+1+k)+j+k+s2 < (N+1+2*k)*(N+1+k) && (i+k+s1)*(N+1+k)+j+k+s2 > -1){
            xi[0] = x0[(i+s1+k)*(N+1+k)+j+k+s2] - x0[(i+k)*(N+1+k)+j+k];
            xi[1] = x0[(i+s1+k+N+1+2*k)*(N+1+k)+j+k+s2] - x0[(i+k+N+1+2*k)*(N+1+k)+j+k];
            if(xi.Norm2() < delta*h && !(s1 == 0 && s2 == 0)){
              nei[i*(N+1)+j].push_back((i+k+s1)*(N+1+k)+j+k+s2);
            }
          }
        }
      }       
    }
  }


  double* WORK = new double[1000];
  for(int i=0;i<N+1;i++){
    for(int j=0;j<N+1;j++){
      int s = nei[i*(N+1)+j].size();
      weight[i*(N+1)+j].resize(s,0);
      double* stiffweight = new double[(s+2*basedim)*(s+2*basedim)];
      double* rhsweight = new double[s+2*basedim];
      double* U = new double[(s+2*basedim)*(s+2*basedim)];
      double* VT = new double[(s+2*basedim)*(s+2*basedim)];
      double* S = new double[s+2*basedim];
      int INFO;
      int LWORK = 1000;
      int dim1 = s+2*basedim;
      for(int t=0;t<(s+2*basedim)*(s+2*basedim);t++){
        stiffweight[t] = 0;
      }
      for(int t=0;t<s+2*basedim;t++){
        rhsweight[t] = 0;
      }
      for(int t=0;t<s;t++){
        zeta[0] = x0[(i+k)*(N+1+k)+j+k];
        zeta[1] = x0[(i+k)*(N+1+k)+j+k+(N+1+k)*(N+1+2*k)];
        gamma[0] = x0[nei[i*(N+1)+j][t]];
        gamma[1] = x0[nei[i*(N+1)+j][t]+(N+1+k)*(N+1+2*k)];
        xi[0] = gamma[0] - zeta[0]; 
        xi[1] = gamma[1] - zeta[1];
        stiffweight[t*(s+2*basedim)+t] = 2;
        for(int m=0;m<basedim;m++){
          eta = phi(m,gamma) - phi(m,zeta);
          stiffweight[t*(s+2*basedim)+s+m] = (xi[0]*xi[0]*eta[0] + xi[0]*xi[1]*eta[1])/pow(xi.Norm2(),3);
          stiffweight[t*(s+2*basedim)+s+basedim+m] = (xi[0]*xi[1]*eta[0] + xi[1]*xi[1]*eta[1])/pow(xi.Norm2(),3);
          stiffweight[(s+m)*(s+2*basedim)+t] = stiffweight[t*(s+2*basedim)+s+m];
          stiffweight[(s+basedim+m)*(s+2*basedim)+t] = stiffweight[t*(s+2*basedim)+s+basedim+m];
        }

      }
      if(j < N-k){
        rhsweight[s+4] = pi*pow(delta*h,3)/4;
        rhsweight[s+6] = pi*pow(delta*h,3)/12;
        rhsweight[s+9] = pi*pow(delta*h,3)/12;
        rhsweight[s+10+5] = pi*pow(delta*h,3)/12;
        rhsweight[s+10+7] = pi*pow(delta*h,3)/4;
        rhsweight[s+10+8] = pi*pow(delta*h,3)/12;
      }
      else if( j<N ){
        double distance = 1 - x0[(i+k)*(N+1+k)+j+k];
        double angle = acos(distance/delta/h);
        rhsweight[s] = pow(distance,2)*sin(angle) + pow(delta*h,2)*(pow(sin(angle),3)/3-sin(angle)); 
        rhsweight[s+3] = pow(distance,2)*(-2*sin(angle)+2*log((sin(angle/2)+cos(angle/2))/(cos(angle/2)-sin(angle/2))))/2 
          - pow(delta*h,2)*pow(sin(angle),3)/3;
        rhsweight[s+4] = 2*x0[(i+k)*(N+1+k)+j+k] * rhsweight[s] + pow(delta*h,3)*(12*(pi-angle)-8*sin(2*angle)-sin(4*angle))/48 + 2*pow(distance,3)*sin(angle)/3;
        rhsweight[s+6] = pow(delta*h,3)*(pi-angle+sin(4*angle))/12 
          +pow(distance,3)*(-2*sin(angle)+2*log((sin(angle/2)+cos(angle/2))/(cos(angle/2)-sin(angle/2))))/3;
        rhsweight[s+7] = 2*x0[(i+k)*(N+1+k)+j+k+(N+1+k)*(N+1+2*k)]*rhsweight[s+3]; 
        rhsweight[s+8] = x0[(i+k)*(N+1+k)+j+k+(N+1+k)*(N+1+2*k)]*rhsweight[s];
        rhsweight[s+9] = x0[(i+k)*(N+1+k)+j+k]*rhsweight[s+3]+pow(delta*h,3)*(pi-angle+sin(4*angle))/12
          +pow(distance,3)*(-2*sin(angle)+2*log((sin(angle/2)+cos(angle/2))/(cos(angle/2)-sin(angle/2))))/3; 


        rhsweight[s+10+1] = rhsweight[s+3];
        rhsweight[s+10+2] = rhsweight[s+3];
        rhsweight[s+10+5] = 2*x0[(i+k)*(N+1+k)+j+k]*rhsweight[s+10+1] + pow(delta*h,3)*(pi-angle+sin(4*angle))/12
          +pow(distance,3)*(-2*sin(angle)+2*log((sin(angle/2)+cos(angle/2))/(cos(angle/2)-sin(angle/2))))/3; 
        rhsweight[s+10+6] = 2*x0[(i+k)*(N+1+k)+j+k+(N+1+k)*(N+1+2*k)]*rhsweight[s+10+1];
        rhsweight[s+10+7] = pow(delta*h,3)*(12*(pi-angle)+8*sin(2*angle)-sin(4*angle))/48
          +pow(distance,3)*(4*sin(angle)+1/pow(cos(angle/2)-sin(angle/2),2)-1/pow(cos(angle/2)+sin(angle/2),2)
              +6*log((cos(angle/2)-sin(angle/2))/(cos(angle/2)+sin(angle/2))))/6;
        rhsweight[s+10+8] = x0[(i+k)*(N+1+k)+j+k]*rhsweight[s+10+2] + pow(delta*h,3)*(pi-angle+sin(4*angle))/12
          +pow(distance,3)*(-2*sin(angle)+2*log((sin(angle/2)+cos(angle/2))/(cos(angle/2)-sin(angle/2))))/3; 
        rhsweight[s+10+9] = x0[(i+k)*(N+1+k)+j+k+(N+1+k)*(N+1+2*k)]*rhsweight[s+10+1];
      }
      else{     //j == N
        rhsweight[s] = -2*pow(delta*h,2)/3;
        rhsweight[s+3] = -1*pow(delta*h,2)/3;
        rhsweight[s+4] = 2*x0[(i+k)*(N+1+k)+j+k]*rhsweight[s] +  pi*pow(delta*h,3)/8;
        rhsweight[s+6] = pi*pow(delta*h,3)/24;
        rhsweight[s+7] = 2*x0[(i+k)*(N+k+1)+j+k+(N+k+1)*(N+2*k+1)]*rhsweight[s+3];
        rhsweight[s+8] = x0[(i+k)*(N+k+1)+j+k+(N+k+1)*(N+2*k+1)]*rhsweight[s];
        rhsweight[s+9] = x0[(i+k)*(N+k+1)+j+k]*rhsweight[s+3] +pi*pow(delta*h,3)/24;

        rhsweight[s+10+1] = -1*pow(delta*h,2)/3;
        rhsweight[s+10+2] = -1*pow(delta*h,2)/3;
        rhsweight[s+10+5] = 2*x0[(i+k)*(N+1+k)+j+k]*rhsweight[s+10+1] + pi*pow(delta*h,3)/24;
        rhsweight[s+10+6] = 2*x0[(i+k)*(N+1+k)+j+k+(N+1+k)*(N+1+2*k)]*rhsweight[s+10+1];
        rhsweight[s+10+7] = pi*pow(delta*h,3)/8;
        rhsweight[s+10+8] = pi*pow(delta*h,3)/24;
        rhsweight[s+10+9] =  x0[(i+k)*(N+k+1)+j+k+(N+k+1)*(N+2*k+1)]*rhsweight[s+10+1];

      }
      dgesvd_(&D,&D,&dim1,&dim1,stiffweight,&dim1,S,U,&dim1,VT,&dim1,WORK,&LWORK,&INFO);
      for(int m=0;m<dim1;m++){
        if(S[m] < 1e-6) S[m] = 0;
      }
      for(int m=0;m<dim1;m++){
        if(S[m] != 0) S[m] = 1./S[m];
      }


      for(int t=0;t<s;t++){
        for(int l=0;l<dim1;l++){
          for(int m=0;m<dim1;m++){
            weight[i*(N+1)+j][t] += VT[t*dim1+l]*S[l]*U[l*dim1+m]*rhsweight[m];
          }
        }
      }
      delete[] stiffweight;
      delete[] rhsweight;
      delete[] U;
      delete[] S;
      delete[] VT;

    }
  }

  delete[] WORK;
  //    for(int i=0;i<N+1;i++){
  //       for(int j=N;j<N+1;j++){
  //            for(int t=0;t<weight[i*(N+1)+j].size();t++){
  //                cout<<weight[i*(N+1)+j][t]<<" ";
  //            }
  //        }
  //        cout<<endl;
  //    }

  for(int i=0;i<dim;i++){
    for(int j=0;j<dim;j++){
      stiffK[i*dim+j] = 0;
    }
    rhs[i] = 0; 
  }


  for(int i=0;i<N+1;i++){
    for(int j=0;j<N+1;j++){
      cenx = (i+k)*(N+1+k)+j+k;
      ceny = (i+k+N+1+2*k)*(N+1+k)+j+k;

      for(int s=0;s<nei[i*(N+1)+j].size();s++){
        xi[0] = x0[nei[i*(N+1)+j][s]] - x0[cenx];
        xi[1] = x0[nei[i*(N+1)+j][s]+(N+1+k)*(N+1+2*k)] - x0[ceny];

        stiffK[cenx*dim+cenx] += -c*xi[0]*xi[0]*weight[i*(N+1)+j][s]/pow(xi.Norm2(),3);
        stiffK[cenx*dim+nei[i*(N+1)+j][s]] += c*xi[0]*xi[0]*weight[i*(N+1)+j][s]/pow(xi.Norm2(),3);
        stiffK[cenx*dim+ceny] += -c*xi[0]*xi[1]*weight[i*(N+1)+j][s]/pow(xi.Norm2(),3);
        stiffK[cenx*dim+nei[i*(N+1)+j][s]+dim/2] += c*xi[0]*xi[1]*weight[i*(N+1)+j][s]/pow(xi.Norm2(),3);

        stiffK[ceny*dim+ceny] += -c*xi[1]*xi[1]*weight[i*(N+1)+j][s]/pow(xi.Norm2(),3);
        stiffK[ceny*dim+nei[i*(N+1)+j][s]+dim/2] += c*xi[1]*xi[1]*weight[i*(N+1)+j][s]/pow(xi.Norm2(),3);
        stiffK[ceny*dim+cenx] += -c*xi[0]*xi[1]*weight[i*(N+1)+j][s]/pow(xi.Norm2(),3);
        stiffK[ceny*dim+nei[i*(N+1)+j][s]] += c*xi[0]*xi[1]*weight[i*(N+1)+j][s]/pow(xi.Norm2(),3);
      }
      //rhs[ceny] = 36*Bulk/5*pow(x[cenx],2);//+9*Bulk*pow(delta*h,2)/25;
      //rhs[cenx] = 18*Bulk/5;
      //rhs[ceny] = 18*Bulk/5;
      // if(j<N && j >= N-k){
      //   double distance = 1 - x[cenx];
      //   double  angle = acos(distance/delta/h);
      // }
      // if(j==N){


      //rhs[cenx] = 36*Bulk/5*pow(x[cenx],2) - c*pow(delta*h,2)*4/3;
      // }

    }

  }

  for(int i=0;i<dim;i++){
    stiffK[i*dim+i] += rho/Dt/Dt;
  }


  //Modify the stiff matrix, apply Dirichlet BC
  for(int i=0;i<N+1+2*k;i++){
    if(i < k || i > N+k){
      for(int j=0;j<N+1+k;j++){
        cenx = i*(N+1+k)+j;
        ceny = (i+N+1+2*k)*(N+1+k)+j;
        stiffK[cenx*dim+cenx] = 1;
        stiffK[ceny*dim+ceny] = 1;
      }
    } 
    else{
      for(int j=0;j<k;j++){
        cenx = i*(N+1+k)+j;
        ceny = (i+N+1+2*k)*(N+1+k)+j;
        stiffK[cenx*dim+cenx] = 1;
        stiffK[ceny*dim+ceny] = 1;
      }
    }
  }
}

void BE(int s,npArray g, npArray v, int UPDATE)
{
  int cenx,ceny,cenx1,ceny1;

  auto g_buffer = g.request();
  double *g_ptr = (double *)g_buffer.ptr;
  auto v_buffer = v.request();
  double *v_ptr = (double *)v_buffer.ptr;

  //apply boundary condition
  for(int i=0;i<N+1;i++){
    for(int j=0;j<N+1;j++){
      cenx = (i+k)*(N+1+k)+j+k;
      ceny = (i+k+N+1+2*k)*(N+1+k)+j+k;
      rhs[cenx] = F_x(x0[cenx],x0[ceny],s*Dt);
      rhs[ceny] = F_y(x0[cenx],x0[ceny],s*Dt);
      if(j < N && j>= N-k){
        double distance = 1- x0[cenx];
        double angle = acos(distance/delta/h);

        rhs[cenx] += 
          - (cg*(pow(delta*h,2)*(sin(angle)-pow(sin(angle),3)/3)-pow(distance,2)*sin(angle))
              - cg*(pow(delta*h,2)*pow(sin(angle),3)/3-pow(distance,2)*(-2*sin(angle)+2*log((sin(angle/2)+cos(angle/2))/(cos(angle/2)-sin(angle/2))))/2))
          *g_ptr[i]/2;

        rhs[ceny] += 
          - cg*(pow(delta*h,2)*pow(sin(angle),3)/3-pow(distance,2)*(-2*sin(angle)+2*log((sin(angle/2)+cos(angle/2))/(cos(angle/2)-sin(angle/2))))/2)
          *g_ptr[i+N+1];
      }
      else if(j == N){
        rhs[cenx] += -(cg*2*pow(delta*h,2)/3 - cg*pow(delta*h,2)/3)*g_ptr[i]/2;
        rhs[ceny] += -cg*pow(delta*h,2)/3*g_ptr[i+N+1];
      }
    }
  }


  //Dirichlet boundary condition 
  for(int i=0;i<N+1+2*k;i++){
    if(i < k || i > N+k){
      for(int j=0;j<N+1+k;j++){
        rhs[i*(N+1+k)+j] = u_exactx(x0[i*(N+1+k)+j],x0[i*(N+1+k)+j+(N+1+k)*(N+1+2*k)],s*Dt);
        rhs[(i+N+1+2*k)*(N+1+k)+j] =u_exacty(x0[i*(N+1+k)+j],x0[i*(N+1+k)+j+(N+1+k)*(N+1+2*k)],s*Dt);
      }
    }
    else{
      for(int j=0;j<k;j++){
        rhs[i*(N+1+k)+j] = u_exactx(x0[i*(N+1+k)+j],x0[i*(N+1+k)+j+(N+1+k)*(N+1+2*k)],s*Dt);
        rhs[(i+N+1+2*k)*(N+1+k)+j] =u_exacty(x0[i*(N+1+k)+j],x0[i*(N+1+k)+j+(N+1+k)*(N+1+2*k)],s*Dt);
      }
    }
  }

  for(int i=0;i<N+1;i++){
    for(int j=0;j<N+1;j++){
      cenx = (i+k)*(N+1+k)+j+k;
      ceny = (i+k+N+1+2*k)*(N+1+k)+j+k;
      rhs[cenx] += rho*(u0[cenx]/Dt/Dt + du0[cenx]/Dt);
      rhs[ceny] += rho*(u0[ceny]/Dt/Dt + du0[ceny]/Dt);
    }
  }

  for(int i=0;i<dim;i++){
    for(int j=0;j<dim;j++){
      stiffKT[i*dim+j] = stiffK[i*dim+j];
    }
  }


  dgetrf_(&dim,&dim,stiffKT,&dim,IPIV,&info);
  dgetrs_(&C,&dim,&nrhs,stiffKT,&dim, IPIV,rhs,&dim,&info);

  for(int i=0;i<N+1;i++){
    for(int j=0;j<N+1;j++){
        cenx = (i+k)*(N+1+k)+j+k;
        ceny = (i+k)*(N+1+k)+j+k+(N+1+k)*(N+1+2*k);
        du0[cenx] = (rhs[cenx]-u0[cenx])/Dt;
        du0[ceny] = (rhs[ceny]-u0[ceny])/Dt;
    }
  }

  if(UPDATE)
  {
    for(int i=0;i<dim;i++){
      u0[i] = rhs[i];
    }
  }

  for(int i=0;i<N+1;i++){
    int j=N;
    int cenx = (i+k)*(N+1+k)+j+k;
    int ceny = (i+k+N+1+2*k)*(N+1+k)+j+k;
    v_ptr[i] = du0[cenx];
    v_ptr[i+N+1] = du0[ceny];
  }

}




PYBIND11_MODULE(SIGNATURE, m)
{
    m.def("Preprocess",&Preprocess,"");
    m.def("BE",&BE,"");
}
"""


# Number of FE/peridynamic nodes across the width of the domain

N_ELEM = 10
N_NODES = N_ELEM+1

# fluid mesh
mesh = UnitSquareMesh(N_ELEM,N_ELEM)
# space dimension
d = mesh.geometry().dim()
# Velocity and pressure (equal-order interpolation)
dofsPerNode = d+1
VscalarE = FiniteElement("Lagrange",mesh.ufl_cell(),1)
VupE = MixedElement(dofsPerNode*[VscalarE,])
VuE = MixedElement(d*[VscalarE,])
Vup = FunctionSpace(mesh,VupE)
Vscalar = FunctionSpace(mesh,VscalarE)
Vu = FunctionSpace(mesh,VuE)
def uFromDofs(dofs):
    return as_vector([dofs[0],dofs[1]])
def pFromDofs(dofs):
    return dofs[dofsPerNode-1]

# (reverse engineered)
def nodeToDof(node,dofsPerNode,field):
    return dofsPerNode*node + field
def dofToNode(dof,dofsPerNode):
    return dof//dofsPerNode

# Get array of node coordinates and number of nodes from equal-order
# mixed function space V with nFields DoFs per node
def getNodeCoordinates(V,nFields):
    dofxFlat = V.tabulate_dof_coordinates()
    dofx = dofxFlat.reshape((-1,d))
    nNodes = dofx.shape[0]//nFields
    nodex = zeros((nNodes,d))
    for i in range(0,nNodes):
        nodex[i,:] = dofx[nodeToDof(i,nFields,0),:]
    return nodex, nNodes

###############
#test by yyu
ext_module = compile_cpp_code(cpp_code, libs=["lapack"])
ext_module.Preprocess()

gin = zeros(2*N_NODES,dtype="float64")
vout = zeros(2*N_NODES,dtype="float64")
################
    
# Dummy solid subproblem
solidMesh = UnitSquareMesh(N_ELEM,N_ELEM)
# Shift solid mesh up by 1
solidx = solidMesh.coordinates()
for i in range(0,len(solidx)):
    solidx[i,1] += 1.0
# Displacement function space
VyE = MixedElement(d*[VscalarE,])
Vy = FunctionSpace(solidMesh,VyE)

# Get coordinates of fluid nodes
fluidNodex, nFluidNodes = getNodeCoordinates(Vup,dofsPerNode)

# Get coordinates of solid nodes
solidNodex, nSolidNodes = getNodeCoordinates(Vy,d)

# Get set of boundary nodes
fsBdryNodesFluid = zeros(N_NODES,dtype='int32')
fsBdryNodesSolid = zeros(N_NODES,dtype='int32')

fluidCounter = 0
solidCounter = 0
for i in range(0,nFluidNodes):
    if(near(fluidNodex[i,1],1.0)):
        fsBdryNodesFluid[fluidCounter] = i
        fluidCounter += 1
    if(near(solidNodex[i,1],1.0)):
        fsBdryNodesSolid[solidCounter] = i
        solidCounter += 1

# Create mappings between fluid and solid interface nodes (brute force).
# Mappings are based on interface indices, not node indices in mesh, and
# must be composed with fsBdryNodes* to retrieve mesh node indices.
fluidBdryToSolidBdry = zeros(N_NODES,dtype='int32')
solidBdryToFluidBdry = zeros(N_NODES,dtype='int32')
for i in range(0,N_NODES):
    xFluid = fluidNodex[fsBdryNodesFluid[i],:]
    for j in range(0,N_NODES):
        xSolid = solidNodex[fsBdryNodesSolid[j],:]
        if(npNorm(xFluid-xSolid) < 1e3*DOLFIN_EPS):
            fluidBdryToSolidBdry[i] = j
            solidBdryToFluidBdry[j] = i
        
# Stokes flow
up = TrialFunction(Vup)

u = uFromDofs(up)
p = pFromDofs(up)

vq = TestFunction(Vup)
v = uFromDofs(vq)
q = pFromDofs(vq)

up_old = Function(Vup)
u_old = uFromDofs(up_old)

MU_F = Constant(1.0)
RHO_F = Constant(1.0)
DELTA_T = Constant(0.01)

f = Constant((1.0,0.0))

def eps(u):
    return 0.5*(grad(u) + grad(u).T)
I = Identity(d)
def sigma_f(u,p):
    return -p*I + 2.0*MU_F*eps(u)

def res_PSPG(u,p,v,q):
    
    #DuDt = (u-u_old)/DELTA_T + dot(u,nabla_grad(u))
    DuDt = (u-u_old)/DELTA_T # (ignore convection for Stokes)
    ii,jj = ufl.indices(2)
    r_M = RHO_F*DuDt-as_tensor(grad(sigma_f(u,p))[ii,jj,jj],(ii,))
    r_C = RHO_F*div(u)

    C_t = 4.0
    C_I = 36.0
    dx_dxi = ufl.Jacobian(mesh)
    dxi_dx = inv(dx_dxi)
    G = dxi_dx.T*dxi_dx
    nu_f = MU_F/RHO_F
    #tau_M = 1.0/sqrt(C_t/DELTA_T**2 + inner(u,G*u) + C_I*nu_f*nu_f*inner(G,G))
    tau_M = 1.0/sqrt(C_t/DELTA_T**2 + C_I*nu_f*nu_f*inner(G,G)) 
    tau_C = 1.0/(tau_M*tr(G))

    # Ansatz for fine scale velocity and pressure:
    uPrime = -tau_M*r_M
    pPrime = -tau_C*r_C
    # Residual of stabilized PSPG formulation:
    res = RHO_F*inner(DuDt,v)*dx + inner(sigma_f(u,p),grad(v))*dx \
          - dot(f,v)*dx \
          + inner(div(u),q)*dx \
          - inner(grad(q)/RHO_F, uPrime)*dx \
          - inner(pPrime,div(v))*dx

    return res

# Functions that will provide Dirichlet BC for fluid velocity components:
u0BC = Function(Vscalar)
u1BC = Function(Vscalar)
uDirichlet0 = DirichletBC(Vup.sub(0),u0BC,"on_boundary")
uDirichlet1 = DirichletBC(Vup.sub(1),u1BC,"on_boundary")

# For solving for nodal fluxes in fluid subdomain.  
antiBC = DirichletBC(Vu,Constant(d*(0.0,)),"x[1]<1.0-DOLFIN_EPS")

fluidBCs = [uDirichlet0,uDirichlet1]

MAX_SUBITERS = 1
N_TIMESTEPS = 1

# Current velocity/pressure
up_current = Function(Vup)
u_current = uFromDofs(up_current)
p_current = pFromDofs(up_current)

# Fluid subproblem residual
resFluid = res_PSPG(u,p,v,q)

# Nodal traction in current solution
h_current = Function(Vu)

# Flux extraction residual
h = TrialFunction(Vu)
dh = TestFunction(Vu)
qZero = Function(Vscalar)
resFlux = res_PSPG(u_current,p_current,dh,qZero) - dot(h,dh)*ds

# Files for output
u0File = File("results/ux.pvd")
u1File = File("results/uy.pvd")
pFile = File("results/p.pvd")
hFile = File("results/h.pvd")

for timeStep in range(0,N_TIMESTEPS):

    for subIter in range(0,MAX_SUBITERS):

        # TODO: solve dummy solid subproblem
        ext_module.BE(timeStep,gin,vout,0)

        # Solid velocity at bdry nodes, indexed by solid
        # bdry node
        ydotBdry0 = zeros(N_NODES)
        ydotBdry1 = zeros(N_NODES)

        # TODO: fill in the above arrays based on solution to dummy problem
        
        # test lid-driven cavity
        for i in range(0,N_NODES):
            ydotBdry0[i] = vout[i]
            ydotBdry1[i] = vout[i+N_NODES]
        
        # Convert to fluid boundary indexing
        uBdry0 = zeros(N_NODES)        
        uBdry1 = zeros(N_NODES)
        for i in range(0,N_NODES):
            uBdry0[i] = ydotBdry0[fluidBdryToSolidBdry[i]]
            uBdry1[i] = ydotBdry1[fluidBdryToSolidBdry[i]]

        # Write into correct vector elements of u*BC functions for
        # boundary conditions on velocity components
        u0BCarray = u0BC.vector().get_local()
        u1BCarray = u1BC.vector().get_local()
        for i in range(0,N_NODES):
            # 0th (only) field of a scalar space with 1 dof per node
            dofi = nodeToDof(fsBdryNodesFluid[i],1,0)
            u0BCarray[dofi] = uBdry0[i]
            u1BCarray[dofi] = uBdry1[i]
        u0BC.vector().set_local(u0BCarray)
        u1BC.vector().set_local(u1BCarray)

        # Solve fluid subproblem
        solve(lhs(resFluid)==rhs(resFluid),up_current,fluidBCs)

        # Solve for nodal flux on fluid--solid interface.
        # (Cannot use solve() directly on variational problem because of
        # zeros on diagonal.)
        #solve(lhs(resFlux)==rhs(resFlux),h_current,antiBC)
        ABdry = assemble(lhs(resFlux),keep_diagonal=True)
        bBdry = assemble(rhs(resFlux))
        antiBC.apply(ABdry,bBdry)
        solve(ABdry,h_current.vector(),bBdry)

        # arrays indexed by solid bdry node for traction components
        h0ArrayBdry = zeros(N_NODES)
        h1ArrayBdry = zeros(N_NODES)

        # Transfer data from h_current into the solid bdry node arrays:
        hArray = h_current.vector().get_local()
        # i in this loop is an index of a fluid bdry node
        for i in range(0,N_NODES):
            # DoFs in Vu for 0 and 1 components of h_current at fluid
            # boundary node i
            fluidDoFi0 = nodeToDof(fsBdryNodesFluid[i],2,0)
            fluidDoFi1 = nodeToDof(fsBdryNodesFluid[i],2,1)
            h0ArrayBdry[fluidBdryToSolidBdry[i]] = hArray[fluidDoFi0]
            h1ArrayBdry[fluidBdryToSolidBdry[i]] = hArray[fluidDoFi1]

    ext_module.BE(timeStep,gin,vout,1)

    u0out,u1out,pout = up_current.split()
    u0File << u0out
    u1File << u1out
    pFile << pout
    hFile << h_current
    
    up_old.assign(up_current)
