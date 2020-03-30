#include<stdio.h>
#include <stdlib.h>
#include<math.h>

//cc -fPIC -shared -o solver.so solver.c

int f(double t,double y[2], double yp[2], double hh, double mass_bd[4], double p_grid[1800], double eos[1800][4])
{
  double rho;
  double G = 6.67408e-11;
  double PI = 3.14159265358979;

  yp[0] = -(G*t)*hh;
  yp[0] /= 4.0*PI*pow(y[1],4); // dp/dm
  
  if (t <= mass_bd[3] && t > mass_bd[2]) {
    for (int i = 0; i<1800; i++) {
      if (log10(y[0]) > p_grid[i] && log10(y[0]) <= p_grid[i+1]) {
	rho = eos[i][3];
	if (eos[i][3] != eos[i+1][3]) {
	  rho += (log10(y[0])-p_grid[i])*(eos[i+1][3]-eos[i][3])/(p_grid[i+1]-p_grid[i]);
	}
	//printf("first\n");
	rho = pow(10,rho);
	break;
      }
    }
  }
  else if (t <= mass_bd[2] && t > mass_bd[1]) {
    for (int i = 0; i<1800; i++) {
      if (log10(y[0]) > p_grid[i] && log10(y[0]) <= p_grid[i+1]) {
	rho = eos[i][2];
	if (eos[i][2] != eos[i+1][2]) {
	  rho += (log10(y[0])-p_grid[i])*(eos[i+1][2]-eos[i][2])/(p_grid[i+1]-p_grid[i]);
	}
	//printf("second\n");
	rho = pow(10,rho);
	//printf("%f\t%f\t%f\t%f\t%f\n",eos[i][2],eos[i+1][2],log10(y[0]),p_grid[i],p_grid[i+1]);
	break;
      }
    }
  }
  else if (t <= mass_bd[1] && t > mass_bd[0]) {
    for (int i = 0; i<1800; i++) {
      if (log10(y[0]) > p_grid[i] && log10(y[0]) <= p_grid[i+1]) {
	rho = eos[i][1];
	if (eos[i][1] != eos[i+1][1]) {
	  rho += (log10(y[0])-p_grid[i])*(eos[i+1][1]-eos[i][1])/(p_grid[i+1]-p_grid[i]);
	}
	//printf("third\n");
	rho = pow(10,rho);
	break;
      }
    }
  }
  else {
    for (int i = 0; i<1800; i++) {
      if (log10(y[0]) > p_grid[i] && log10(y[0]) <= p_grid[i+1]) {
	rho = eos[i][0];
	if (eos[i][0] != eos[i+1][0]) {
	  rho += (log10(y[0])-p_grid[i])*(eos[i+1][0]-eos[i][0])/(p_grid[i+1]-p_grid[i]);
	}
	//printf("last\n");
	//printf("%f\t%f\t%f\t%f\t%f\n",eos[i][2],eos[i+1][2],log10(y[0]),p_grid[i],p_grid[i+1]);
	rho = pow(10,rho);
	break;
      }
    }
  }

  yp[1] = hh/(4.0*PI*pow(y[1],2)*rho);

  //printf("YP1 %f\t%f\t%f\n",hh,y[1],rho);
  //printf("%f\t%f\t%f\n",t,yp[0],yp[1]);

  return 0;

}

double rk4(double y0[2], double t0, double t_end, double nsteps, double mass_bd[4], double p_grid[1800], double eos[1800][4])
{
  double k1[2],k2[2],k3[2],k4[2],k[2],y[2],y_k1[2],y_k2[2],y_k3[2];
  double t,h;
  int i;
    
  t = t0;
  //printf("%f\n",t);
  y[0] = y0[0]; y[1] = y0[1];
  //printf("%e15\t%f\n",t,y[1]);
  h = (t_end - t0)/nsteps;

  while(t>0.0) {
    f(t, y, k1, h, mass_bd, p_grid, eos);
    //printf("%f\t%f\n",k[0],k[1]);
    for (i=0;i<2;i++) {
      y_k1[i] = y[i]+k1[i]/2.0;
    }
    f((t+h/2.0), y_k1, k2, h, mass_bd, p_grid, eos);
    for (i=0;i<2;i++) {
      y_k2[i] = y[i]+k2[i]/2.0;
    }
    f((t+h/2.0), y_k2, k3, h, mass_bd, p_grid, eos);
    for (i=0;i<2;i++) {
      y_k3[i] = y[i]+k3[i];
    }
    f((t+h), y_k3, k4, h, mass_bd, p_grid, eos);

    for (i=0;i<2;i++) {
      k[i] = (k1[i]+2.0*k2[i]+2.0*k3[i]+k4[i])/6.0;
      y[i] += k[i];
    }
    
    t += h;
    //printf("%e\t%f\n",t,y[1]);
    
  }
  
  return y[1];
  
}
/*
int main(void)
{
  double Y0[2],M0,steps;
  double mb[4];
  Y0[0] = 1.0e+07;
  Y0[1] = 13060550.0;
  steps = 2.0e+02;
  double MEarth =  5.972e+24;
  M0 = 0.5*MEarth;
  //mb[0] = 0.1165*MEarth;mb[1] = 0.35*MEarth;mb[2] = 0.5*MEarth;mb[3] = 1.0*MEarth;
  mb[0] = 1.0*MEarth;mb[1] = 1.0*MEarth;mb[2] = 1.0*MEarth;mb[3] = 1.0*MEarth;
  double eos_data[1800][4];
  double pressure_grid[1800];
  
  //loading EOS data
  FILE *eos_file, *pgrid_file;
  eos_file = fopen("c_test_eos.dat","r");
  pgrid_file = fopen("c_test_pgrid.dat","r");

  for (int i = 0; i < 1800; i++){
    fscanf(pgrid_file, "%f", &pressure_grid[i]);
    for (int j = 0; j < 4; j++){
      fscanf(eos_file,"%f",&eos_data[i][j]);
    }
  }

  fclose(eos_file);fclose(pgrid_file);

  double yout = rk4(Y0, M0, 0.0, steps, mb, pressure_grid, eos_data);
  printf("%f\n",yout);
  
  return 0;
}
  
*/
