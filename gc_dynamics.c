/*******************************************************************************
 *
 *  FILE:         gc_dynamics.c
 *
 *  DATE:         14 June 2019
 *
 *  AUTHORS:      Louis Kang, University of Pennsylvania
 *                Vijay Balasubramanian, University of Pennsylvania
 *
 *                Based on code by Yoram Burak and Ila Fiete
 *
 *  LICENSING:    CC BY 4.0
 *
 *  REFERENCE:    Kang L, Balasubramanian V. A geometric attractor mechanism for
 *                self-organization of entorhinal grid modules. eLife 8, e46687
 *                (2019). doi:10.7554/eLife.46687
 *
 *  PURPOSE:      Simulation of multiple continuous attractor grid networks
 *                connected by excitatory coupling.
 *
 *  DEPENDENCIES: FFTW3 single-precision (http://www.fftw.org/)
 *                ziggurat random number generation package by John Burkardt
 *           (https://people.sc.fsu.edu/~jburkardt/c_src/ziggurat/ziggurat.html)
 *
 *******************************************************************************
 *
 *
 *  This is the source code for simulating a hierarchy of grid modules as
 *  multiple continuous attractor networks connected by excitatory coupling. It
 *  begins with a few "flow" phases during which the grid activity patterns are
 *  self-organized and refined. Then the main simulation begins during which
 *  population activity and single neuron recordings are output. See the
 *  reference for full details of the simulation and interpretation of its
 *  results.
 *
 *  Sample usage for the compiled executable file gc_dynamics and trajectory
 *  file trajectory.dat that generates coupled simulations presented in Figures
 *  2 and 3 of the reference:
 *    gc_dynamics -fileroot /path/to/coupled \
 *      -tsim 500000 -tdump_pop -1 -tdump_sn -1 -tdump_t 20 \
 *      -np 192 -h 12 \
 *      -tflow1 5000 -tflow2 50000 \
 *      -wmag 2.4 -lmin 4. -lmax 15. -lexp -1. \
 *      -umag 2.6 -urad 8. -u_vd \
 *      -falloff 4. \
 *      -vgain 3. -wshift 1 \
 *      -traj_in /path/to/trajectory.dat -traj_dt_fac 2 \
 *      -posx_init 145 -posy_init 60 \
 *      -nsn 3
 *
 *  This command will output the following:
 *    - coupled_a.dat:          broad input profile
 *    - coupled_l.dat:          inhibition distance profile
 *    - coupled_w.dat:          inhibitory kernel for network 1
 *    - coupled_u.dat:          coupling kernel
 *    - coupled_p-m1.dat:       population activity during simulation setup and
 *                              "flow" (likewise for m2--m5)
 *    - coupled_p-0000000.dat:  population activity at the start of the main
 *                              simulation
 *    - coupled_p-0500000.dat:  population activity at the end of the main
 *                              simulation
 *    - coupled_sn-0500000.dat: all single neuron rate maps over the entire
 *                              simulation
 *    - coupled_sn-legend.dat:  network locations for each recorded neuron
 *    - coupled_t.dat:          tracked positions of an activity bump on the
 *                              neural sheet
 *    - coupled_traj.dat:       positions of the simulated animal in space
 *    - coupled_visits.dat:     counts of timesteps that the animal spent at
 *                              each location
 *
 ******************************************************************************/


#include <complex.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>

#include "fftw3.h"
#include "ziggurat.h"

//==============================================================================
// Parameters and default value declarations
//==============================================================================

#define PI 3.141592654
#define UNDEF -999

// Variables with an asterisk can be set as command line arguments with flags.
// See Reading in parameters section for syntax.

// Random ----------------------------------------------------------------------
uint32_t randseed = 0;  //* 0: use time as random seed
float r4fn[128];        //  for ziggurat package
uint32_t r4kn[128];     //  for ziggurat package
float r4wn[128];        //  for ziggurat package
float rinit = 0.001;    //* Magnitude of initial firing random rates
float rnoise = 0.;      //* Magnitude of input noise added to each neuron

// Iterations and time ---------------------------------------------------------
int tsim = 100000;    //* Main simulation timesteps
int tscreen = 100;    //* Print to screen every tscreen timesteps
int tflow0 = 500;     //* Flow population activity with no velocity
int tflow1 = 2000;    //* Flow population activity with constant velocity
int tflow2 = 10000;   //* Flow population activity with trajectory
float tau = 10.;      //* Membrane time constant in ms

// Size ------------------------------------------------------------------------
int n = UNDEF;        //* Number of neurons per side; generally keep undefined
int np = 128;         //* Size of fft arrays with padding: choose number with
                      //    small prime factors for speed
int pad;              //  Size of padding on each side
int n2, np2, npfft;   //  Pre-calculated quantities
int h = 12;           //* Number of networks

// Recurrent inhibition --------------------------------------------------------
float wmag = 4.;      //* Inhibition strength
float lmin = 4.;      //* Inhibition distance minimum
float lmax = 8.;      //* Inhibition distance maximum
float lexp = 0.;      //* Inhibition distance exponent; 1 corresponds to linear
                      //    dependence and more negative means more concave
int wshift = 1;       //* Directional shift for each population
float vgain = 4.;     //* Velocity gain

// Excitatory coupling ---------------------------------------------------------
float umag = 0.4;   //* Coupling strength
float urad = 0.;    //* Coupling spread
int u_dv = 0;       //* Dorsal-to-ventral coupling
int u_vd = 1;       //* Ventral-to-dorsal coupling

// Broad input -----------------------------------------------------------------
float amag = 1.;            //* Broad input strength
float falloff = 4.;         //* Broad input falloff
float falloff_low = 0.;     //* Inside this scaled radius, the input is amag
float falloff_high = 1.;    //* Outside this scaled radius, the input is 0

// Enclosure and trajectory ----------------------------------------------------
int d = 180;              //* Size of enclosure
int posx_init = 145;      //* Initial x position
int posy_init = 60;       //* Initial y position
              
char traj_in[256] = "";   //* Trajectory filename
FILE *traj_vfile;         //  Trajectory file pointer
int traj_repeat = 0;      //* If 1, loop trajectory
float dt = 0.5;           //* Timestep corresonding to each trajectory entry
int traj_dt_fac = 1;      //* Factor by which simulation timestep differs from
                          //    trajectory timestep. Negative values correspond
                          //    to fractions (for example, -5 means simulation
                          //    timestep is one-fifth of trajectory timestep).
int traj_dt_count = 0;    //  Counter for repeating trajectory information
int still = 0;            //* No motion

float vflow = 0.05;       //* Constant flow speed
float theta_flow = UNDEF; //* Constant flow direction

// Single neuron recording -----------------------------------------------------
int nsn = 1;              //* Number of neurons to record
float sn_rad_fac = 0.3;   //* Size of central region from which recorded neurons
                          //    are chosen

// Activity bump tracking ------------------------------------------------------
int tracking = 1;           //* Tracking
float track_min = 0.05;     //* Minimum activity
float track_rad_fac = 1.;   //* Tracking radius factor
int *track_rad;             //  Tracking radius for each network

// Spiking ---------------------------------------------------------------------
int spiking = 0;    //* Spiking
int cv_param = 1;   //* Coefficient of variation CV = 1/sqrt(cv_param)
int cv_count = 0;   //  Count spikes for CV decimation

// Output ----------------------------------------------------------------------
char fileroot[256] = "";  //* Output filename root
int tdump_pop = 0;        //* Population activity interval
int dump_setup = 0;       //* Setup population activity
int tdump_sn = 0;         //* Single neuron rate map interval
int tdump_t = 10;         //* Tracking and trajectory interval
int dump_spike = 0;       //* Output spikes
int tdump_pmov = 0;       //* Population activity movie frame interval
int pmov_start = -1;      //* Population activity movie start frame
int pmov_len = -1;        //* Population activity movie end frame

// Fourier transformation ------------------------------------------------------
int threads = 1;              //* Number of fftw threads

char *w_id;                   //  Neural populations
float ***r_dir;               //  Activity rate for each population
fftwf_complex ***r_fourier;   //  Transformed activity rates
fftwf_plan **r_forward;       //  Forward transforms

fftwf_complex ***w_fourier;   //  Transformed inhibition kernels
fftwf_complex *u_fourier;     //  Transformed coupling kernels
fftwf_complex **rwu_fourier;  //  Transformed rates convolved with kernels
float **rwu;                  //  Rates convolved with kernels transformed back
fftwf_plan *rwu_reverse;      //  Reverse transform


static inline int fmini (int x, int y) {
  return (x<y)?x:y;
}

static inline int fmaxi (int x, int y) {
  return (x>y)?x:y;
}

//==============================================================================
// END Parameters and default value declarations
//==============================================================================



//==============================================================================
// Reading in parameters
//==============================================================================

// Parsing command line arguments
void get_parameters (int argc, char *argv[]) {

  int narg = 1;
  while (narg < argc) {
    
    if (!strcmp(argv[narg],"-randseed")) {
      sscanf(argv[narg+1],"%d",&randseed);
      narg += 2;
    } else if (!strcmp(argv[narg],"-rinit")) {
      sscanf(argv[narg+1],"%f",&rinit);
      narg += 2;
    } else if (!strcmp(argv[narg],"-rnoise")) {
      sscanf(argv[narg+1],"%f",&rnoise);
      narg += 2;
    }
    
    else if (!strcmp(argv[narg],"-tsim")) {
      sscanf(argv[narg+1],"%d",&tsim);
      narg += 2;
    } else if (!strcmp(argv[narg],"-tscreen")) {
      sscanf(argv[narg+1],"%d",&tscreen);
      narg += 2;
    } else if (!strcmp(argv[narg],"-tflow0")) {
      sscanf(argv[narg+1],"%d",&tflow0);
      narg += 2;
    } else if (!strcmp(argv[narg],"-tflow1")) {
      sscanf(argv[narg+1],"%d",&tflow1);
      narg += 2;
    } else if (!strcmp(argv[narg],"-tflow2")) {
      sscanf(argv[narg+1],"%d",&tflow2);
      narg += 2;
    } else if (!strcmp(argv[narg],"-tau")) {
      sscanf(argv[narg+1],"%f",&tau);
      narg += 2;
    }
    
    else if (!strcmp(argv[narg],"-n")) {
      sscanf(argv[narg+1],"%d",&n);
      narg += 2;
    } else if (!strcmp(argv[narg],"-np")) {
      sscanf(argv[narg+1],"%d",&np);
      narg += 2;
    } else if (!strcmp(argv[narg],"-h")) {
      sscanf(argv[narg+1],"%d",&h);
      narg += 2;
    }
    
    else if (!strcmp(argv[narg],"-wmag")) {
      sscanf(argv[narg+1],"%f",&wmag);
      narg += 2;
    } else if (!strcmp(argv[narg],"-lmin")) {
      sscanf(argv[narg+1],"%f",&lmin);
      narg += 2;
    } else if (!strcmp(argv[narg],"-lmax")) {
      sscanf(argv[narg+1],"%f",&lmax);
      narg += 2;
    } else if (!strcmp(argv[narg],"-lexp")) {
      sscanf(argv[narg+1],"%f",&lexp);
      narg += 2;
    } else if (!strcmp(argv[narg],"-wshift")) {
      sscanf(argv[narg+1],"%d",&wshift);
      narg += 2;
    } else if (!strcmp(argv[narg],"-vgain")) {
      sscanf(argv[narg+1],"%f",&vgain);
      narg += 2;
    }
    
    else if (!strcmp(argv[narg],"-umag")) {
      sscanf(argv[narg+1],"%f",&umag);
      narg += 2;
    } else if (!strcmp(argv[narg],"-urad")) {
      sscanf(argv[narg+1],"%f",&urad);
      narg += 2;
    } else if (!strcmp(argv[narg],"-u_point")) {
      urad = 0.;
      narg++;
    } else if (!strcmp(argv[narg],"-u_dv")) {
      u_dv = 1;
      u_vd = 0;
      narg++;
    } else if (!strcmp(argv[narg],"-u_vd")) {
      u_dv = 0;
      u_vd = 1;
      narg++;
    } else if (!strcmp(argv[narg],"-u_bi")) {
      u_dv = 1;
      u_vd = 1;
      narg++;
    }
    
    else if (!strcmp(argv[narg],"-amag")) {
      sscanf(argv[narg+1],"%f",&amag);
      narg += 2;
    } else if (!strcmp(argv[narg],"-falloff")) {
      sscanf(argv[narg+1],"%f",&falloff);
      narg +=2 ;
    } else if (!strcmp(argv[narg],"-falloff_low")) {
      sscanf(argv[narg+1],"%f",&falloff_low);
      narg +=2 ;
    } else if (!strcmp(argv[narg],"-falloff_high")) {
      sscanf(argv[narg+1],"%f",&falloff_high);
      narg += 2;
    }
    
    else if (!strcmp(argv[narg],"-d")) {
      sscanf(argv[narg+1],"%d",&d);
      narg += 2;
    } else if (!strcmp(argv[narg],"-posx_init")) {
      sscanf(argv[narg+1],"%d",&posx_init);
      narg += 2;
    } else if (!strcmp(argv[narg],"-posy_init")) {
      sscanf(argv[narg+1],"%d",&posy_init);
      narg += 2;
    } else if (!strcmp(argv[narg],"-traj_in")) {
      sscanf(argv[narg+1],"%s",traj_in);
      if (NULL == (traj_vfile = fopen(traj_in,"r"))) {
        printf("Error opening trajectory file %s\n", traj_in);
        fflush(stdout);
        exit(0);
      }
      printf("Trajectory file opened successfully\n");
      narg += 2;
    } else if (!strcmp(argv[narg],"-traj_repeat")) {
      traj_repeat = 1;
      narg++;
    } else if (!strcmp(argv[narg],"-dt")) {
      sscanf(argv[narg+1],"%f",&dt);
      narg += 2;
    } else if (!strcmp(argv[narg],"-traj_dt_fac")) {
      sscanf(argv[narg+1],"%d",&traj_dt_fac);
      narg += 2;
    } else if (!strcmp(argv[narg],"-still")) {
      still = 1;
      nsn = 0;   // must place this before the -nsn flag case
      tdump_sn = 0;
      tdump_t = 0;
      narg++;
    } else if (!strcmp(argv[narg],"-vflow")) {
      sscanf(argv[narg+1],"%f",&vflow);
      narg += 2;
    } else if (!strcmp(argv[narg],"-theta_flow")) {
      sscanf(argv[narg+1],"%f",&theta_flow);
      narg += 2;
    }
    
    else if (!strcmp(argv[narg],"-nsn")) {
      sscanf(argv[narg+1],"%d",&nsn);
      narg += 2;
    } else if (!strcmp(argv[narg],"-sn_rad_fac")) {
      sscanf(argv[narg+1],"%f",&sn_rad_fac);
      narg += 2;
    
    } else if (!strcmp(argv[narg],"-notrack")) {
      tracking = 0;
      narg++;
    } else if (!strcmp(argv[narg],"-track_min")) {
      sscanf(argv[narg+1],"%f",&track_min);
      narg += 2;
    } else if (!strcmp(argv[narg],"-track_rad_fac")) {
      sscanf(argv[narg+1],"%f",&track_rad_fac);
      narg += 2;
    }

    else if (!strcmp(argv[narg],"-spike")) {
      spiking = 1;
      narg++;
    } else if (!strcmp(argv[narg],"-cv_param")) {
      sscanf(argv[narg+1],"%d",&cv_param);
      narg += 2;
    }

    else if (!strcmp(argv[narg],"-fileroot")) {
      sscanf(argv[narg+1],"%s",fileroot);
      narg += 2;
    } else if (!strcmp(argv[narg],"-tdump_pop")) {
      sscanf(argv[narg+1],"%d",&tdump_pop);
      narg += 2;
    } else if (!strcmp(argv[narg],"-dump_setup")) {
      dump_setup = 1;
      narg++;
    } else if (!strcmp(argv[narg],"-tdump_sn")) {
      sscanf(argv[narg+1],"%d",&tdump_sn);
      narg += 2;
    } else if (!strcmp(argv[narg],"-tdump_t")) {
      sscanf(argv[narg+1],"%d",&tdump_t);
      narg += 2;
    } else if (!strcmp(argv[narg],"-dump_spike")) {
      dump_spike = 1;
      narg++;
    } else if (!strcmp(argv[narg],"-tdump_pmov")) {
      sscanf(argv[narg+1],"%d",&tdump_pmov);
      narg += 2;
    } else if (!strcmp(argv[narg],"-pmov_start")) {
      sscanf(argv[narg+1],"%d",&pmov_start);
      narg += 2;
    } else if (!strcmp(argv[narg],"-pmov_len")) {
      sscanf(argv[narg+1],"%d",&pmov_len);
      narg += 2;
    }
    
    else if (!strcmp(argv[narg],"-threads")) {
      sscanf(argv[narg+1],"%d",&threads);
      narg += 2;
    }
    
    else {
      printf("unknown option: %s\n", argv[narg]);
      exit(0);
    }
  }  
}


// Printing parameter values to stdout
void print_parameters () {
 
  printf("randseed        = %d\n", randseed);
  printf("rinit           = %f\n", rinit);
  printf("rnoise          = %f\n", rnoise);

  printf("tflow0          = %d\n", tflow0);
  printf("tflow1          = %d\n", tflow1);
  printf("tflow2          = %d\n", tflow2);
  printf("tau             = %f\n", tau);

  printf("n               = %d\n", n);
  printf("np              = %d\n", np);
  printf("h               = %d\n", h);

  printf("wmag            = %f\n", wmag);
  printf("lmin            = %f\n", lmin);
  printf("lmax            = %f\n", lmax);
  printf("lexp            = %f\n", lexp);
  printf("wshift          = %d\n", wshift);
  printf("vgain           = %f\n", vgain);

  printf("umag            = %f\n", umag);
  printf("urad            = %f\n", urad);
  printf("u_dv            = %d\n", u_dv);
  printf("u_vd            = %d\n", u_vd);

  printf("amag            = %f\n", amag);
  printf("falloff         = %f\n", falloff);
  printf("falloff_low     = %f\n", falloff_low);
  printf("falloff_high    = %f\n", falloff_high);

  printf("d               = %d\n", d);
  printf("posx_init       = %d\n", posx_init);
  printf("posy_init       = %d\n", posy_init);
  printf("traj_in         = %s\n", traj_in);
  printf("traj_repeat     = %d\n", traj_repeat);
  printf("dt              = %f\n", dt);
  printf("traj_dt_fac     = %d\n", traj_dt_fac);
  printf("still           = %d\n", still);
  printf("vflow           = %f\n", vflow);
  printf("theta_flow      = %f\n", theta_flow);

  printf("nsn             = %d\n", nsn);
  printf("sn_rad_fac      = %f\n", sn_rad_fac);

  printf("track_min       = %f\n", track_min);
  printf("track_rad_fac   = %f\n", track_rad_fac);

  printf("cv_param        = %d\n", cv_param);

  printf("pmov_start      = %d\n", pmov_start);
  printf("pmov_len        = %d\n", pmov_len);

  fflush(stdout);
}


void output_float_list (float *list, int idim, char *fileroot,
                                                  char *filename) {

  char name[256];
  FILE *outf;
  int i;

  sprintf(name, "%s_%s.dat", fileroot, filename);
  if (NULL == (outf = fopen(name, "w"))) {
    printf("Error opening output file %s\n", name); fflush(stdout);
    exit(0);
  }

  for (i = 0; i < idim; i++)
    fprintf(outf, "%f ", list[i]); 

  fclose(outf);
}

void output_complex_list (fftwf_complex *list, int idim, char *fileroot,
                                                          char *filename) {

  char name[256];
  FILE *outf;
  int i;

  sprintf(name, "%s_%s.dat", fileroot, filename);
  if (NULL == (outf = fopen(name, "w"))) {
    printf("Error opening output file %s\n", name); fflush(stdout);
    exit(0);
  }

  for (i = 0; i < idim; i++)
    fprintf(outf, "%f%+fI ", crealf(list[i]), cimagf(list[i])); 

  fclose(outf);
}

void output_float_array (float **arr, int idim, int jdim, char *fileroot,
                                                            char *filename) {

  char name[256];
  FILE *outf;
  int i, j;

  sprintf(name, "%s_%s.dat", fileroot, filename);
  if (NULL == (outf = fopen(name, "w"))) {
    printf("Error opening output file %s\n", name); fflush(stdout);
    exit(0);
  }

  for (i = 0; i < idim; i++) {
    for (j = 0; j < jdim; j++) 
      fprintf(outf, "%f ", arr[i][j]); 
    fprintf(outf, "\n");
  }
  fclose(outf);
}

//==============================================================================
// END Reading in parameters
//==============================================================================



//==============================================================================
// Network structure setup
//==============================================================================

// Setup neural sheet size
void setup_network_dimensions () {

  int pad_min;

  // Minimum amount of zero-padding required by inihibition and coupling kernels
  pad_min = fmaxf(ceil(2*lmax)+wshift, floor(urad));

  // FFTW works fastest when np has small prime factors so setting np is
  // required
  if (np <= pad_min) {
    printf("Must set padded size np larger than %d\n", pad_min); fflush(stdout);
    exit(0);
  }

  // If n not set, calculate largest valid n given np size
  if (n < 0) {
    n = np - pad_min;
    if (n % 2 == 1)
      n--;
    pad = np - n;
    printf("Defining n in terms of np\n"); fflush(stdout);
  }
  else if (n > np - pad_min) {
    printf("Unpadded size n too large for padded size np\n"); fflush(stdout);
    exit(0);
  }

  n2 = n * n;
  np2 = np * np;
  npfft = np * (np/2 + 1);

}


// Setup inhibition lengthscales
void setup_lengthscales (float *l) {

  int k;
  float zz;

  for (k = 0; k < h; k++) {

    zz = (float) k / (h-1);
    
    if (lexp == 0.)
      l[k] = pow(lmin, 1.-zz) * pow(lmax, zz);
    else
      l[k] = pow(pow(lmin,lexp) + (pow(lmax,lexp)-pow(lmin,lexp)) * zz,
                                                                    1./lexp);
  }

}


// Formula for recurrent inhibition
float w_value (float x, float y, float l_value) {

  float r;
  float w;
  
  r = sqrt(x*x + y*y);

  if (r < 2. * l_value)
    w = -wmag * (1. - cos(PI*r/l_value)) / (2.*l_value*l_value);
  else
    w = 0.;

  return w;
}


// Setup recurrent inhibition kernel
void setup_recurrent () {

  int i, j, b, k, p;
  float shifted[np];
  
  float *l;
  float ***w;
  float **w_dir;
  fftwf_plan *w_forward;

  for (i = 0; i < np/2; i++)
    shifted[i] = i;
  for (; i < np; i++)
    shifted[i] = i - np;


  // Calculate the recurrent inhibition lengthscales 
  l = malloc(h * sizeof(float));
  setup_lengthscales(l);
  output_float_list(l, h, fileroot, "l");

  // Calculate tracking radii as multiple of inhibition lengthscales
  if (tdump_t && tracking) {
    track_rad = malloc(h * sizeof(int));
    for (k = 0; k < h; k++)
      track_rad[k] = (int)round(track_rad_fac * l[k]);
  }

  // Create the recurrent inhibition kernel and fourier transforms 
  w = malloc(h * sizeof(float **));
  for (k = 0; k < h; k++) {
    w[k] = malloc(np * sizeof(float *));
    for (i = 0; i < np; i++) {
      w[k][i] = malloc(np * sizeof(float));
      for (j = 0; j < np; j++)
        w[k][i][j] = w_value(shifted[i], shifted[j], l[k]);
    }
  }
  output_float_array(w[0], np, np, fileroot, "w");

  w_dir = malloc(4 * sizeof(float *));
  for (p = 0; p < 4; p++)
    w_dir[p] = malloc(np2 * sizeof(float));

  w_fourier = malloc(h * sizeof(fftwf_complex **));
  for (k = 0; k < h; k++) {
    w_fourier[k] = malloc(4 * sizeof(fftwf_complex *));
    for (p = 0; p < 4; p++)
      w_fourier[k][p] = malloc(npfft * sizeof(fftwf_complex));
  }

  w_forward = malloc(4 * sizeof(fftwf_plan));

  // Shift kernel for each population, transform, and normalize
  for (k = 0; k < h; k++) {

    for (p = 0; p < 4; p++)
      w_forward[p] = fftwf_plan_dft_r2c_2d(np, np, w_dir[p],
                                           w_fourier[k][p], FFTW_PATIENT);

    for (i = 0; i < np; i++)
      for (j = 0; j < np; j++) {
        w_dir[0][np*i+j] = w[k][(i+wshift+np)%np][ j              ]; // left
        w_dir[1][np*i+j] = w[k][ i              ][(j-wshift+np)%np]; // up
        w_dir[2][np*i+j] = w[k][ i              ][(j+wshift+np)%np]; // down
        w_dir[3][np*i+j] = w[k][(i-wshift+np)%np][ j              ]; // right
      }

    for (p = 0; p < 4; p++) {
      fftwf_execute(w_forward[p]); fftwf_destroy_plan(w_forward[p]);
      for (b = 0; b < npfft; b++)
        w_fourier[k][p][b] /= np2;
    }

  }

  // Once we saved the transformed kernels, we can delete everything else
  free(l);

  for (k = 0; k < h; k++) {
    for (i = 0; i < np; i++)
      free(w[k][i]);
    free(w[k]);
  }
  free(w);
  
  for (p = 0; p < 4; p++)
    free(w_dir[p]);
  free(w_dir);

  free(w_forward);

}


// Formula for excitatory coupling
float u_value (float x, float y) {

  float r;
  float u;
  
  r = sqrt(x*x + y*y);

  if (r < urad)
    u = umag * (cos(PI*r/urad) + 1) / ((PI-4/PI)*urad*urad);
  else
    u = 0.;

  return u;
}


// Setup coupling kernel
void setup_coupling () {

  int i, j, b;
  float shifted[np];

  float **u;
  float *u_flat;
  fftwf_plan u_forward;

  u_fourier = (fftwf_complex *)malloc(npfft * sizeof(fftwf_complex));

  // No coupling
  if (umag == 0. || (!u_dv && !u_vd)) {
    u_dv = 0;
    u_vd = 0;
  }
  // Point-to-point coupling, Fourier transform is constant
  else if (urad <= 1.0) {
    for (b = 0; b < npfft; b++)
      u_fourier[b] = umag / np2;
  }
  // Coupling with spread
  else {
  
    for (i = 0; i < np/2; i++)
      shifted[i] = i;
    for (; i < np; i++)
      shifted[i] = i - np;

    // Obtain coupling kernel
    u = malloc(np * sizeof(float *));
    for (i = 0; i < np; i++) {
      u[i] = malloc(np * sizeof(float));
      for (j = 0; j < np; j++)
        u[i][j] = u_value(shifted[i], shifted[j]);
    }
    output_float_array(u, np, np, fileroot, "u");

    u_flat = malloc(np2 * sizeof(float));

    // Transform coupling kernel
    u_forward = fftwf_plan_dft_r2c_2d(np, np, u_flat, u_fourier, FFTW_PATIENT);
    for (i = 0; i < np; i++)
      for (j = 0; j < np; j++)
        u_flat[np*i+j] = u[i][j];
    fftwf_execute(u_forward); fftwf_destroy_plan(u_forward);

    // Normalize coupling kernel
    for (b = 0; b < npfft; b++)
      u_fourier[b] /= np2;  


    for (i = 0; i < np; i++)
      free(u[i]);
    free(u);
  
    free(u_flat);
  }

}


// Setup kernels and execution plans for Fourier transforms
void setup_fourier () {

  int b, k, p;

  if (threads == 1) {
    printf("FFTW configured for 1 thread\n"); fflush(stdout);
  } else if (threads < 1 || threads > 4) {
    printf("Invalid FFTW thread number; must be 1, 2, 3, or 4\n");
      fflush(stdout);
    exit(0);
  } else {
    printf("Initializing FFTW for %d threads... ", threads); fflush(stdout);
    if (!fftwf_init_threads()) {
      printf("error!\n"); fflush(stdout);
      exit(0);
    }
    fftwf_plan_with_nthreads(threads);
    printf("done\n"); fflush(stdout);
  }


  printf("Creating fourier transform plans... "); fflush(stdout);

  setup_coupling();

  setup_recurrent(); 

  // Allocate space for temporary storage of transformed firing rates
  r_fourier = malloc(h * sizeof(fftwf_complex **));
  for (k = 0; k < h; k++) {
    r_fourier[k] = malloc(5 * sizeof(fftwf_complex *));
    for (p = 0; p < 5; p++)
      r_fourier[k][p] = malloc(npfft * sizeof(fftwf_complex));
  }

  rwu_fourier = malloc(h * sizeof(fftwf_complex *));
  for (k = 0; k < h; k++)
    rwu_fourier[k] = malloc(npfft * sizeof(fftwf_complex));

  // Allocate space for firing rates
  r_dir = malloc(h * sizeof(float **));
  for (k = 0; k < h; k++) {
    r_dir[k] = malloc(4 * sizeof(float *));
    for (p = 0; p < 4; p++)
      r_dir[k][p] = malloc(np2 * sizeof(float));
  }

  // Allocate space for firing rates convolved with kernels
  rwu = malloc(h * sizeof(float *));
  for (k = 0; k < h; k++) {
    rwu[k] = malloc(np2 * sizeof(float));
  }

  // Forward plans
  r_forward = malloc(h * sizeof(fftwf_plan *));
  for (k = 0; k < h; k++) {
    r_forward[k] = malloc(4 * sizeof(fftwf_plan));
    for (p = 0; p < 4; p++)
      r_forward[k][p] = fftwf_plan_dft_r2c_2d(np, np, r_dir[k][p],
                                              r_fourier[k][p], FFTW_PATIENT);
  }

  for (k = 0; k < h; k++)
    for (p = 0; p < 4; p++)
      for (b = 0; b < np2; b++)
        r_dir[k][p][b] = 0.;

  // Reverse plans
  rwu_reverse = malloc(h * sizeof(fftwf_plan));
  for (k = 0; k < h; k++)
    rwu_reverse[k] = fftwf_plan_dft_c2r_2d(np, np, rwu_fourier[k],
                                           rwu[k], FFTW_PATIENT);

  printf("done\n"); fflush(stdout);

}


// Setup broad input
void setup_input (float **a) {

  int i, j;
  float scaled[n];
  float r, rshifted;

  for (i = 0; i < n; i++)
    scaled[i] = (i-n/2.+0.5)/(n/2.);

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
      
      r = sqrt(scaled[i]*scaled[i] + scaled[j]*scaled[j]);

      if (falloff_high != UNDEF && r >= falloff_high)
        a[i][j] = 0.;
      else if (falloff_low == UNDEF)
        a[i][j] = amag * exp(-falloff * r*r);
      else if (r <= falloff_low)
        a[i][j] = amag;
      else {
        rshifted = r - falloff_low;
        a[i][j] = amag * exp(-falloff * rshifted*rshifted);
      }
    
    }

  output_float_array(a, n, n, fileroot, "a");

}

//==============================================================================
// END Network structure setup
//==============================================================================



//==============================================================================
// Network dynamics
//==============================================================================

// Setup neural populations and initialize firing rates
void setup_population (float ***r) {

  int i, j, b, k;


  w_id = malloc(np2 * sizeof(char));

  for (i = 0; i < n; i++)               // 2x2 population blocks: D R
    for (j = 0; j < n; j++)             //                        L U
      w_id[np*i+j] = 2*(i%2) + (j%2);

  // Random initial firing rates
  for (k = 0; k < h; k++)
    for (i = 0; i < n; i++)
      for (j = 0; j < n; j++) {
        b = np * i + j;
        r[k][i][j] = r_dir[k][w_id[b]][b] = rinit * drand48();
      }
} 


// Convolutions without coupling
void convolve_no () {

  int b, k, p;

  // Forward transform of firing rates
  for (k = 0; k < h; k++)
    for (p = 0; p < 4; p++)
      fftwf_execute(r_forward[k][p]);

  // Convolve rates with recurrent inhibition, then reverse transform to yield
  // field terms
  for (k = 0; k < h; k++)
    for (b = 0; b < npfft; b++)
      rwu_fourier[k][b] = 0.;

  for (k = 0; k < h; k++) {
    for (p = 0; p < 4; p++)
      for (b = 0; b < npfft; b++)
        rwu_fourier[k][b] += w_fourier[k][p][b] * r_fourier[k][p][b];
    fftwf_execute(rwu_reverse[k]);
  }
}

// Convolutions with dorsal-to-ventral coupling
void convolve_dv () {

  int b, k, p;

  // Forward transform of firing rates
  for (k = 0; k < h; k++)
    for (p = 0; p < 4; p++)
      fftwf_execute(r_forward[k][p]);

  // Storing total transformed rate for coupling
  for (k = 0; k < h-1; k++)
    for (b = 0; b < npfft; b++)
      r_fourier[k][4][b] =  r_fourier[k][0][b] + r_fourier[k][1][b]
                          + r_fourier[k][2][b] + r_fourier[k][3][b];

  // Convolve rates with recurrent inhibition and coupling, then reverse
  // transform to yield field terms
  for (k = 0; k < h; k++)
    for (b = 0; b < npfft; b++)
      rwu_fourier[k][b] = 0.;

  for (p = 0; p < 4; p++)
    for (b = 0; b < npfft; b++)
      rwu_fourier[0][b] += w_fourier[0][p][b] * r_fourier[0][p][b];
  fftwf_execute(rwu_reverse[0]);

  for (k = 1; k < h; k++) {
    for (p = 0; p < 4; p++)
      for (b = 0; b < npfft; b++)
        rwu_fourier[k][b] += w_fourier[k][p][b] * r_fourier[k][p][b];
    for (b = 0; b < npfft; b++)
      rwu_fourier[k][b] += u_fourier[b] * r_fourier[k-1][4][b];
    fftwf_execute(rwu_reverse[k]);
  }
}

// Convolutions with ventral-to-dorsal coupling
void convolve_vd () {

  int b, k, p;

  // Forward transform of firing rates
  for (k = 0; k < h; k++)
    for (p = 0; p < 4; p++)
      fftwf_execute(r_forward[k][p]);

  // Storing total transformed firing rate for coupling
  for (k = 1; k < h; k++)
    for (b = 0; b < npfft; b++)
      r_fourier[k][4][b] =  r_fourier[k][0][b] + r_fourier[k][1][b]
                          + r_fourier[k][2][b] + r_fourier[k][3][b];

  // Convolve rates with recurrent inhibition and coupling, then reverse
  // transform to yield field terms
  for (k = 0; k < h; k++)
    for (b = 0; b < npfft; b++)
      rwu_fourier[k][b] = 0.;

  for (p = 0; p < 4; p++)
    for (b = 0; b < npfft; b++)
      rwu_fourier[h-1][b] += w_fourier[h-1][p][b] * r_fourier[h-1][p][b];
  fftwf_execute(rwu_reverse[h-1]);

  for (k = 0; k < h-1; k++) {
    for (p = 0; p < 4; p++)
      for (b = 0; b < npfft; b++)
        rwu_fourier[k][b] += w_fourier[k][p][b] * r_fourier[k][p][b];
    for (b = 0; b < npfft; b++)
      rwu_fourier[k][b] += u_fourier[b] * r_fourier[k+1][4][b];
    fftwf_execute(rwu_reverse[k]);
  }
}

// Convolutions with bidirectional coupling
void convolve_bi () {

  int b, k, p;

  // Forward transform of firing rates
  for (k = 0; k < h; k++)
    for (p = 0; p < 4; p++)
      fftwf_execute(r_forward[k][p]);

  // Storing total transformed firing rate for coupling
  for (k = 0; k < h; k++)
    for (b = 0; b < npfft; b++)
      r_fourier[k][4][b] =  r_fourier[k][0][b] + r_fourier[k][1][b]
                          + r_fourier[k][2][b] + r_fourier[k][3][b];

  // Convolve rates with recurrent inhibition and coupling, then reverse
  // transform to yield field terms
  for (k = 0; k < h; k++)
    for (b = 0; b < npfft; b++)
      rwu_fourier[k][b] = 0.;

  for (p = 0; p < 4; p++)
    for (b = 0; b < npfft; b++)
      rwu_fourier[0][b] += w_fourier[0][p][b] * r_fourier[0][p][b];
  for (b = 0; b < npfft; b++)
    rwu_fourier[0][b] += u_fourier[b] * r_fourier[1][4][b];
  fftwf_execute(rwu_reverse[0]);

  for (p = 0; p < 4; p++)
    for (b = 0; b < npfft; b++)
      rwu_fourier[h-1][b] += w_fourier[h-1][p][b] * r_fourier[h-1][p][b];
  for (b = 0; b < npfft; b++)
    rwu_fourier[h-1][b] += u_fourier[b] * r_fourier[h-2][4][b];
  fftwf_execute(rwu_reverse[h-1]);

  for (k = 1; k < h-1; k++) {
    for (p = 0; p < 4; p++)
      for (b = 0; b < npfft; b++)
        rwu_fourier[k][b] += w_fourier[k][p][b] * r_fourier[k][p][b];
    for (b = 0; b < npfft; b++)
      rwu_fourier[k][b] += u_fourier[b] * (  r_fourier[k-1][4][b]
                                           + r_fourier[k+1][4][b] );
    fftwf_execute(rwu_reverse[k]);
  }
}


// Calculate total field for each neuron
void calculate_field (
    float ***r, float ***r_field,
    float **a, float *vgain_fac
) {

  int i, j, b, k;

  for (k = 0; k < h; k++)
    for (i = 0; i < n; i++)
      for (j = 0; j < n; j++) {
        b = np * i + j;

        r_dir[k][w_id[b]][b] = r[k][i][j];

        r_field[k][i][j] = rwu[k][b] + a[i][j] * vgain_fac[w_id[b]];
      }

} 


// If not spiking, update firing rate with field inputs
void update_rate (float ***r, float ***r_field) {

  int i, j, k;
  float dt_tau;

  dt_tau = dt / tau;

  for (k = 0; k < h; k++)
    for (i = 0; i < n; i++)
      for (j = 0; j < n; j++) {

        if (r_field[k][i][j] > 0.)
          r[k][i][j] += dt_tau * (-r[k][i][j] + r_field[k][i][j]);
        else
          r[k][i][j] -= dt_tau * r[k][i][j];
      }

}


// If spiking, generate spikes with field inputs and update firing rate
void update_spike (float ***r, float ***r_field, char ***spike) {

  int i, j, k, m;

  for (k = 0; k < h; k++)
    for (i = 0; i < n; i++)
      for (j = 0; j < n; j++) {

        if (r_field[k][i][j] > 0.) {
            
          if (cv_param == 1) 
            spike[k][i][j] = (drand48() < r_field[k][i][j]*dt) ? 1 : 0;
          else {
            // For CV < 1 (cv_param > 0), generate multiple spikes and choose
            // every cv_param-th of them
            for (m = 0; m < cv_param; m++) 
              cv_count += (drand48() < r_field[k][i][j]*dt) ? 1 : 0;
            spike[k][i][j] = (int) cv_count/cv_param;
            cv_count %= cv_param;
          }

        } else
          spike[k][i][j] = 0; 
        
        r[k][i][j] += (-dt*r[k][i][j] + spike[k][i][j])/tau; 

      }

}

// Update neuron activity
void update_neuron_activity (
    float ***r, float ***r_field,
    float **a,
    float vx, float vy,
    char ***spike
) {

  int i, j, k;
  float vgain_fac[4];

  // Perform convolutions
  switch (2*u_vd + u_dv) {

    case 0: // no coupling
      convolve_no();
      break;
    case 1: // dorsal to ventural coupling
      convolve_dv();
      break;
    case 2: // ventral to dorsal coupling
      convolve_vd();
      break;
    case 3: // bidreictional coupling
      convolve_bi();
      break;

  }

  // Update field with broad input, inhibition, and coupling
  vgain_fac[0] = 1.-vgain*vx; // left
  vgain_fac[1] = 1.+vgain*vy; // up
  vgain_fac[2] = 1.-vgain*vy; // down
  vgain_fac[3] = 1.+vgain*vx; // right

  calculate_field(r, r_field, a, vgain_fac);

  // Add noise
  if (rnoise > 0.)
    for (k = 0; k < h; k++)
      for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
          r_field[k][i][j] += rnoise * r4_nor(&randseed,r4kn,r4fn,r4wn);      

  // Update population activity
  if (spiking)
    update_spike(r, r_field, spike);
  else
    update_rate(r, r_field);

}

// Flow the population activity without animal motion
void flow_neuron_activity (
    float v, float theta,
    int tflow, int nphase,
    float ***r, float ***r_field,
    float **a, 
    char ***spike
) {

  int tnow;
  float vx, vy;

  clock_t tic, toc;

  vx = v * cos(theta);
  vy = v * sin(theta);
  
  tic = clock();
  for (tnow = 1; tnow <= tflow; tnow++) {

    if (tscreen && tnow % tscreen == 0)
      printf("constant flow phase %d: %d\n", nphase, tnow); fflush(stdout);

    update_neuron_activity(r, r_field, a, vx, vy, spike);
    
  }
  toc = clock();
  printf("Flow phase %d: %f seconds\n", nphase,
                  (double)(toc-tic)/CLOCKS_PER_SEC);

}

//==============================================================================
// END Network dynamics
//==============================================================================



//==============================================================================
// Animal trajectory
//==============================================================================

// Get velocities from file if they exist
int get_velocity (FILE *traj_vfile, float *vx, float *vy) {

  int m;

  // If traj_dt_fac > 0, take every traj_dt_fac-th entry
  if (traj_dt_fac > 0) {

    for (m = 0; m < traj_dt_fac; m++)
      if (fscanf(traj_vfile, "%f %f", vx, vy) != 2)
        return 1;
  
  }
  // If traj_dt_fac < 0, repeat each entry -traj_dt_fac times
  else {
  
    traj_dt_count++;
    
    if (traj_dt_count == -traj_dt_fac) {
      if (fscanf(traj_vfile, "%f %f", vx, vy) != 2)
        return 1;
      traj_dt_count = 0;
    }

  }

  return 0;
}


// Initialize trajectory and timestep
void setup_trajectory_external (
    float *posx, float *posy, float *vx, float *vy
) {

  *posx = posx_init;
  *posy = posy_init;

  if (get_velocity(traj_vfile, vx, vy) != 0) {
    printf("Error reading velocity file\n"); fflush(stdout);
    exit(0);
  }

  // Scale simulation timestep dt according to traj_dt_fac
  if (traj_dt_fac > 0)
    dt *= traj_dt_fac;
  else if (traj_dt_fac < 0) {
    dt /= (float)(-traj_dt_fac);
    traj_dt_count = 0;
  } else {
    printf("traj_dt_fac cannot be 0\n"); fflush(stdout);
    exit(0);
  }
}


// Update trajectory
int update_position_external (
    float *posx, float *posy, float *vx, float *vy
) {

  *posx += *vx * dt;
  *posy += *vy * dt;

  if (get_velocity(traj_vfile, vx, vy) != 0) {
    if (traj_repeat) {
      printf("End of trajectory file. returning to beginning...\n");
        fflush(stdout);
      rewind(traj_vfile);
      get_velocity(traj_vfile, vx, vy);
    }
    else return 1;
  }
  
  return 0;
}


// Setup trajectory output files
void setup_trajfile (FILE **trajfile) {

  char name[256];

  sprintf(name, "%s_traj.dat", fileroot);
  *trajfile = fopen(name, "w");

}

//==============================================================================
// END Animal trajectory
//==============================================================================



//==============================================================================
// Output population activity
//==============================================================================

// Output population as h concatenated n x n tables
void output_population (int tnow, float ***r) {

  char name[256];
  int i, j, k;

  if (tnow >= 0)
    sprintf(name, "%s_p-%07d.dat", fileroot, tnow);
  else
    sprintf(name, "%s_p-m%d.dat", fileroot, -tnow);
  FILE *outf = fopen(name, "w");
      
  for (k = 0; k < h; k++) {
    for (i = 0; i < n; i++) {
      for (j = 0; j < n; j++) 
        fprintf(outf, "%.4f ", r[k][i][j]); 
      fprintf(outf, "\n");
    }
    fprintf(outf, "\n");
  }

  fclose(outf);
}


// Concatenate current frame as n x n tables to movie files
void output_population_movie (FILE **pmovfiles, float ***r) {

  int i, j, k;

  for (k = 0; k < h; k++)
    for (i = 0; i < n; i++) {
      for (j = 0; j < n; j++) 
        fprintf(pmovfiles[k], "%.3f ", r[k][i][j]); 
      fprintf(pmovfiles[k], "\n");
    }
}

// Open population activity video files
void setup_pmovfiles (FILE **pmovfiles) {

  char name[256];
  int k;

  for (k = 0; k < h; k++) {
    sprintf(name, "%s_pmov-%02d.dat", fileroot, k);
    pmovfiles[k] = fopen(name, "w");
  }
}


// Close population activity video files
void close_pmovfiles (FILE **pmovfiles) {

  int k;

  for (k = 0; k < h; k++)
    fclose(pmovfiles[k]);
}

//==============================================================================
// END Output population activity
//==============================================================================



//==============================================================================
// Single neuron recording
//==============================================================================

// Choose single neurons for recording and open files
void setup_rec (int **sn_neurons, float ***sn_histogram, int **has_been) {

  int c, x, y, k, m;
  float irand, jrand;
  char name[256];
  FILE *outf;

  for (k = 0; k < h; k++)
    for (c = 0; c < nsn; c++) {

      // Uniformly random coordinates within unit circle
      do {
        irand = 2 * drand48() - 1;
        jrand = 2 * drand48() - 1;
      } while (irand*irand + jrand*jrand > 1);

      // Record from neurons near center of neural sheet. sn_rad_fac multipled
      // by half the network size is the radius of the recording region.
      m = k * nsn + c;
      sn_neurons[m][0] = k;
      sn_neurons[m][1] = (int)round(n/2.-0.5 + irand * sn_rad_fac*(n/2.));
      sn_neurons[m][2] = (int)round(n/2.-0.5 + jrand * sn_rad_fac*(n/2.));
      
      for (x = 0; x < d; x++)
        for (y = 0; y < d; y++)
          sn_histogram[m][x][y] = 0.; 
    }
  
  // Print locations of recorded neurons
  sprintf(name, "%s_sn-legend.dat", fileroot);
  outf = fopen(name, "w");

  for (m = 0; m < h*nsn; m++)
    fprintf(outf, "%d %d %d %d\n", 
                   m, sn_neurons[m][0], sn_neurons[m][1], sn_neurons[m][2]);

  fclose(outf);
  
  // Initialize occupancy counts
  for (x = 0; x < d; x++)
    for (y = 0; y < d; y++)
      has_been[x][y] = 0;
}

// Update single neuron histograms and occupancy counts
void update_rec (
    float ***r, char ***spike, float posx, float posy,
    int **sn_neurons, float ***sn_histogram, int **has_been
) {

  int m;
  int xint, yint;

  xint = (int)floor(posx);
  yint = (int)floor(posy);
  
  if (xint < 0 || xint > d-1 || yint < 0 || yint > d-1) {
    printf("Warning: trajectory out of rec bound: %3d, %3d\n", xint, yint);
      fflush(stdout);
  }
  else {
    
    if (spiking)
      for (m = 0; m < h*nsn; m++)
        sn_histogram[m][xint][yint] +=
            (float)spike[sn_neurons[m][0]][sn_neurons[m][1]][sn_neurons[m][2]];
    else
      for (m = 0; m < h*nsn; m++)
        sn_histogram[m][xint][yint] +=
                       r[sn_neurons[m][0]][sn_neurons[m][1]][sn_neurons[m][2]];

    (has_been[xint][yint])++;

  }
}

// Output single neuron rate map and then clear histograms
void output_single_neuron (int tnow, int **has_been, float ***sn_histogram) {

  int m, x, y;
  FILE *outf;
  char name[256];
  int norm;

  if (tnow >= 0)
    sprintf(name, "%s_sn-%07d.dat", fileroot, tnow);
  else 
    sprintf(name, "%s_sn-m%d.dat", fileroot, -tnow);
  outf = fopen(name, "w");

  for (m = 0; m < h*nsn; m++) {
    for (x = 0; x < d; x++) {
      for (y = 0; y < d; y++) {

        norm = has_been[x][y];
        if (norm == 0)
          norm = 1;
        fprintf(outf, "%.4f ", sn_histogram[m][x][y]/norm);
        sn_histogram[m][x][y] = 0.;

      }
      fprintf(outf,"\n");
    }
    fprintf(outf,"\n");
  }

  fclose(outf);

  if (tnow >= 0)
    sprintf(name, "%s_visits-%07d.dat", fileroot, tnow);
  else 
    sprintf(name, "%s_visits-m%d.dat", fileroot, -tnow);
  outf = fopen(name, "w");

  for (x = 0; x < d; x++) {
    for (y = 0; y < d; y++) {
      fprintf(outf, "%d ", has_been[x][y]);
      has_been[x][y] = 0;
    }
    fprintf(outf,"\n");
  }

  fclose(outf);
  
}


// Open spiking output files
void setup_spikefiles (FILE **spikefiles) {

  int m;
  char name[256];

  for (m = 0; m < h*nsn; m++) {
    sprintf(name, "%s_sp-%02d.dat", fileroot, m);
    spikefiles[m] = fopen(name, "w");
  }
}

// Close spiking output files
void close_spikefiles (FILE **spikefiles) {

  int m;

  for (m = 0; m < h*nsn; m++)
    fclose(spikefiles[m]);
}

//==============================================================================
// END Single neuron recording
//==============================================================================



//==============================================================================
// Tracking population activity
//==============================================================================

// Refine bump location near recent location for one depth
void update_bump (float **r, float *icm, float *jcm, int track_rad, int *jump) {

  int icm0, jcm0;               // Old bump location
  int imin, imax, jmin, jmax;   // Limits of summation
  float isum, jsum;
  float rsum;

  int i, j, b;

  icm0 = (int)round(*icm);
  jcm0 = (int)round(*jcm);

  imin = fmaxi(     -icm0, -track_rad);
  imax = fmini(n-1 - icm0,  track_rad);
  jmin = fmaxi(     -jcm0, -track_rad);
  jmax = fmini(n-1 - jcm0,  track_rad);

  isum = jsum = 0.;
  rsum = 0.;

  // Calculating center of mass of activity within a radius track_rad of the old
  // location
  for (i = imin; i <= imax; i++)
    for (j = jmin; j <= jmax; j++) {
      if (i*i+j*j > track_rad*track_rad)
        continue;
      isum += i * r[icm0+i][jcm0+j];
      jsum += j * r[icm0+i][jcm0+j];
      rsum += r[icm0+i][jcm0+j];  
    }
  isum /= rsum;
  jsum /= rsum;

  *icm = icm0 + isum;
  *jcm = jcm0 + jsum;

  // If center of mass moves to the edge of the sheet or if activity is too weak,
  // indicate that a new bump needs to be found
  if (*icm >= n - track_rad || *icm < track_rad ||
      *jcm >= n - track_rad || *jcm < track_rad ||
      rsum / (PI * track_rad*track_rad) < track_min)
    *jump = 1;
  else
    *jump = 0;

}


// Find bump near center of network for one depth
void reset_bump (float **r, float *icm, float *jcm, int track_rad) {

  float icm0, jcm0;
  int jumptemp, count;
  int i, j;

  // Coarsely locate bump
  count = 0;
  do {
    *icm = n/2.-0.5 + track_rad * (2*drand48()-1);
    *jcm = n/2.-0.5 + track_rad * (2*drand48()-1);
    update_bump(r, icm, jcm, track_rad, &jumptemp); 
    count++;
    if (count > 100) {
      printf("No activity bumps found\n"); fflush(stdout);
      exit(0);
    }
  } while (jumptemp);

  // Refine bump location
  do {
    icm0 = *icm;
    jcm0 = *jcm;
    update_bump(r, icm, jcm, track_rad, &jumptemp);
  } while ((*icm-icm0)*(*icm-icm0) + (*jcm-jcm0)*(*jcm-jcm0) > 1e-8);

}


// Update bump tracking
void update_tracking (float ***r, float *icm, float *jcm, int *jump, int tnow) {

  int jumptemp;
  int k;

  for (k = 0; k < h; k++) {
    update_bump(r[k], &icm[k], &jcm[k], track_rad[k], &jumptemp);

    // Find a new bump if old one is lost
    if (jumptemp) {
      jump[k] = 1;
      reset_bump(r[k], &icm[k], &jcm[k], track_rad[k]);
      printf("Tracking reset during tnow %d for depth %d\n", tnow, k);
        fflush(stdout);
    }
  }
}


// Initialize bump tracking
void setup_tracking (
    float ***r, float *icm, float *jcm,
    int *jump, FILE **trackfile
) {

  int k;
  char name[256];

  for (k = 0; k < h; k++) {
    reset_bump(r[k], &icm[k], &jcm[k], track_rad[k]);
    jump[k] = 0;
  }

  sprintf(name, "%s_t.dat", fileroot);
  *trackfile = fopen(name, "w");  
}


// Output bump tracking
void output_tracking (
    int tnow, float *icm, float *jcm,
    int *jump, FILE *trackfile
) {

  int k;

  fprintf(trackfile, "%d ", tnow);
  for (k = 0; k < h; k++) {
    fprintf(trackfile, "%.3f %.3f %1d  ", icm[k], jcm[k], jump[k]);
    jump[k] = 0;
  }
  fprintf(trackfile, "\n");
}


//==============================================================================
// END Tracking population activity
//==============================================================================




int main (int argc, char *argv[]) {

  float ***r;             // Firing rates
  float ***r_field;       // Input field to each neuron

  float **a;              // Broad excitatory input

  int **sn_neurons;       // Network locations of recorded neurons
  float ***sn_histogram;  // Single neuron histograms
  int **has_been;         // Occupancy of spatial positions

  char ***spike;          // Spikes
  FILE **spikefiles;

  FILE **pmovfiles;

  float posx, posy;       // Animal locations
  float vx, vy;           // Animal velocities
  
  FILE *trajfile;

  float *icm, *jcm;       // Activity bump locations
  int *jump;              // Activity bump resets
  FILE *trackfile;

  int tnow;               // Overall time
  int i, k, m;
  int x;
  int c, place;



  get_parameters(argc, argv);
  
  if (strcmp(fileroot, "") == 0 ||
      fileroot[0] == '-' ||
      fileroot[strlen(fileroot)-1] == '/') {
    printf("Must set -fileroot correctly\n"); fflush(stdout);
    exit(0);
  }

  if (!randseed) {
    randseed = (unsigned)time(NULL);
    place = 1;
    for (c = 0; c < strlen(fileroot); c++) {
      randseed += fileroot[c] * place;
      place = (place * 128) % 100000;
    }
  }
  srand48(randseed);

  setup_network_dimensions();

  print_parameters();
  



  // Setup recurrent inhibition
  setup_fourier();


  // Setup broad excitatory input
  a = (float **)malloc(n * sizeof(float *));
  for (i = 0; i < n; i++)
    a[i] = (float *)malloc(n * sizeof(float));
  setup_input(a);
  

  // Setup firing rates
  r = (float ***)malloc(h * sizeof(float **));
  for (k = 0; k < h; k++) {
    r[k] = (float **)malloc(n * sizeof(float *));
    for (i = 0; i < n; i++)
      r[k][i] = (float *)malloc(n * sizeof(float));
  }
  r_field = (float ***)malloc(h * sizeof(float **));
  for (k = 0; k < h; k++) {
    r_field[k] = (float **)malloc(n * sizeof(float *));
    for (i = 0; i < n; i++)
      r_field[k][i] = (float *)malloc(n * sizeof(float));
  }
  setup_population(r);


  // Setup trajectory
  if (still) {
    posx = posx_init;
    posy = posy_init;
    vx = 0.;
    vy = 0.;
  }
  else {
    setup_trajectory_external(&posx, &posy, &vx, &vy);
    setup_trajfile(&trajfile);
  }


  // Setup spiking
  if (spiking) {
    spike = (char ***)malloc(h * sizeof(char **));
    for (k = 0; k < h; k++) {
      spike[k] = (char **)malloc(n * sizeof(char *));
      for (i = 0; i < n; i++)
        spike[k][i] = (char *)malloc(n * sizeof(char));
    }
  }


  // Setup single neuron recording
  if (tdump_sn || dump_spike) {

    if (tdump_sn == -1)
      tdump_sn = tsim;

    sn_neurons = (int **)malloc(h*nsn * sizeof(int *));
    for (m = 0; m < h*nsn; m++)
      sn_neurons[m] = (int *)malloc(3 * sizeof(int));

    sn_histogram = (float ***)malloc(h*nsn * sizeof(float **));
    for (m = 0; m < h*nsn; m++) {
      sn_histogram[m] = (float **)malloc(d * sizeof(float *));
      for (x = 0; x < d; x++)
        sn_histogram[m][x] = (float *)malloc(d * sizeof(float));
    }
  
    has_been = (int **)malloc(d * sizeof(int *));
    for (x = 0; x < d; x++)
      has_been[x] = (int *)malloc(d * sizeof(int));

    setup_rec(sn_neurons, sn_histogram, has_been);
    
    if (dump_spike) {
      spikefiles = (FILE **)malloc(h*nsn * sizeof(FILE *));
      setup_spikefiles(spikefiles);
    }
      
  }


  if (dump_setup)
    output_population(-5, r);


  // Flow neural population activity with constant velocity. If flow angle is
  // undefined, flow at angles PI/2 - PI/5, 2*PI/5, and PI/4 sequentially.
  printf("\nSTARTING FLOW\n"); fflush(stdout);
  flow_neuron_activity(0., 0., tflow0, 1, r, r_field, a, spike);
  output_population(-4, r);
  
  if (theta_flow == UNDEF) {
    flow_neuron_activity(vflow, PI/2-PI/5, tflow1, 2, r, r_field, a, spike);
    if (dump_setup)
      output_population(-3, r);
    flow_neuron_activity(vflow, 2*PI/5, tflow1, 3, r, r_field, a, spike);
    if (dump_setup)
      output_population(-2, r);
    flow_neuron_activity(vflow, PI/4, tflow1, 4, r, r_field, a, spike);
    if (dump_setup)
      output_population(-1, r);
  } else {
    flow_neuron_activity(vflow, theta_flow, tflow1, 2, r, r_field, a, spike);
    if (dump_setup)
      output_population(-3, r);
    flow_neuron_activity(vflow, theta_flow, tflow1, 3, r, r_field, a, spike);
    if (dump_setup)
      output_population(-2, r);
    flow_neuron_activity(vflow, theta_flow, tflow1, 4, r, r_field, a, spike);
    if (dump_setup)
      output_population(-1, r);
  }

  // Flow neural population activity with trajectory
  if (!still)
    for (tnow = 1; tnow <= tflow2; tnow++) {

      if (tnow % tscreen == 0)
        printf("trajectory flow: %d\n", tnow); fflush(stdout);

      if (update_position_external(&posx, &posy, &vx, &vy) != 0) {
        printf("not enough trajectory data.\n"); fflush(stdout);
        exit(0);
      }
      
      update_neuron_activity(r, r_field, a, vx, vy, spike);
      
    }


  if (tdump_pop) {
    if (tdump_pop == -1)
      tdump_pop = tsim;
    output_population(0, r);
  }

  if (!still)
    fprintf(trajfile, "%d %.3f %.3f\n", 0, posx, posy);


  // Setup population activity video
  if (tdump_pmov) {
    if (pmov_len == -1)
      pmov_len = tsim;
    if (pmov_start == -1)
      pmov_start = tsim - pmov_len;
    printf("Pop movie frames: %d\n", pmov_len/tdump_pmov + 1);
    pmovfiles = (FILE **)malloc(h * sizeof(FILE *)); fflush(stdout);
    setup_pmovfiles(pmovfiles);
    if (pmov_start == 0)
      output_population_movie(pmovfiles, r);
  }


  // Setup tracking
  if (tdump_t && tracking) {
    icm = (float *)malloc(h * sizeof(float));
    jcm = (float *)malloc(h * sizeof(float));
    jump = (int *)malloc(h * sizeof(int));
    
    setup_tracking(r, icm, jcm, jump, &trackfile);
    output_tracking(0, icm, jcm, jump, trackfile);
  }

  

  // Main simulation loop
  printf("\nSTARTING MAIN SIMULATION\n"); fflush(stdout);
  for (tnow = 1; tnow <= tsim; tnow++) {

    if (tscreen && tnow % tscreen == 0)
      printf("%d, pos %.3f %.3f\n", tnow, posx, posy); fflush(stdout);

    // Update position
    if (!still) {
      if (update_position_external(&posx, &posy, &vx, &vy) != 0) {
        printf("No more trajectroy data.\n"); fflush(stdout);
        tnow = tsim + 1;
      }

      if (tdump_t && tnow % tdump_t == 0)
        fprintf(trajfile, "%d %.3f %.3f\n", tnow, posx, posy);
    }
    
    // Update activity
    update_neuron_activity(r, r_field, a, vx, vy, spike);


    // Update and output data
    if (tdump_pop && tnow % tdump_pop == 0)
      output_population(tnow, r);
    if (tdump_pmov && tnow % tdump_pmov == 0 && 
        tnow >= pmov_start && tnow <= pmov_start + pmov_len)
      output_population_movie(pmovfiles, r);

    if (tdump_sn) {
      update_rec(r, spike, posx, posy, sn_neurons, sn_histogram, has_been);
      if (tnow % tdump_sn == 0)
        output_single_neuron(tnow, has_been, sn_histogram);
    }

    if (dump_spike) 
      for (m = 0; m < h*nsn; m++)
        if (spike[sn_neurons[m][0]][sn_neurons[m][1]][sn_neurons[m][2]] == 1)
          fprintf(spikefiles[m], "%d %.3f %.3f\n", tnow, posx, posy);

    if (tdump_t && tnow % tdump_t == 0 && tracking) {
      update_tracking(r, icm, jcm, jump, tnow);
      output_tracking(tnow, icm, jcm, jump, trackfile);
    }

  }

  // Close files
  if (tdump_pmov)
    close_pmovfiles(pmovfiles);
  if (dump_spike)
    close_spikefiles(spikefiles);
  if (!still)
    fclose(trajfile);
  if (tdump_t && tracking)
    fclose(trackfile);

}
