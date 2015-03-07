#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <limits.h>

#define BLOCK_SIZE 1024
unsigned int nextPow2(unsigned int x)
{
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}
typedef struct
{
    unsigned int* code_length;
    unsigned long long int* codeword;
    int* d_check;
}Code;
// __device__ unsigned long long int hamming_distance( unsigned long long int x,  unsigned long long int y)
// {
//     unsigned long long int  dist;
//     unsigned long long int  val;
//     dist = 0;
//     val = x ^ y;
//     while (val != 0)
//     {
//         dist++;
//         val &= val - 1;
//     }
//     return dist;
// }
__global__ void intial_code(Code output)
{
  output.codeword[0]=0;
  output.codeword[1]=ULONG_MAX;
  output.code_length[0]=1;
  output.d_check[0]=0;
}
__global__ void cal_code_kernal( unsigned long int n_length, unsigned long int distance,Code output, unsigned long long int check_word)
{
  unsigned long int index = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned long long int  val;
  const unsigned long long int m1  = 0x5555555555555555; //binary: 0101...
  const unsigned long long int m2  = 0x3333333333333333; //binary: 00110011..
  const unsigned long long int m4  = 0x0f0f0f0f0f0f0f0f; //binary:  4 zeros,  4 ones ...
  const unsigned long long int m8  = 0x00ff00ff00ff00ff; //binary:  8 zeros,  8 ones ...
  const unsigned long long int m16 = 0x0000ffff0000ffff; //binary: 16 zeros, 16 ones ...
  const unsigned long long int m32 = 0x00000000ffffffff; //binary: 32 zeros, 32 ones
  const unsigned long long int hff = 0xffffffffffffffff; //binary: all ones
  const unsigned long long int h01 = 0x0101010101010101; //the sum of 256 to the power of 0,1,2,3...
  if(index<output.code_length[0])
  {
    val = output.codeword[index] ^ check_word;
    val -= (val >> 1) & m1;             //put count of each 2 bits into those 2 bits
    val = (val & m2) + ((val >> 2) & m2); //put count of each 4 bits into those 4 bits 
    val = (val + (val >> 4)) & m4;        //put count of each 8 bits into those 8 bits 
    val= (val * h01)>>56;  //returns left 8 bits of x + (x<<8) + (x<<16) + (x<<24) + ... 
    output.d_check[index] =!(val<distance);  
  }
//   if(index<output.code_length[0])
//   {
//     val = output.codeword[index] ^ check_word;
//     output.d_check[index] =!(__popcll(val)<distance);  
//   }
}
__global__ void add_code_kernal(Code output, unsigned long long int add_word)
{
  if(output.d_check[0]==output.code_length[0])
  {
    output.codeword[ output.code_length[0]]=add_word;
    output.codeword[ output.code_length[0]+1]=ULONG_MAX;
    output.code_length[0]++; 
  }
}

__global__ void parallel_reduction(Code output)
{
  int* g_idata=output.d_check;
  int* g_odata=output.d_check;
  unsigned long int n= output.code_length[0];
  unsigned int s=0;
  extern __shared__ int sdata[];
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
  
  int mySum = (i < n) ? g_idata[i] : 0;
  if (i + blockDim.x < n)
    mySum += g_idata[i+blockDim.x];
  
  sdata[tid] = mySum;
  __syncthreads();
  for (s=blockDim.x/2; s>0; s>>=1)
  {
    if (tid < s)
    {
      sdata[tid] = mySum = mySum + sdata[tid + s];
    }
    
    __syncthreads();
  }
  if (tid == 0) g_odata[blockIdx.x] = mySum;
}
void cal_code( unsigned long int n_length, unsigned long int distance,Code output,unsigned int output_k)
{
  float compute_time,dtransfer_time;
  cudaEvent_t start, stop;
 
  unsigned long int i;
  unsigned long int no_vectors=pow(2,n_length);
  int threads=BLOCK_SIZE;
  int blocks =1;
  dim3 block;
  block.x=0;
  block.y=1;
  block.z=1;
  dim3 grid;
  grid.x=0;
  grid.y=1;
  grid.z=1;
  
  Code d_output;
  cudaMalloc(&d_output.code_length,sizeof(unsigned int));
  
  size_t mem_size_output =pow(2,output_k)* sizeof(unsigned long long int); 
  cudaMalloc(&d_output.codeword, mem_size_output);
  
  size_t mem_size_check =pow(2,output_k)* sizeof(int); 
  cudaMalloc(&d_output.d_check, mem_size_check);
  

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  
  unsigned int code_length[1];
  int s=0;
  int smemsize=0;
  intial_code<<<1,1>>>(d_output);
  for(i=1;i<no_vectors;i++)
  {
    cudaMemcpy(code_length,d_output.code_length,sizeof(unsigned int), cudaMemcpyDeviceToHost);
    threads = (code_length[0] < BLOCK_SIZE) ? nextPow2(code_length[0]) : BLOCK_SIZE;
    blocks= (code_length[0] + threads - 1) / threads;
    block.x=threads;
    grid.x=blocks;
    cal_code_kernal<<<grid,block>>>(n_length,distance,d_output,i);
    
    smemsize = (threads <= 32) ? 2 * threads * sizeof(unsigned int) : threads * sizeof(unsigned int);
    parallel_reduction<<<grid,block,smemsize>>>(d_output);
    s=blocks;
    while (s>1)
    {
      threads = (s< BLOCK_SIZE) ? nextPow2(s) : BLOCK_SIZE;
      blocks= (s + threads - 1) / threads;
      block.x=threads;
      grid.x=blocks;
      parallel_reduction<<<grid,block,smemsize>>>(d_output);
      s = (s + threads - 1) / threads;
    }
    add_code_kernal<<<1,1>>>(d_output,i);
  }
  
  
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&compute_time, start, stop);
  FILE * file = fopen("time2.txt","w");
  fprintf(file,"computation time:  %f ms \n", compute_time);
  
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  
  cudaMemcpy(output.codeword, d_output.codeword, mem_size_output, cudaMemcpyDeviceToHost); 
  
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&dtransfer_time, start, stop);
  
  fprintf(file,"data transfer time:  %f ms \n", dtransfer_time);
  
  float total_time= (compute_time+dtransfer_time)/1000;
  fprintf(file,"total time:  %f s \n", total_time);
  
  cudaFree(d_output.codeword);
}
void int_to_binary( unsigned long long int n, unsigned long long int integer,FILE * file){
  
  unsigned long long int k;
  unsigned long long int c;
  for (c = n-1; c >0; c--)
    {
        k = integer >> c;
        if (k & 1)
            fprintf(file,"1");
        else
            fprintf(file,"0");

    }
     k = integer >> c;
        if (k & 1)
            fprintf(file,"1");
        else
            fprintf(file,"0");
    fprintf(file,"\n");
    
}
int main() 
{
  unsigned long int i;
  unsigned long int n_length=23,distance=3;
  unsigned int output_k=20;//put output_length manually
  //printf("Enter \'n\' and \'distance\' and\'len\' :=\n");
 // scanf("%d %d %d",&n_length,&distance,&output_k);
  Code output; 
  output.codeword=( unsigned long long int*)malloc(pow(2,output_k)*sizeof( unsigned long long int));
  cal_code(n_length,distance,output,output_k);
  FILE * file = fopen("code2.txt","w");
  for(i=0;;i++)
  {
    if(output.codeword[i]==ULONG_MAX)
    {
      break;
    }
    int_to_binary(n_length,output.codeword[i],file);
    
  }
  fclose(file);
}
