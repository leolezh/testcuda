  __global__  void vecAdd(float* A, float* B, float* C)
       {
          // threadIdx.x is a built-in variable  provided by CUDA at runtime
          int i = threadIdx.x;
          A[i]=0;
          B[i]=i;
          C[i] = A[i] + B[i];
          //printf("from thread No.%d",threadIdx.x);
       }

       #include  <stdio.h>
       #define  SIZE 10
       int  main()
       {
        int N=SIZE;
        float A[SIZE], B[SIZE], C[SIZE];
        float *devPtrA;
        float *devPtrB;
        float *devPtrC;
        int memsize= SIZE * sizeof(float);
        cudaMalloc(&devPtrA, memsize);
        cudaMalloc(&devPtrB, memsize);
        cudaMalloc((void**)&devPtrC, memsize); // the same as cudaMalloc(&devPtrC, memsize);


        cudaMemcpy(devPtrA, A, memsize,  cudaMemcpyHostToDevice);
        cudaMemcpy(devPtrB, B, memsize,  cudaMemcpyHostToDevice);
        // __global__ functions are called:  Func<<< Dg, Db, Ns  >>>(parameter);
        vecAdd<<<1, N>>>(devPtrA,  devPtrB, devPtrC);
        cudaError_t error = cudaGetLastError();
        if(error != cudaSuccess)
        {
                printf("CUDA error: errorstring %s\n", cudaGetErrorString(error));
                exit(-1);
        }

        cudaMemcpy(C, devPtrC, memsize,  cudaMemcpyDeviceToHost);

        for (int i=0; i<SIZE; i++)
           printf("C[%d]=%f\n",i,C[i]);

        cudaFree(devPtrA);
        cudaFree(devPtrA);
        cudaFree(devPtrA);
       }
