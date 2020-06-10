#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <pthread.h>
#include "thpool.h"

#include <unistd.h>

#define THREAD_MAX 10

struct arg_struct {
    float *img;
    unsigned char *mask;
    long N;
    int thread_id;
};


void argmax3D_thread(void *a){
    struct arg_struct *arg = (struct arg_struct*) a;
    int thread_id = (*arg).thread_id;
    float *img = (*arg).img;
    unsigned char *mask = (*arg).mask;
    long N = (*arg).N;

    // unsigned long N = 500;
    int C = 21, H = 360, W = 640;
    long in_index, out_index;
    int ign_cls[14] = {0,1,2,3,4,5,9,10,13,14,15,16,17,18};
    float pixel;
    long n, h, w, c, ign;

    for(n=thread_id ; n<N ; n+=THREAD_MAX){
        for(h=0 ; h<H ; h++){
            for(w=0 ; w<W ; w++){
                pixel = 0;
                out_index = n*H*W + h*W + w;
                for(c=0; c<C ; c++){
                    in_index = n*C*H*W + c*H*W + h*W + w;
                    if(thread_id==0){
                        // printf("thread_id : %d , in_index: %d, out_index: %d\n", thread_id, in_index, out_index);
                    }
                    if(pixel <= img[in_index]){
                        pixel = img[in_index];
                        mask[out_index] = c;
                    }
                }
                for(ign =0; ign<14 ; ign++){
                    if (mask[out_index]==ign_cls[ign]){
                        mask[out_index] = 0;
                    }
                }
            }
        }
        // printf("thread_id : %d , n: %d , cls: %d \n", thread_id, n, mask[out_index]);
        // total_img++;
    }
}



// void Cargmax(void *img_array, unsigned long N ){
//     unsigned long C = 21, H = 360, W = 640;
//     unsigned char *out = malloc(N * H * W * sizeof(unsigned char) );

//     struct arg_struct args[THREAD_MAX];
//     for(int i = 0 ; i < THREAD_MAX ; i++){
//         args[i].img = img_array;
//         args[i].mask = out;
//         args[i].N = N;
//         args[i].thread_id = i;
//     }

//     threadpool thpool = thpool_init(THREAD_MAX);

//     for (int i = 0; i < THREAD_MAX; i++) { //create multiple threads
//         thpool_add_work(thpool, (void*)&argmax3D_thread, (void*) &args[i]);
//     };

//     thpool_wait(thpool);
//     thpool_destroy(thpool);
// }


void showdata(float *img_array){
    printf("Show_B0: %#010x\n", img_array); //0
    printf("Show_B0_1: %#010x\n", img_array+1);
    printf("Show_B1_0: %#010x\n", img_array+4838400*1); //1
    printf("Show_B2_0: %#010x\n", img_array+4838400*2); //2
    printf("Show: %#010x\n", img_array+4838400*3);
    printf("Show: %#010x\n", img_array+4838400*4);
    printf("%d \n",img_array+1);
    printf("%d \n",img_array[10]);
    printf("%d \n",img_array[4838401]);
    printf("%f \n",img_array[4838400*2+1]);
    printf("%f \n",img_array[4838400*3+1]);
    printf("%f \n",img_array[4838400*4+1]);
    printf("%f \n",img_array[4838400*5+1]);
    printf("%f \n",img_array[4838400*6+1]);
    
    // printf();
    // printf();
    // printf();

}

void Cargmax(float *img_array, unsigned char *mask_array, unsigned long N ){
    // printf("ptr: %#010x\n", img_array);
    // showdata(img_array);
    
    // unsigned long C = 21, H = 360, W = 640;
    // unsigned char *out = malloc(N * H * W * sizeof(unsigned char) );
    struct arg_struct args[THREAD_MAX];
    for(int i = 0 ; i < THREAD_MAX ; i++){
        args[i].img = img_array;
        args[i].mask = mask_array;
        args[i].N = N;
        args[i].thread_id = i;
    }

    threadpool thpool = thpool_init(THREAD_MAX);

    for (int i = 0; i < THREAD_MAX; i++) { //create multiple threads
        thpool_add_work(thpool, argmax3D_thread,&args[i]);
    };

    thpool_wait(thpool);
    thpool_destroy(thpool);
}


// clear && gcc -shared -fPIC argmax_API.c thpool.c -D THPOOL_DEBUG -pthread  -o argmax.so
