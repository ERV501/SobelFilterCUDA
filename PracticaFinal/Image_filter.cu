#include <thread>
#include <chrono>
#include <time.h>
#include <iostream>
#include <math.h>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace cv;
using namespace std;

//Numero de hilos por bloque
#define N_THREADS 32.0

__global__ void filter_Sobel(unsigned char* src_img,unsigned char* out_img, unsigned int width, unsigned int height) {

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;
    float Gx, Gy; //Kernel para las direcciones x e y
    float G;
    float G_0_0,G_0_1,G_0_2,G_1_0,G_1_1,G_1_2,G_2_0,G_2_1,G_2_2;

    /* Comprobar los limites de la imagen */
    if(idx > 0 && idy > 0 && idx < width-1 && idy < height-1) { 

        /***********************************   Multiplicamos los valores de cada gradiente por la posición correspondiente (idx,idy) de la imagen original
              -1 0 +1              -1 -2 -1    Iteramos por los pixeles de la imagen mediante el uso de idx, idy y el ancho de la imagen
        Gx => -2 0 +2        Gy =>  0  0  0     
              -1 0 +1              +1 +2 +1
        ***********************************/
        G_0_0 = src_img[(idy-1)*width + (idx-1)];                
        G_0_1 = src_img[(idy-1)*width + (idx)];                       
        G_0_2 = src_img[(idy-1)*width + (idx+1)];                      
        G_1_0 = src_img[(idy)*width + (idx-1)];                        
        G_1_1 = src_img[(idy)*width + (idx)];                          
        G_1_2 = src_img[(idy)*width + (idx+1)];                        
        G_2_0 = src_img[(idy+1)*width + (idx-1)];                      
        G_2_1 = src_img[(idy+1)*width + (idx)];                        
        G_2_2 = src_img[(idy+1)*width + (idx+1)];                      

        Gx = (-1 * G_0_0) + (0 * G_0_1) + (1 * G_0_2) +
             (-2 * G_1_0) + (0 * G_1_1) + (2 * G_1_2) +
             (-1 * G_2_0) + (0 * G_2_1) + (1 * G_2_2);

        Gy = (1 * G_0_0) + (2 * G_0_1) + (1 * G_0_2) +
             (0 * G_1_0) + (0 * G_1_1) + (0 * G_1_2) +
             (-1 * G_2_0) + (-2 * G_2_1) + (-1 * G_2_2);
        
        /* El gradiente resultante (G) es la raiz cuadrada de (Gx^2 + Gy^2) */
        G = sqrt(pow(Gx,2) + pow(Gy,2));

        /* Modificamos el pixel que estamos comprobando */
        if(G > 255){  
            out_img[idy*width + idx] = 255;     //En caso de que sobrepasemos el valor maximo posible para este pixel (255), ponemos este ultimo como su valor actual
        }else{
            out_img[idy*width + idx] = G;       //Cualquier otro caso -> valor correspondiente
        }
    }
}

int main(int argc, char*argv[]) {
    /** Comprobar linea de comandos **/
    if(argc < 2 || argc > 3) {
        printf("\033[1;31mError: Invalid number of command line arguments.\nUsage: %s [image.png] [filter_option]\033[0m \n", argv[0]);
        return 1;
    }
    /** Propiedades de nuestro dispositvo CUDA **/
	cudaDeviceProp dev_properties;
	cudaGetDeviceProperties(&dev_properties, 0);
	int cores = dev_properties.multiProcessorCount;
	switch (dev_properties.major)
	{
	case 2: // Fermi
		if (dev_properties.minor == 1) cores *= 48;
		else cores *= 32; break;
	case 3: // Kepler
		cores *= 192; break;
	case 5: // Maxwell
		cores *= 128; break;
	case 6: // Pascal
		if (dev_properties.minor == 1) cores *= 128;
		else if (dev_properties.minor == 0) cores *= 64;
        break;
    case 7: // Volta and Turing
        if ((dev_properties.minor == 0) || (dev_properties.minor == 5)) cores *= 64;
        else printf("Unknown device type\n");
        break;
    case 8: // Ampere
        if (dev_properties.minor == 0) cores *= 64;
        else if (dev_properties.minor == 6) cores *= 128;
        else printf("Unknown device type\n");
        break;
    default:
        printf("Unknown device type\n"); 
        break;
    }
    
    /** Imprimir informacion (hardware threads, GPU info, etc) **/
    printf("CPU: %d hardware threads\n", std::thread::hardware_concurrency());
    printf("GPU: %s, CUDA %d.%d, %zd Mbytes global memory, %d CUDA cores\n",
    dev_properties.name, dev_properties.major, dev_properties.minor, dev_properties.totalGlobalMem / 1048576, cores);

    /** Buscar nuestra imagen **/
    string image_path = "./input_images/";

    /** Cargar nuestra imagen en escala de grises **/
    Mat original_img = imread(image_path + argv[1],IMREAD_GRAYSCALE);
    //Mat original_img = imread(image_path + argv[1], IMREAD_GRAYSCALE);
    if(original_img.empty()){
        printf("\033[1;31mError: Image not found\nPlease make sure it's on the \"input_images\" folder\033[0m \n");
        return 1;
    }

    /** En caso de que pasemos un filtro Gaussiano para suavizar el ruido de la imagen y mejorar su resultado **/
    Mat modified_img;
    if (argc == 3){
        int gauss_size = atoi(argv[2]);

        if(gauss_size % 2 == 0){
            printf("\033[1;31mError: Gauss filter size must be an odd number (Ej: 3,5,7,etc)\033[0m \n"); //Impar, ya que el kernel debe ser simetrico
            return 1;
        }
        GaussianBlur(original_img,modified_img,Size(gauss_size,gauss_size),0);

    /** Para la ejecucion normal del filtro Sobel **/
    }else{
        modified_img = original_img;
    }


    /** Datos que necesitamos de la imagen **/
    unsigned int img_data_height= modified_img.rows;
    unsigned int img_data_width = modified_img.cols;
    int img_data_size = img_data_width * img_data_height;

    /** Asignar espacio en la GPU para nuestra original img, new img, y dimensiones **/
    unsigned char *src_img, *out_img;
    cudaMalloc( (void**)&src_img, img_data_size);
    cudaMalloc( (void**)&out_img, img_data_size);

    
    /** Transferir memoria del host al device **/
    cudaMemcpy(src_img, modified_img.data, img_data_size, cudaMemcpyHostToDevice);
   
    /** Preparar argumentos dim3 para la GPU (threads per block & num of blocks)**/
    dim3 dimBlocks(N_THREADS, N_THREADS, 1); //hebras por bloque, bloque de 2 dimensiones
    dim3 dimGrid(ceil(img_data_width/N_THREADS), ceil(img_data_height/N_THREADS), 1); //numero de bloques (ceil para redondear valores al alza)

    auto start_time = chrono::system_clock::now(); //Tiempo inicio

    auto c = std::chrono::system_clock::now();

    /** Sobel mediante GPU **/
    filter_Sobel<<<dimGrid, dimBlocks>>>(src_img, out_img, img_data_width, img_data_height);

    chrono::duration<double> end_time = chrono::system_clock::now() - start_time; //Tiempo final
    printf("Execution time: \033[1;34m%*.5f\033[0m miliseconds\n",5,1000*end_time.count()); //Imprimir tiempo de ejecucion

    /** Copiar datos de vuelta al host **/
    cudaMemcpy(modified_img.data, out_img, img_data_size, cudaMemcpyDeviceToHost);

    /** Tamaño de nuestra imagen **/
    printf("\nProcessing %s: \033[1;34m%d\033[0m rows x \033[1;34m%d\033[0m columns\nTotal size: \033[1;34m%d\033[0m pixels \r\n", argv[1], img_data_width, img_data_height, img_data_size);

    /** Escribir imagen mediante OpenCV **/
    imwrite( "output_image.png", modified_img );

    /** Liberar memoria asignada previamente **/
    cudaFree(src_img);
    cudaFree(out_img);
    return 0;
}