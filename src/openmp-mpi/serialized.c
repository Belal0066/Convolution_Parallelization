#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/time.h>


typedef enum {RGB, GREY} color_t;

void convolute(uint8_t *src, uint8_t *dst, int width, int height, float **h, color_t imageType);
void convolute_grey(uint8_t *src, uint8_t *dst, int x, int y, int width, float **h);
void convolute_rgb(uint8_t *src, uint8_t *dst, int x, int y, int width, float **h);
uint8_t *offset(uint8_t *array, int i, int j, int width);
void Usage(int argc, char **argv, char **image, int *width, int *height, int *loops, color_t *imageType);

int main(int argc, char **argv) {
    int i, j, width, height, loops;
    char *image;
    color_t imageType;

    Usage(argc, argv, &image, &width, &height, &loops, &imageType);


    float **h = malloc(3 * sizeof(float *));
    for (i = 0; i < 3; i++) {
        h[i] = malloc(3 * sizeof(float));
        for (j = 0; j < 3; j++) {
            int gaussian_blur[3][3] = {{1, 2, 1}, {2, 4, 2}, {1, 2, 1}};
            h[i][j] = gaussian_blur[i][j] / 16.0f;
        }
    }

    int pixels = (imageType == GREY) ? width * height : width * height * 3;
    int padded_width = (imageType == GREY) ? width + 2 : width * 3 + 6;
    int padded_height = height + 2;

    uint8_t *src = calloc(padded_width * padded_height, sizeof(uint8_t));
    uint8_t *dst = calloc(padded_width * padded_height, sizeof(uint8_t));
    if (!src || !dst) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }

    // Read image file
    FILE *f = fopen(image, "rb");
    if (!f) {
        fprintf(stderr, "Cannot open image: %s\n", image);
        exit(EXIT_FAILURE);
    }
    for (i = 1; i <= height; i++) {
        fread(offset(src, i, (imageType == GREY) ? 1 : 3, padded_width),
              sizeof(uint8_t), (imageType == GREY) ? width : width * 3, f);
    }
    fclose(f);

    //start timer
    struct timeval start, end;
    gettimeofday(&start, NULL);

    // Convolution loop
    for (int t = 0; t < loops; t++) {
        convolute(src, dst, width, height, h, imageType);
        uint8_t *tmp = src;
        src = dst;
        dst = tmp;
    }

    // Write output
    char out_name[256];
    snprintf(out_name, sizeof(out_name), "blur_serial", image);
    f = fopen(out_name, "wb");
    if (!f) {
        fprintf(stderr, "Cannot write image: %s\n", out_name);
        exit(EXIT_FAILURE);
    }
    for (i = 1; i <= height; i++) {
        fwrite(offset(src, i, (imageType == GREY) ? 1 : 3, padded_width),
               sizeof(uint8_t), (imageType == GREY) ? width : width * 3, f);
    }
    fclose(f);

    gettimeofday(&end, NULL); // End timing

    double time_taken = (end.tv_sec - start.tv_sec) * 1e6;
    time_taken = (time_taken + (end.tv_usec - start.tv_usec)) / 1e6;
    printf("Serial convolution completed in %.6f seconds\n", time_taken);

    // Cleanup
    free(src);
    free(dst);
    for (i = 0; i < 3; i++) free(h[i]);
    free(h);
    free(image);
    return 0;
}

void convolute(uint8_t *src, uint8_t *dst, int width, int height, float **h, color_t imageType) {
    int padded_width = (imageType == GREY) ? width + 2 : width * 3 + 6;

    for (int i = 1; i <= height; i++) {
        for (int j = 1; j <= width; j++) {
            if (imageType == GREY) {
                convolute_grey(src, dst, i, j, padded_width, h);
            } else {
                convolute_rgb(src, dst, i, j * 3, padded_width, h);
            }
        }
    }
}

void convolute_grey(uint8_t *src, uint8_t *dst, int x, int y, int width, float **h) {
    float val = 0.0f;
    for (int i = x - 1, k = 0; i <= x + 1; i++, k++) {
        for (int j = y - 1, l = 0; j <= y + 1; j++, l++) {
            val += src[i * width + j] * h[k][l];
        }
    }
    dst[x * width + y] = (uint8_t)val;
}

void convolute_rgb(uint8_t *src, uint8_t *dst, int x, int y, int width, float **h) {
    float r = 0.0f, g = 0.0f, b = 0.0f;
    for (int i = x - 1, k = 0; i <= x + 1; i++, k++) {
        for (int j = y - 3, l = 0; j <= y + 3; j += 3, l++) {
            r += src[i * width + j] * h[k][l];
            g += src[i * width + j + 1] * h[k][l];
            b += src[i * width + j + 2] * h[k][l];
        }
    }
    dst[x * width + y] = (uint8_t)r;
    dst[x * width + y + 1] = (uint8_t)g;
    dst[x * width + y + 2] = (uint8_t)b;
}

uint8_t *offset(uint8_t *array, int i, int j, int width) {
    return &array[i * width + j];
}

void Usage(int argc, char **argv, char **image, int *width, int *height, int *loops, color_t *imageType) {
    if (argc == 6 && strcmp(argv[5], "grey") == 0) {
        *image = strdup(argv[1]);
        *width = atoi(argv[2]);
        *height = atoi(argv[3]);
        *loops = atoi(argv[4]);
        *imageType = GREY;
    } else if (argc == 6 && strcmp(argv[5], "rgb") == 0) {
        *image = strdup(argv[1]);
        *width = atoi(argv[2]);
        *height = atoi(argv[3]);
        *loops = atoi(argv[4]);
        *imageType = RGB;
    } else {
        fprintf(stderr, "Usage: %s <image> <width> <height> <loops> <grey|rgb>\n", argv[0]);
        exit(EXIT_FAILURE);
    }
}
