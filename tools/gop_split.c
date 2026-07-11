#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define AVS2_SC_SEQUENCE_HEADER 0xB0

static void usage(const char *prog) {
    fprintf(stderr, "Usage: %s -i <input.avs2> -o <output.avs2> -n <num_gop>\n", prog);
    fprintf(stderr, "  Extract first N GOPs from AVS2 Annex B bitstream\n");
}

int main(int argc, char *argv[]) {
    const char *in_path = NULL, *out_path = NULL;
    int num_gop = 0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-i") && i + 1 < argc) in_path = argv[++i];
        else if (!strcmp(argv[i], "-o") && i + 1 < argc) out_path = argv[++i];
        else if (!strcmp(argv[i], "-n") && i + 1 < argc) num_gop = atoi(argv[++i]);
        else {
            usage(argv[0]);
            return 1;
        }
    }

    if (!in_path || !out_path || num_gop <= 0) {
        usage(argv[0]);
        return 1;
    }

    FILE *fin = fopen(in_path, "rb");
    if (!fin) {
        fprintf(stderr, "Cannot open input: %s\n", in_path);
        return 1;
    }

    fseek(fin, 0, SEEK_END);
    long file_size = ftell(fin);
    fseek(fin, 0, SEEK_SET);

    uint8_t *buf = (uint8_t *)malloc((size_t)file_size);
    if (!buf) {
        fprintf(stderr, "Memory allocation failed\n");
        fclose(fin);
        return 1;
    }

    fread(buf, 1, (size_t)file_size, fin);
    fclose(fin);

    int gop_count = 0;
    long end_pos = file_size;

    for (long i = 0; i < file_size - 3; i++) {
        if (buf[i] == 0 && buf[i + 1] == 0 && buf[i + 2] == 1) {
            uint8_t sc_id = buf[i + 3];
            if (sc_id == AVS2_SC_SEQUENCE_HEADER) {
                gop_count++;
                if (gop_count == num_gop + 1) {
                    end_pos = i;
                    break;
                }
            }
        }
    }

    if (gop_count < num_gop) {
        fprintf(stderr, "Warning: only %d GOPs found in input\n", gop_count);
        end_pos = file_size;
    }

    FILE *fout = fopen(out_path, "wb");
    if (!fout) {
        fprintf(stderr, "Cannot open output: %s\n", out_path);
        free(buf);
        return 1;
    }

    fwrite(buf, 1, (size_t)end_pos, fout);
    fclose(fout);
    free(buf);

    fprintf(stderr, "Extracted %d GOPs, output size: %ld bytes\n", 
            (gop_count < num_gop) ? gop_count : num_gop, end_pos);

    return 0;
}