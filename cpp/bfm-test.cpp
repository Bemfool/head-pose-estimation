#include "hpe.h"

int main() {
    init_bfm();
    data_check();
    generate_random_face(0.0);
    ply_write("rnd_face.ply");
    return 0;
}