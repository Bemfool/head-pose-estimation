#include "bfm.h"

int main() {
    init_bfm();
    data_check();
    generate_random_face();
    save_ply("rnd_face.ply");
    return 0;
}