#pragma once
#include <string>

#ifdef USE_QT
#include <QDebug>
#define bfm_out qDebug()
#else
#define bfm_out std::cout
#endif

enum model_write_mode {
	NONE_MODE      = 0L << 0, 
	PICK_FP 	   = 1L << 0,
	CAMERA_COORD   = 1L << 1,
	NO_EXPR        = 1L << 2,
	EXTRA_EXT_PARM = 1L << 3,
};


/* camera type */
typedef int camera_type;
enum {
	PARALLEL = 0,
	PINHOLE = 1,
};

typedef int op_type;
enum {
	COARSE = 0,
	REAL = 1,
};