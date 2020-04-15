#pragma once
#include <string>
#include <iostream>
#include <fstream>
#include <dlib/matrix.h>

template<typename _Tp> inline 
void mat_write(std::string fn, const dlib::matrix<_Tp> &mat, std::string matn)
{
	std::ofstream out;
	out.open(fn, std::ios::out | std::ios::app);
	if (!out) {
		std::cout << "Open of " << fn.c_str() << " failed.\n";
		return;
	}
    out << matn << "\n";
    out << mat << "\n";
    out.close();
}


void str_write(std::string fn, const std::string &s)
{
	std::ofstream out;
	out.open(fn, std::ios::out | std::ios::app);
	if (!out) {
		std::cout << "Open of " << fn.c_str() << " failed.\n";
		return;
	}
    out << s << "\n";
    out.close();
}