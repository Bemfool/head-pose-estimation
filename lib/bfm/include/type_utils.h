#pragma once


/* 
 * Struct: is_double
 * Usage: if(is_double<T>()) { }
 * Return:
 * 		Is `double` type or not.
 * ***********************************************************************************
 * Judge that the type is `double` or not.
 * 
 */

template <typename T>
struct is_double
{
	operator bool() 
	{
		return false;
	}
};
 
template <>
struct is_double<double>
{
	operator bool() 
	{
		return true;
	}
};


/* 
 * Function: length
 * Usage: int len = length(array);
 * Parameters:
 * 		@arr: Array.
 * Return:
 * 		Size of array.
 * ***********************************************************************************
 * Get the size of an array.
 * 
 */

template<class T> inline 
int length(T& arr) 
{
    return sizeof(arr) / sizeof(arr[0]);
}


/* 
 * Function: print_array
 * Usage: print_array(arr);
 * Parameters:
 * 		@array:	Array to be printed;
 * 		@len: Size of array.
 * ***********************************************************************************
 * Print a list.
 * 
 */

template<class _Tp> inline
void print_array(_Tp *array, int len)
{
	for(int i=0; i<len; i++)
		std::cout << array[i] << " ";
	std::cout << "\n";
}