template <typename T>
struct is_double{
	operator bool() {
		return false;
	}
};
 
template <>
struct is_double<double>{
	operator bool() {
		return true;
	}
};
