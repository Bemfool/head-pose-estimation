#pragma once
#include <string>
#include <vector>

/* Function: split_string
 * Usage: split_string(source, object, separater);
 * Parameters:
 * 		s: source string to be split
 * 		v: a series of string after split
 * 		c: separater
 * --------------------------------------------------------------------------------------------
 * Seperate s with c, and save split series into v.
 */
void split_string(const std::string& s, std::vector<std::string>& v, const std::string& c);
