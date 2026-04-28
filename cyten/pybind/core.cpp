#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>

#include <check.h>

using namespace std;
using namespace pybind11::literals; // provides "arg"_a literals

PYBIND11_MODULE(_core, m) {
    m.doc() = "check that python bindings work."; // optional module docstring
    m.def("add", &cyten::add, "A function that adds two numbers");
}
