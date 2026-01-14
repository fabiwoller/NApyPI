#include <pybind11/pybind11.h>
#include "algo.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_core, m) {
    m.doc() = "napy C++ backend";

    m.def("add", &add, "Add two integers");
    m.def("beta", &beta, "Get beta value");
}
