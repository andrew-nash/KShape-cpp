#include <Eigen/Dense>
#include <Eigen/Core>
#include <utility>
#include <vector>

using namespace Eigen;

typedef Matrix<long double,Dynamic,Dynamic> MatrixXld;
typedef Matrix< std::complex< long double >, Dynamic, Dynamic > MatrixXcld;
typedef Matrix< long double, Dynamic, 1 > VectorXld;
typedef Matrix< std::complex< long double >, Dynamic, 1 > VectorXcld;

using namespace Eigen;

VectorXld coefWiseDivision(VectorXld x, VectorXld y);
VectorXld circshift(VectorXld vec,int shift);
VectorXld shiftWithZeors(VectorXld vec, int shift);
VectorXld NCC(VectorXld x, VectorXld y);
std::vector<MatrixXld> NCC3D(MatrixXld x, MatrixXld y);
std::pair<long double,VectorXld> SBD (VectorXld x, VectorXld y);
MatrixXld z_norm(MatrixXld a, int axis,int ddof);
VectorXld extractShape(MatrixXld cluster, VectorXld cur_center);
std::pair<std::vector<int>, MatrixXld> kshape(MatrixXld x, int k, int runs_to_average_over);
MatrixXld read_csv(std::string filename, int rows, int cols);
void write_csv(std::string filename, MatrixXld mat);
void write_csv(std::string filename, std::vector<int> v);
