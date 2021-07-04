#include <Eigen/Dense>
#include <Eigen/Core>

#include <iostream>
#include <utility>

#include "tests.h"
#include "kshape.h"

using namespace Eigen;

typedef Matrix<long double,Dynamic,Dynamic> MatrixXld;
typedef Matrix< std::complex< long double >, Dynamic, Dynamic > MatrixXcld;
typedef Matrix< long double, Dynamic, 1 > VectorXld;
typedef Matrix< std::complex< long double >, Dynamic, 1 > VectorXcld;

void _unit_test_NCC(){
  /*
    >>> _ncc_c([1,2,3,4], [1,2,3,4])
    array([ 0.13333333,  0.36666667,  0.66666667,  1.        ,  0.66666667,
            0.36666667,  0.13333333])
    >>> _ncc_c([1,1,1], [1,1,1])
    array([ 0.33333333,  0.66666667,  1.        ,  0.66666667,  0.33333333])
    >>> _ncc_c([1,2,3], [-1,-1,-1])
    array([-0.15430335, -0.46291005, -0.9258201 , -0.77151675, -0.46291005])
  VectorXld v1(4);
  VectorXld v2(4);
  v1 << 1,2,3,4;
  v2 << 1,2,3,4;
  std::cout << NCC(v1,v2);
  VectorXld v1(3);
  VectorXld v2(3);
  v1 << 1,1,1;
  v2 << 1,1,1;
  std::cout << NCC(v1,v2);
  */
  VectorXld v1(3);
  VectorXld v2(3);
  v1 <<1,2,3;
  v2 <<-1,-1,-1;
  std::cout << NCC(v1,v2);
}

void _unit_test_kshape(){
  MatrixXld x(5,4);
  x << 1,2,3,4,5,6,7,8,12,-2234,1000,19,-1,-1,0,0,0,0,1,0;
  auto a = kshape(z_norm(x,1,0),2,1);
  std::vector<int> cluster_counts(2);
  for (auto i: a.first) cluster_counts[i]++;
  std::cout << std::endl;
  for (int i=0;i<2;++i){
    std::cout << i << ": " << cluster_counts[i] << std::endl;
  }
  write_csv("out_centroids.csv", a.second);
  write_csv("out_indices.csv", a.first);
}

void _unit_test_SBD(){
    /*VectorXld x(3);
    VectorXld y(3);
    x << 1,1,1;
    y << 1,1,1;
    std::pair<long double,VectorXld> res = SBD (x, y);
    std::cout << res.first << std::endl;
    std::cout << res.second.transpose() << std::endl;
    VectorXld x(3);
    VectorXld y(3);
    x << 0,1,2;
    y << 1,2,3;
    std::pair<long double,VectorXld> res = SBD (x, y);
    std::cout << res.first << std::endl;
    std::cout << res.second.transpose() << std::endl;*/
    VectorXld x(3);
    VectorXld y(3);
    x << 1,2,3;
    y << 0,1,2;
    std::pair<long double,VectorXld> res = SBD (x, y);
    std::cout << res.first << std::endl;
    std::cout << res.second.transpose() << std::endl;
}

void _unit_test_shape_extract(){
    VectorXld cur_center(3);
    cur_center << 0,3,4;
    MatrixXld cluster(1,3);
    cluster << 4,5,6;
    std::cout << extractShape(cluster,  cur_center) << std::endl;
}

void _unit_test_znorm(){
  MatrixXld mat(3,3);

  mat << 1,2,3,4,6,8,100,200,3000;

  std::cout << "Axis 0,0\n\n" << z_norm(mat, 0,0);
  std::cout << "\n\nAxis 0,1\n\n" << z_norm(mat, 0,1);
  std::cout << "\n\nAxis 1,0\n\n" << z_norm(mat, 1,0);
  std::cout << "\n\nAxis 1,1\n\n" << z_norm(mat, 1,1) << std::endl;
}

void _unit_test_NCC3(){
      MatrixXld x(3,3);
      MatrixXld y(2,3);
      x << 8,7,1,23,1,34,8,234,1;
      y << 1,2,3,6,7,8;
      std::vector<MatrixXld> v = NCC3D(x, y);
      for (auto tx: v){ std::cout << tx << std::endl <<std::endl; }
}
