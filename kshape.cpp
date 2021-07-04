#include <Eigen/Dense>
#include <Eigen/Core>

#include <iostream>
#include <fstream>
#include <utility>
#include <cmath>
#include <experimental/random>

#include <cstdlib>
#include <Eigen/unsupported/Eigen/FFT>


using namespace Eigen;

#include "kshape.h"

typedef Matrix<long double,Dynamic,Dynamic> MatrixXld;
typedef Matrix< std::complex< long double >, Dynamic, Dynamic > MatrixXcld;
typedef Matrix< long double, Dynamic, 1 > VectorXld;
typedef Matrix< std::complex< long double >, Dynamic, 1 > VectorXcld;


VectorXld coefWiseDivision(VectorXld x, VectorXld y){
    VectorXld ans(x.size());
    for (int i=0;i<x.size();++i){
          ans(i) = x(i)/y(i);
    }
    return ans;
}

VectorXld circshift(VectorXld vec,int shift)
{
  if (shift==0)
	{
		return vec;
	}

	int n = vec.size();
	VectorXld y(n);
  y.setZero();


	if (shift > 0) // shift right
	{
		y.head(shift) = vec.tail(shift);
		y.tail(n - shift) = vec.head(n - shift);
	}

  if(shift<0) // shift left
	{
		y.head(n + shift) = vec.tail(n + shift);
		y.tail(abs(shift)) = vec.head(abs(shift));

	}

	return y;
}


VectorXld shiftWithZeors(VectorXld vec, int shift){
  int n = vec.size();
  VectorXld newvec(n);
  newvec.setZero();
  if (shift==0)
	{
    newvec.array() = vec.array();
		return newvec;
	}

  if (std::abs(shift)>n){
    return newvec;
  }

	 if (shift > 0) // shift right
	{
		newvec.tail(n - shift) = vec.head(n - shift);
    //vec.head(shift).setZero();
	}

  if(shift<0) // shift left
	{
		newvec.head(n + shift) = vec.tail(n + shift);
		//vec.tail(abs(shift)).setZero();
	}

	return newvec;
}

VectorXld NCC(VectorXld x, VectorXld y){
    long double normed =  x.norm()*y.norm();
    if (normed == 0){
        normed =  1000000000000.0 ;
    }

    int x_len = x.size();

    int fft_size = 1 << (int)ceil(log2((2*x_len - 1)));

    FFT<long double> fft;
    VectorXcld x_out;

    VectorXld transformed_x(fft_size);
    transformed_x.setZero();

    transformed_x.head(x_len) = x;
    fft.fwd(x_out, transformed_x);

    VectorXcld y_out;
    VectorXld transformed_y(fft_size);
    transformed_y.setZero();
    transformed_y.head(y.size()) = y;
    fft.fwd(y_out, transformed_y);

    VectorXld cc_out;
    VectorXcld inter =  x_out.cwiseProduct( y_out.conjugate() );
    fft.inv(cc_out, inter);

    VectorXld cc(2*x_len - 1);
    cc.setZero();
    cc.head(x_len-1) = cc_out.tail(x_len-1);
    cc.tail(x_len) = cc_out.head(x_len);

    return cc.real() / normed;
}

std::vector<MatrixXld> NCC3D(MatrixXld x, MatrixXld y){
    VectorXld x_norm = x.rowwise().norm();
    VectorXld y_norm = y.rowwise().norm();
    MatrixXld den(x_norm.size(),y_norm.size());
    for (int i = 0;i<x_norm.size();++i){
          den.row(i) = x_norm(i)*(y_norm);
    }
    den = den.unaryExpr([](long double v) { return v==0.0 ? 1000000000.0 : v; });
    int x_len = x.cols();
    int y_len = y.cols();

    int fft_size = 1 << (int)ceil(log2((2*x_len - 1)));

    FFT<long double> fft;

    MatrixXcld x_out;
    MatrixXcld y_out;

    std::vector<MatrixXld> cc(y.rows());

    for (int k=0;k<y.rows();++k){

      VectorXld y_in(fft_size);
      y_in.setZero();
      VectorXcld fft_y_out;

      y_in.head(y_len) = y.row(k);
      fft.fwd(fft_y_out, y_in);

      MatrixXld cc_row(x.rows(), 2*x_len-1);
      for (int j=0;j<x.rows();++j){
          VectorXld x_in(fft_size);
          x_in.setZero();
          VectorXcld fft_x_out;

          x_in.head(x_len) = x.row(j);
          fft.fwd(fft_x_out, x_in);
          VectorXld inverted_product;
          VectorXcld t1 = fft_x_out.cwiseProduct(fft_y_out.conjugate());
          fft.inv(inverted_product, t1);

          cc_row.row(j).head(x_len-1) = inverted_product.tail(x_len-1);
          cc_row.row(j).tail(x_len) = inverted_product.head(x_len);
          cc_row.row(j) = cc_row.row(j).array() / den.coeff(j,k);
      }
      cc[k]=cc_row;
    }

    return cc;
}



std::pair<long double,VectorXld> SBD (VectorXld x, VectorXld y){
      VectorXld ncc = NCC(x, y);
      int idx;
      long double dist = 1.0 - ncc.maxCoeff(&idx);

      VectorXld yshift = shiftWithZeors(y, (idx+1)-std::max(x.size(),y.size()));

     return std::make_pair(dist, yshift);
}

MatrixXld z_norm(MatrixXld a, int axis = 0,int ddof=0){
      VectorXld means;
      VectorXld sstd;
      if (axis == 0){
        means = a.colwise().mean();
        int n = means.size();

        VectorXld sstd = VectorXld(n);
        for (int i=0;i<n;++i){
          sstd(i)=((a.col(i).array()-means(i)).pow(2).colwise().sum().sum()/(a.rows()-ddof));
        }

        sstd = sstd.array().pow(0.5);


        MatrixXld temp = (a.rowwise()-means.transpose());
        for (int i=0;i<n;i++){

          temp.col(i) = (temp.col(i)/(sstd(i)));
        }

        return temp;
      }
      else {
        means = a.rowwise().mean();
        int n = means.size();
        VectorXld sstd = VectorXld(n);
        for (int i=0;i<n;++i){
          sstd(i)=((a.row(i).array()-means(i)).pow(2).rowwise().sum().sum()/(a.cols()-ddof));
        }

        sstd = sstd.array().pow(0.5);


        MatrixXld temp = (a.colwise()-means);
        for (int i=0;i<n;i++){
          temp.row(i) = (temp.row(i)/(sstd(i)));
        }

        return temp;
      }
}

VectorXld extractShape(MatrixXld cluster, VectorXld cur_center){
    if (cluster.rows()==1){
      return cluster.row(0);
    }

      for (int i=0;i<cluster.rows();++i){
          if (cur_center.sum()!=0.0){
              cluster.row(i) = SBD(cur_center,cluster.row(i)).second;
          }
    }
    MatrixXld y = z_norm(cluster, 1,1);
    MatrixXld s = y.transpose()*y;

    int columns = cluster.cols();

    MatrixXld p(columns, columns);
    p.array() = 1.0/(long double)columns;

    p = MatrixXld::Identity(columns, columns) - p;

    MatrixXld m = (p.transpose()*s)*p;
    SelfAdjointEigenSolver<MatrixXld> eigensolver(m);
    MatrixXcld eigenVectors = eigensolver.eigenvectors();
    int cols = eigenVectors.cols();
    VectorXcld eigenvalues = eigensolver.eigenvalues();

    VectorXld centroid = eigenVectors.col(cols-1).real();

    long double dist1 = (cluster.row(0)-centroid.transpose()).array().pow(2).sum();
    long double dist2 = (cluster.row(0)+centroid.transpose()).array().pow(2).sum();

    if (dist1>=dist2){
      centroid.array() *= -1.0;
    }

    return z_norm(centroid,0,1);

  }


std::pair<std::vector<int>, MatrixXld> kshape(MatrixXld x, int k, int runs_to_average_over){
    int m = x.rows();
    int n = x.cols();

    std::vector<int> vidx(m);
    std::vector<int> bestidx(m);

    MatrixXld bestcentroids(k,n);
    bestcentroids.setZero();

    int number_in_center[k];
    long double bestdist=0.0;

    for (int run=0;run<runs_to_average_over;++run){
          MatrixXld centroids(k,n);
          centroids.setZero();

          bool changed;
          long double dist;
          memset(number_in_center,0,sizeof(number_in_center));

          for (int i=0;i<m;i++){
            int rint = std::experimental::randint(0, k-1);
            vidx[i] = rint;
            number_in_center[rint] += 1;
          }

          for (int iter=0;iter<100;++iter){
            std::cout << "Commence Iteration " << iter;
            for (int j=0;j<k;++j){
                int center_j_size = number_in_center[j];
                MatrixXld center_j = MatrixXld(center_j_size, n);
                int row = 0;
                for (int l=0;l<m;++l){
                    if (vidx[l]==j){
                        center_j.row(row) = x.row(l);
                        ++row;
                    }
                }

                if (row==0){
                  centroids.row(j).array() = 0;
                }
                else {
                  centroids.row(j) = extractShape(center_j, centroids.row(j)).transpose();
                }

            }

            std::vector<MatrixXld> x_corr = NCC3D(x, centroids);

            MatrixXld distances(m, k);

            for (int i=0;i<k;++i){
              distances.col(i) = 1 - x_corr[i].rowwise().maxCoeff().array();
            }
            memset(number_in_center,0,sizeof(number_in_center));
            changed = false;
            dist = 0.0;
            for (int i=0;i<x.rows();++i){
              int tempi;
              dist+=std::abs(distances.row(i).minCoeff(&tempi));
              if (vidx[i]!=tempi){
                vidx[i]=tempi;
                changed = true;
              }
              number_in_center[tempi]++;
            }

            std::cout << "\r";
            if (!changed){
              if (dist>bestdist){
                bestdist=dist;
                bestidx=vidx;
                bestcentroids = centroids;
              }
              break;
            }
          }

          if (changed){
            if (dist>bestdist){
              bestdist=dist;
              bestidx=vidx;
              bestcentroids = centroids;
            }
          }
    }

    return std::make_pair(bestidx, bestcentroids);
}

MatrixXld read_csv(std::string filename, int rows, int cols){
      std::ifstream file(filename);
      MatrixXld mat(rows,cols);

      std::string line;
      long double val; int row=0;

      while(std::getline(file, line))
      {
          std::stringstream ss(line);
          // Keep track of the current column index
          int col = 0;
          // Extract each integer
          while(ss >> val){
              // Add the current integer to the 'colIdx' column's values vector
              mat(row,col) = val;
              // If the next token is a comma, ignore it and move on
              if(ss.peek() == ',') ss.ignore();
              // Increment the column index
              col++;
          }
        row++;
      }

      file.close();

      return z_norm(mat,1,1);
}

void write_csv(std::string filename, MatrixXld mat){
      std::ofstream file(filename);

      for(int i = 0; i < mat.rows(); ++i)
      {
          for(int j = 0; j < mat.cols(); ++j)
          {
              file << mat(i,j);
              if(j != mat.cols() - 1) file << ",";
          }
          file << "\n";
      }
      file.close();

}

void write_csv(std::string filename, std::vector<int> v){
      std::ofstream file(filename);

      for(int j = 0; j < v.size(); ++j){
            file << v[j];
            if(j != v.size() - 1) file << ",";
      }
      file << "\n";
      file.close();
}


int main(int argc, char* argv[]){

  if (argc < 6){
    std::cerr << "Need to speficy a csv, and its number of rows and columns, as well as K and a number of runs" << std::endl;
    return 1;
  }

  std::string csv_filename = std::string(argv[1]);
  int rows = atoi(argv[2]), cols = atoi(argv[3]), k = atoi(argv[4]), runs=atoi(argv[5]);

  MatrixXld mat = read_csv(csv_filename, rows, cols);
  mat = z_norm(mat, 1,1);
  std::pair<std::vector<int>, MatrixXld> result = kshape(mat, k, runs);
  write_csv("out_centroids.csv", result.second);
  write_csv("out_indices.csv", result.first);
  std::vector<int> cluster_counts(k);
  for (auto i: result.first) cluster_counts[i]++;
  std::cout << std::endl;
  for (int i=0;i<k;++i){
    std::cout << i << ": " << cluster_counts[i] << std::endl;
  }

  return 0;
}
