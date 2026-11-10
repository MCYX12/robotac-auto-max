#include <fstream>
#include <iostream>
#include "scan_surface_filter.hpp"
int main(int argc,char **argv) {
  if(argc!=2) {return 2;}
  std::ifstream input(argv[1]);std::size_t count=0;input>>count;
  std::vector<Eigen::Vector3d> points;points.reserve(count);
  for(std::size_t i=0;i<count;++i) {
    double x,y,z;input>>x>>y>>z;if(!input) {return 3;}points.emplace_back(x,y,z);
  }
  robotac_mapping_baseline::ScanSurfaceFilter filter(points);
  for(const auto & p:points) {std::cout<<(filter.accepts(p)?1:0)<<'\n';}
}
