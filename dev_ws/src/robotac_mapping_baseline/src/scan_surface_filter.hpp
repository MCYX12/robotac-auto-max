#ifndef ROBOTAC_SCAN_SURFACE_FILTER_HPP_
#define ROBOTAC_SCAN_SURFACE_FILTER_HPP_
#include <array>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <unordered_map>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>

namespace robotac_mapping_baseline {
// Per-acquisition geometric support. Never reads accumulated map geometry,
// scene boundaries, ground truth, or rejected candidates from an earlier scan.
// Support samples are original measured coordinates, deduplicated into 5 mm
// buckets so repeated identical rays cannot manufacture a dense surface.
class ScanSurfaceFilter {
  struct Key {
    int64_t x,y,z;
    bool operator==(const Key & b) const {return x==b.x && y==b.y && z==b.z;}
  };
  struct Hash {
    std::size_t operator()(const Key & k) const {
      std::size_t h=std::hash<int64_t>{}(k.x);
      h^=std::hash<int64_t>{}(k.y)+0x9e3779b9+(h<<6)+(h>>2);
      h^=std::hash<int64_t>{}(k.z)+0x9e3779b9+(h<<6)+(h>>2);return h;
    }
  };
  static Key key(const Eigen::Vector3d & p) {
    return {static_cast<int64_t>(std::floor(p.x()/.005)),
      static_cast<int64_t>(std::floor(p.y()/.005)),
      static_cast<int64_t>(std::floor(p.z()/.005))};
  }
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_{new pcl::PointCloud<pcl::PointXYZ>()};
  pcl::KdTreeFLANN<pcl::PointXYZ> tree_;
  std::unordered_map<Key, bool, Hash> decisions_;
public:
  explicit ScanSurfaceFilter(const std::vector<Eigen::Vector3d> & points) {
    std::unordered_map<Key,bool,Hash> seen;
    cloud_->reserve(points.size());
    for(const auto & p:points) {
      if(!p.allFinite() || p.norm()>100.0) {continue;}
      if(seen.emplace(key(p),true).second) {cloud_->emplace_back(p.x(),p.y(),p.z());}
    }
    if(!cloud_->empty()) {tree_.setInputCloud(cloud_);}
  }
  bool accepts(const Eigen::Vector3d & point) {
    if(!point.allFinite() || point.norm()>100.0 || cloud_->size()<3) {return false;}
    const auto bucket=key(point);const auto cached=decisions_.find(bucket);
    if(cached!=decisions_.end()) {return cached->second;}
    std::vector<int> ids(40);std::vector<float> distances(40);
    const int found=tree_.nearestKSearch(pcl::PointXYZ(point.x(),point.y(),point.z()),40,ids,distances);
    std::vector<Eigen::Vector3d> neighbors;neighbors.reserve(40);
    for(int i=0;i<found;++i) {
      if(distances[i]>.75*.75) {break;}
      const auto & p=(*cloud_)[ids[i]];neighbors.emplace_back(p.x,p.y,p.z);
    }
    const auto nearby=std::count_if(distances.begin(),distances.begin()+found,[](float d){return d<=.35*.35;});
    if(nearby<3) {decisions_.emplace(bucket,false);return false;}
    // Sparse or multi-surface neighborhoods are unknown, not proof that the
    // query is wrong. Existing independent temporal admission still applies.
    bool accepted=true;
    if(neighbors.size()>=12) {
      uint32_t state=4829;
      const auto next=[&state]() {state^=state<<13;state^=state>>17;state^=state<<5;return state;};
      int best=0;Eigen::Vector3d normal=Eigen::Vector3d::Zero();double offset=0;
      for(int trial=0;trial<48;++trial) {
        const auto a=next()%neighbors.size(),b=next()%neighbors.size(),c=next()%neighbors.size();
        if(a==b || a==c || b==c) {continue;}
        Eigen::Vector3d n=(neighbors[b]-neighbors[a]).cross(neighbors[c]-neighbors[a]);
        const double length=n.norm();if(length<.000001) {continue;}n/=length;
        const double d=-n.dot(neighbors[a]);int support=0;
        for(const auto & p:neighbors) {if(std::abs(n.dot(p)+d)<.012) {++support;}}
        if(support>best) {best=support;normal=n;offset=d;}
      }
      const bool decisive=best>=8 && best>=.60*neighbors.size();
      if(decisive && std::abs(normal.dot(point)+offset)>=.015) {
        accepted=false;
        std::vector<Eigen::Vector3d> remaining;
        for(const auto & p:neighbors) {if(std::abs(normal.dot(p)+offset)>=.025) {remaining.push_back(p);}}
        // A corner can contain a coherent minority face. Fit it using only
        // observations outside the primary face; one outlier plus primary
        // inliers cannot manufacture a second plane.
        if(remaining.size()>=6) {
          int second_best=0;Eigen::Vector3d second_normal=Eigen::Vector3d::Zero();double second_offset=0;
          for(int trial=0;trial<48;++trial) {
            const auto a=next()%remaining.size(),b=next()%remaining.size(),c=next()%remaining.size();
            if(a==b || a==c || b==c) {continue;}
            Eigen::Vector3d n=(remaining[b]-remaining[a]).cross(remaining[c]-remaining[a]);
            const double length=n.norm();if(length<.000001) {continue;}n/=length;
            if(std::abs(n.dot(normal))>.94) {continue;}
            const double d=-n.dot(remaining[a]);int support=0;
            for(const auto & p:remaining) {if(std::abs(n.dot(p)+d)<.012) {++support;}}
            if(support>second_best) {second_best=support;second_normal=n;second_offset=d;}
          }
          accepted=second_best>=6 && second_best>=.70*remaining.size() && std::abs(second_normal.dot(point)+second_offset)<.015;
        }
      }
    }
    decisions_.emplace(bucket,accepted);return accepted;
  }
};
}
#endif
