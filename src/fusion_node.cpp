#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/point_field.hpp>
#include <std_msgs/msg/header.hpp>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <yaml-cpp/yaml.h>
#include <sensor_msgs/point_cloud2_iterator.hpp>

class LidarCameraProjectionNode : public rclcpp::Node {
public:
  LidarCameraProjectionNode()
    : Node("lidar_camera_projection_node"),
      tf_buffer_(this->get_clock()),
      tf_listener_(tf_buffer_)
  {
    this->declare_parameter<std::string>("general.camera_intrinsic_calibration", "");
    this->declare_parameter<std::string>("general.camera_extrinsic_calibration", "");
    this->declare_parameter<std::string>("camera.image_topic", "/image");
    this->declare_parameter<std::string>("camera.confidence_topic", "/confidence");
    this->declare_parameter<std::string>("lidar.lidar_topic", "/points_raw");
    this->declare_parameter<std::string>("lidar.lidar_topic2", "None");
    this->declare_parameter<std::string>("lidar.lidar_frame", "");
    this->declare_parameter<std::string>("lidar.lidar_frame2", "");
    this->declare_parameter<std::string>("lidar.colored_cloud_topic", "/colored_cloud");
    this->declare_parameter<double>("camera.horizontal_fov_deg", 120.0);
    this->declare_parameter<bool>("visualize", true);

    std::string cam_yaml = this->get_parameter("general.camera_intrinsic_calibration").as_string();
    std::string ext_yaml = this->get_parameter("general.camera_extrinsic_calibration").as_string();
    std::string image_topic = this->get_parameter("camera.image_topic").as_string();
    std::string confidence_topic = this->get_parameter("camera.confidence_topic").as_string();
    std::string lidar_topic = this->get_parameter("lidar.lidar_topic").as_string();
    std::string lidar_topic2 = this->get_parameter("lidar.lidar_topic2").as_string();
    lidar_frame_ = this->get_parameter("lidar.lidar_frame").as_string();
    lidar_frame2_ = this->get_parameter("lidar.lidar_frame2").as_string();
    std::string cloud_topic = this->get_parameter("lidar.colored_cloud_topic").as_string();
    horizontal_fov_deg_ = this->get_parameter("camera.horizontal_fov_deg").as_double();
    horizontal_fov_rad_ = horizontal_fov_deg_ * M_PI / 180.0;
    visualize_ = this->get_parameter("visualize").as_bool();

    // print calibration yaml file paths
    RCLCPP_INFO(this->get_logger(), "Camera calibration YAML: %s", cam_yaml.c_str());
    RCLCPP_INFO(this->get_logger(), "Camera extrinsic YAML: %s", ext_yaml.c_str());

    if (!ext_yaml.empty()) {
      if(!loadExtrinsicMatrix(ext_yaml)) {
        RCLCPP_ERROR(this->get_logger(), "Failed to load extrinsic matrix from YAML file.");
        rclcpp::shutdown();
      }
      is_extrinsic_loaded_ = true;
    }

    if (!loadCameraCalibration(cam_yaml)) {
      RCLCPP_ERROR(this->get_logger(), "Failed to load calibration YAML file.");
      rclcpp::shutdown();
    }

    image_pub_ = this->create_publisher<sensor_msgs::msg::Image>("/projected_image", 10);
    cloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(cloud_topic, 10);

    rclcpp::QoS qos_profile(rclcpp::QoSInitialization::from_rmw(rmw_qos_profile_sensor_data));
    qos_profile.keep_last(10).best_effort();

    image_sub_.subscribe(this, image_topic);
    confidence_sub_.subscribe(this, confidence_topic);
    lidar_sub_.subscribe(this, lidar_topic, qos_profile.get_rmw_qos_profile());

    if(!lidar_topic2.empty())
    {
      lidar_sub2_.subscribe(this, lidar_topic2, qos_profile.get_rmw_qos_profile());
      sync2_.reset(new message_filters::Synchronizer<SyncPolicy>(SyncPolicy(10), image_sub_, confidence_sub_, lidar_sub2_));
      sync2_->registerCallback(std::bind(&LidarCameraProjectionNode::syncCallback, this,
                                      std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));
    }
    else{
      is_extrinsic_loaded_2 = true;
    }

    sync_.reset(new message_filters::Synchronizer<SyncPolicy>(SyncPolicy(10), image_sub_, confidence_sub_, lidar_sub_));
  
    sync_->registerCallback(std::bind(&LidarCameraProjectionNode::syncCallback, this,
                                      std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));
    
  }

private:
  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::Image,
                                                          sensor_msgs::msg::Image,
                                                          sensor_msgs::msg::PointCloud2> SyncPolicy;

  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_pub_;

  message_filters::Subscriber<sensor_msgs::msg::Image> image_sub_;
  message_filters::Subscriber<sensor_msgs::msg::Image> confidence_sub_;
  message_filters::Subscriber<sensor_msgs::msg::PointCloud2> lidar_sub_;
  message_filters::Subscriber<sensor_msgs::msg::PointCloud2> lidar_sub2_;

  std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;
  std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync2_;

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  cv::Mat camera_matrix_;
  cv::Mat dist_coeffs_;
  Eigen::Matrix4d T_lidar_to_cam_;
  Eigen::Matrix4d T_lidar2_to_cam_;
  bool is_extrinsic_loaded_ = false;
  bool is_extrinsic_loaded_2 = false;


  std::string lidar_frame_;
  std::string lidar_frame2_;

  bool visualize_;

  double horizontal_fov_deg_;
  double horizontal_fov_rad_;

  bool loadCameraCalibration(const std::string& yaml_file) {
    try {
      YAML::Node calib = YAML::LoadFile(yaml_file);
      std::vector<double> cam_data;
      for (const auto& row : calib["camera_matrix"]["data"]) {
        for (const auto& val : row) {
          cam_data.push_back(val.as<double>());
        }
      }
      camera_matrix_ = cv::Mat(3, 3, CV_64F, cam_data.data()).clone();
      auto dist_data = calib["distortion_coefficients"]["data"].as<std::vector<double>>();
      dist_coeffs_ = cv::Mat(1, dist_data.size(), CV_64F, dist_data.data()).clone();
      return true;
    } catch (...) {
      return false;
    }
  }

  bool loadExtrinsicMatrix(const std::string& yaml_file) {
    try {
      YAML::Node node = YAML::LoadFile(yaml_file);
      auto matrix = node["extrinsic_matrix"].as<std::vector<std::vector<double>>>();
      Eigen::Matrix4d mat;
      for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
          mat(i, j) = matrix[i][j];
        }
      }
      T_lidar_to_cam_ = mat;
      return true;
    } catch (...) {
      return false;
    }
  }


  void syncCallback(const sensor_msgs::msg::Image::ConstSharedPtr& image_msg,
                  const sensor_msgs::msg::Image::ConstSharedPtr& confidence_msg,
                  const sensor_msgs::msg::PointCloud2::ConstSharedPtr& lidar_msg) {

    int lidar = lidar_frame_ == lidar_msg->header.frame_id ? 1 : 2;

    bool &loaded = lidar == 1 ? is_extrinsic_loaded_ : is_extrinsic_loaded_2;

    if (!loaded) {
      RCLCPP_WARN(this->get_logger(), "Extrinsic file not provided. Attempting to lookup transform via TF.");
      try {
        geometry_msgs::msg::TransformStamped tf =
          tf_buffer_.lookupTransform(image_msg->header.frame_id, lidar_msg->header.frame_id, tf2::TimePointZero);

        Eigen::Translation3d t(tf.transform.translation.x,
                              tf.transform.translation.y,
                              tf.transform.translation.z);
        Eigen::Quaterniond q(tf.transform.rotation.w,
                            tf.transform.rotation.x,
                            tf.transform.rotation.y,
                            tf.transform.rotation.z);

        if(lidar == 1)
        {
          T_lidar_to_cam_ = (t * q).matrix();
          is_extrinsic_loaded_ = true;
        }
        else
        {
          T_lidar2_to_cam_ = (t * q).matrix();
          is_extrinsic_loaded_2 = true;
        }
      } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Failed to lookup transform: %s", e.what());
        return;
      }
    }

    auto T_lidar_to_cam = lidar == 1 ? T_lidar_to_cam_ : T_lidar2_to_cam_;

    // get the camera image and confidence map
    cv::Mat image = cv_bridge::toCvCopy(image_msg, "bgr8")->image;
    cv::Mat confidence = cv_bridge::toCvCopy(confidence_msg, "32FC1")->image;

    sensor_msgs::PointCloud2ConstIterator<float> iter_x(*lidar_msg, "x");
    sensor_msgs::PointCloud2ConstIterator<float> iter_y(*lidar_msg, "y");
    sensor_msgs::PointCloud2ConstIterator<float> iter_z(*lidar_msg, "z");

    std::vector<cv::Point3f> lidar_points;
    for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z) {
      lidar_points.emplace_back(*iter_x, *iter_y, *iter_z);
    }

    std::vector<cv::Point3f> cam_points;
    for (const auto& pt : lidar_points) {
      Eigen::Vector4d pt_h(pt.x, pt.y, pt.z, 1.0);
      Eigen::Vector4d pt_cam = T_lidar_to_cam * pt_h;

      // Skip points behind the camera
      if (pt_cam.z() <= 0) continue;  

      // Filter out points outside the horizontal FOV
      double angle = std::atan2(pt_cam.x(), pt_cam.z());
      if (std::abs(angle) > horizontal_fov_rad_ / 2.0) continue;

      cam_points.emplace_back(pt_cam.x(), pt_cam.y(), pt_cam.z());
    }

    std::vector<cv::Point2f> projected_points;
    cv::projectPoints(cam_points, cv::Vec3d(0, 0, 0), cv::Vec3d(0, 0, 0), camera_matrix_, dist_coeffs_, projected_points);

    sensor_msgs::msg::PointCloud2 cloud_out;
    cloud_out.header = lidar_msg->header;
    cloud_out.header.frame_id = image_msg->header.frame_id;
    cloud_out.height = 1;
    cloud_out.width = 0;
    cloud_out.is_bigendian = false;
    cloud_out.is_dense = false;

    sensor_msgs::msg::PointField field_x;
    field_x.name = "x";
    field_x.offset = 0;
    field_x.datatype = sensor_msgs::msg::PointField::FLOAT32;
    field_x.count = 1;

    sensor_msgs::msg::PointField field_y;
    field_y.name = "y";
    field_y.offset = 4;
    field_y.datatype = sensor_msgs::msg::PointField::FLOAT32;
    field_y.count = 1;

    sensor_msgs::msg::PointField field_z;
    field_z.name = "z";
    field_z.offset = 8;
    field_z.datatype = sensor_msgs::msg::PointField::FLOAT32;
    field_z.count = 1;

    sensor_msgs::msg::PointField field_rgb;
    field_rgb.name = "rgb";
    field_rgb.offset = 12;
    field_rgb.datatype = sensor_msgs::msg::PointField::UINT32;
    field_rgb.count = 1;

    sensor_msgs::msg::PointField field_label;
    field_label.name = "label";
    field_label.offset = 16;
    field_label.datatype = sensor_msgs::msg::PointField::UINT32;
    field_label.count = 1;

    cloud_out.fields = {field_x, field_y, field_z, field_rgb, field_label};

    cloud_out.point_step = 20;

    std::vector<uint8_t> cloud_data;
    int valid_points = 0;
    for (size_t i = 0; i < projected_points.size(); ++i) {
      int u = static_cast<int>(std::round(projected_points[i].x));
      int v = static_cast<int>(std::round(projected_points[i].y));

      if (u >= 0 && u < image.cols && v >= 0 && v < image.rows) {
        cv::Vec3b bgr = image.at<cv::Vec3b>(v, u);
        float conf = confidence.at<float>(v, u);

        uint32_t rgb = (bgr[2] << 16) | (bgr[1] << 8) | bgr[0];
        uint32_t label = static_cast<uint32_t>(conf);

        float x = cam_points[i].x;
        float y = cam_points[i].y;
        float z = cam_points[i].z;

        uint8_t buffer[20];
        memcpy(buffer + 0, &x, 4);
        memcpy(buffer + 4, &y, 4);
        memcpy(buffer + 8, &z, 4);
        memcpy(buffer + 12, &rgb, 4);
        memcpy(buffer + 16, &label, 4);

        cloud_data.insert(cloud_data.end(), buffer, buffer + 20);
        valid_points++;
      }
    }

    if (valid_points == 0) return;
    cloud_out.data = std::move(cloud_data);
    cloud_out.width = valid_points;
    cloud_out.row_step = cloud_out.point_step * valid_points;

    cloud_pub_->publish(cloud_out);

    if (visualize_) {
      for (const auto& pt : projected_points) {
        int u = static_cast<int>(std::round(pt.x));
        int v = static_cast<int>(std::round(pt.y));
        if (u >= 0 && u < image.cols && v >= 0 && v < image.rows)
          cv::circle(image, cv::Point(u, v), 2, cv::Scalar(0, 255, 0), -1);
      }
      auto out_img = cv_bridge::CvImage(image_msg->header, "bgr8", image).toImageMsg();
      image_pub_->publish(*out_img);
    }
  }
};

int main(int argc, char** argv) {

  rclcpp::init(argc, argv);
  auto node = std::make_shared<LidarCameraProjectionNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}