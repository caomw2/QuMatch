// ============================================================================
// QuMatchUtils - result visualization and saving
// ============================================================================
#pragma once

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

#include <string>
#include <vector>

namespace qumatch
{

	// Draw the matching result (only matched keypoints are drawn):
	//   - matched keypoints in the left image  : red
	//   - matched keypoints in the right image : yellow
	//   - match connections                    : green
	//
	// img1 / img2 may be 8-bit grayscale, BGR, or BGRA images; 16-bit and
	// floating-point inputs are converted to 8-bit (floating-point images are
	// assumed to be in [0, 1]). Returns an empty Mat if either input is empty
	// or has an unsupported channel count or depth.
	cv::Mat drawColoredMatches(
		const cv::Mat& img1,
		const std::vector<cv::KeyPoint>& kp1,
		const cv::Mat& img2,
		const std::vector<cv::KeyPoint>& kp2,
		const std::vector<cv::DMatch>& matches);

	// Draw and save a match image (drawColoredMatches + cv::imwrite).
	// Parent directories are created as needed.
	bool saveMatchImage(
		const std::string& path,
		const cv::Mat& img1, const std::vector<cv::KeyPoint>& kp1,
		const cv::Mat& img2, const std::vector<cv::KeyPoint>& kp2,
		const std::vector<cv::DMatch>& matches);

	// Save text results (e.g. the statistics report).
	// Parent directories are created as needed.
	bool saveText(const std::string& path, const std::string& content);

} // namespace qumatch
