#include "QuMatchUtils.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <system_error>

namespace qumatch
{

	namespace
	{

		const cv::Scalar kLeftKeypointColor(0, 0, 255);    // BGR: red
		const cv::Scalar kRightKeypointColor(0, 255, 255); // BGR: yellow
		const cv::Scalar kMatchLineColor(0, 255, 0);       // BGR: green

		// Convert a grayscale, BGR, or BGRA image to 8-bit BGR.
		cv::Mat toBgr8U(const cv::Mat& img)
		{
			if (img.empty())
			{
				return {};
			}

			cv::Mat bgr;
			if (img.channels() == 1)
			{
				cv::cvtColor(img, bgr, cv::COLOR_GRAY2BGR);
			}
			else if (img.channels() == 3)
			{
				bgr = img;
			}
			else if (img.channels() == 4)
			{
				cv::cvtColor(img, bgr, cv::COLOR_BGRA2BGR);
			}
			else
			{
				return {};
			}

			if (bgr.depth() != CV_8U)
			{
				double scale = 1.0;
				if (bgr.depth() == CV_16U)
				{
					scale = 1.0 / 257.0;
				}
				else if (bgr.depth() == CV_32F || bgr.depth() == CV_64F)
				{
					// Assume floating-point images are normalized to [0, 1].
					scale = 255.0;
				}
				else
				{
					return {};
				}

				cv::Mat bgr8;
				bgr.convertTo(bgr8, CV_8U, scale);
				bgr = bgr8;
			}

			return bgr.clone();
		}

		// Create the parent directory of an output path if it does not exist.
		bool ensureParentDirectory(const std::string& path)
		{
			const std::filesystem::path p(path);
			const std::filesystem::path parent = p.parent_path();
			if (parent.empty())
			{
				return true;
			}

			std::error_code ec;
			std::filesystem::create_directories(parent, ec);
			return !ec;
		}

	} // namespace

	// ============================================================================
	// Draw the matching result
	// ============================================================================
	cv::Mat drawColoredMatches(
		const cv::Mat& img1,
		const std::vector<cv::KeyPoint>& kp1,
		const cv::Mat& img2,
		const std::vector<cv::KeyPoint>& kp2,
		const std::vector<cv::DMatch>& matches)
	{
		cv::Mat left;
		cv::Mat right;
		try
		{
			left = toBgr8U(img1);
			right = toBgr8U(img2);
		}
		catch (const cv::Exception&)
		{
			return {};
		}
		if (left.empty() || right.empty())
		{
			return {};
		}

		constexpr int kGap = 10;
		const int height = std::max(left.rows, right.rows);
		const int width = left.cols + kGap + right.cols;
		const int offset_x = left.cols + kGap;

		cv::Mat canvas(height, width, CV_8UC3, cv::Scalar::all(0));
		left.copyTo(canvas(cv::Rect(0, 0, left.cols, left.rows)));
		right.copyTo(canvas(cv::Rect(offset_x, 0, right.cols, right.rows)));

		constexpr int kRadius = 3;
		constexpr int kLineThickness = 1;

		for (const cv::DMatch& m : matches)
		{
			if (m.queryIdx < 0 || m.queryIdx >= static_cast<int>(kp1.size()) ||
				m.trainIdx < 0 || m.trainIdx >= static_cast<int>(kp2.size()))
			{
				continue;
			}

			const cv::Point2f p1 = kp1[m.queryIdx].pt;
			const cv::Point2f p2 = kp2[m.trainIdx].pt +
				cv::Point2f(static_cast<float>(offset_x), 0.0f);

			// Draw the green connection first, then the keypoints on top
			cv::line(canvas, p1, p2, kMatchLineColor, kLineThickness,
				cv::LINE_AA);

			// Left keypoint: red
			cv::circle(canvas, p1, kRadius, kLeftKeypointColor, cv::FILLED,
				cv::LINE_AA);
			// Right keypoint: yellow
			cv::circle(canvas, p2, kRadius, kRightKeypointColor, cv::FILLED,
				cv::LINE_AA);
		}

		return canvas;
	}

	// ============================================================================
	// Save a match image
	// ============================================================================
	bool saveMatchImage(const std::string& path,
		const cv::Mat& img1, const std::vector<cv::KeyPoint>& kp1,
		const cv::Mat& img2, const std::vector<cv::KeyPoint>& kp2,
		const std::vector<cv::DMatch>& matches)
	{
		if (img1.empty() || img2.empty())
		{
			return false;
		}
		if (!ensureParentDirectory(path))
		{
			return false;
		}

		try
		{
			const cv::Mat canvas =
				drawColoredMatches(img1, kp1, img2, kp2, matches);
			if (canvas.empty())
			{
				return false;
			}
			return cv::imwrite(path, canvas);
		}
		catch (const cv::Exception&)
		{
			return false;
		}
	}

	// ============================================================================
	// Save text results
	// ============================================================================
	bool saveText(const std::string& path, const std::string& content)
	{
		if (!ensureParentDirectory(path))
		{
			return false;
		}

		std::ofstream out(path, std::ios::binary);
		if (!out)
		{
			return false;
		}

		out.write(content.data(), static_cast<std::streamsize>(content.size()));
		out.flush();
		return static_cast<bool>(out);
	}

} // namespace qumatch
