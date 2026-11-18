// ============================================================================
// QuMatch - end-to-end coherent L2 candidate selection + swap-test matching
// ============================================================================
#pragma once

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

#include <cstddef>
#include <string>
#include <vector>

namespace qumatch
{

	// ----------------------------------------------------------------------------
	// Matching parameters
	// ----------------------------------------------------------------------------
	struct MatchParams
	{
		int patch_size = 8;                 // local patch size in [2, 32]
		                                    // (8x8 -> 64 dims)
		std::size_t max_database = 32768;   // kNN prefilter database limit in
		                                    // (0, 2^20]: the first
		                                    // max_database keypoints of image 2
		float ratio_threshold = 0.80f;      // Lowe ratio-test threshold
		double min_ncc = 0.20;              // minimum NCC for an accepted match
		double min_swap_similarity = 0.10;  // swap-test threshold (NCC^2)

		// End-to-end coherent squared-L2 oracle over the top-K candidates.
		int candidates = 8;                 // power of two in [2, 256]
		int proj_dims = 4;                  // PCA dimensions
		int bits_per_dim = 2;               // quantization bits per dimension
		int grover_shots = 32;              // shots per coherent Grover search
		int max_searches = 8;               // Durr-Hoyer iteration limit
	};

	// ----------------------------------------------------------------------------
	// Matching statistics
	// ----------------------------------------------------------------------------
	struct MatchStatistics
	{
		std::size_t keypoints1 = 0;
		std::size_t keypoints2 = 0;
		std::size_t database_size = 0;

		std::size_t classical_candidates = 0;
		std::size_t selection_agreement = 0;
		std::size_t verified_matches = 0;
		std::size_t geometric_inliers = 0;

		double avg_sift_ncc = 0.0;
		double avg_selected_ncc = 0.0;

		double candidate_ms = 0.0;
		double selection_ms = 0.0;
		double swap_ms = 0.0;
		double total_ms = 0.0;

		std::size_t oracle_candidates = 0;
		int oracle_qubits = 0;
		std::size_t oracle_calls = 0;
		std::size_t search_queries = 0;
	};

	// ============================================================================
	// QuMatch
	//
	// Input : keypoints and descriptors detected in main()
	// Steps : Lowe ratio test -> top-K SIFT-L2 candidates ->
	//         coherent squared-L2 Grover/Durr-Hoyer selection ->
	//         swap-test verification -> RANSAC filtering
	// ============================================================================
	class QuMatch
	{
	public:
		explicit QuMatch(const MatchParams& params = MatchParams());

		// Run the matching pipeline.
		// img1 / img2 : 8-bit grayscale, BGR, or BGRA images.
		// kp1 / kp2   : keypoints, one entry per descriptor row.
		// desc1/desc2 : CV_32F descriptors with the same number of columns.
		//
		// The kNN prefilter searches the first max_database keypoints of
		// image 2. The coherent oracle then selects among the top candidates
		// for each query. Throws std::invalid_argument on invalid inputs or
		// parameters.
		void match(const cv::Mat& img1, const std::vector<cv::KeyPoint>& kp1,
			const cv::Mat& desc1,
			const cv::Mat& img2, const std::vector<cv::KeyPoint>& kp2,
			const cv::Mat& desc2);

		// Candidate matches from the Lowe ratio test. DMatch::distance is the
		// descriptor L2 distance.
		const std::vector<cv::DMatch>& classicalMatches() const noexcept
		{
			return classical_matches_;
		}

		// Matches selected by the coherent L2 oracle and verified by the
		// NCC/swap test. DMatch::distance stores 1 - swap-test similarity
		// (smaller is better).
		const std::vector<cv::DMatch>& verifiedMatches() const noexcept
		{
			return verified_matches_;
		}

		// RANSAC inliers of verifiedMatches(); the distance convention is the
		// same as verifiedMatches().
		const std::vector<cv::DMatch>& geometricInliers() const noexcept
		{
			return inlier_matches_;
		}

		const MatchStatistics& statistics() const noexcept
		{
			return stats_;
		}

		// Formatted statistics report
		std::string report() const;

		// Synthetic test of the coherent L2 oracle against the classical
		// minimum.
		static bool selfTest(int trials = 20);

	private:
		double swapTestSimilarity(
			const std::vector<double>& vecA,
			const std::vector<double>& vecB) const;

		static std::vector<double> extractPatchFeature(
			const cv::Mat& gray,
			const cv::KeyPoint& kp, int patch_size);

		static double dot(
			const std::vector<double>& a,
			const std::vector<double>& b);

		MatchParams params_;
		std::vector<cv::DMatch> classical_matches_;
		std::vector<cv::DMatch> verified_matches_;
		std::vector<cv::DMatch> inlier_matches_;
		MatchStatistics stats_;
	};

} // namespace qumatch
