#include "QuMatch.h"

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <qpp/qpp.hpp>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <utility>

using namespace qpp;

namespace qumatch
{

	namespace
	{

		constexpr double kPi = 3.14159265358979323846;

		// Bilinear interpolation with clamped coordinates.
		double bilinearSample(const cv::Mat& img, double x, double y)
		{
			x = std::clamp(x, 0.0, static_cast<double>(img.cols - 1));
			y = std::clamp(y, 0.0, static_cast<double>(img.rows - 1));

			const int x0 = static_cast<int>(std::floor(x));
			const int y0 = static_cast<int>(std::floor(y));
			const int x1 = std::min(x0 + 1, img.cols - 1);
			const int y1 = std::min(y0 + 1, img.rows - 1);
			const double fx = x - x0;
			const double fy = y - y0;

			const double v00 = img.at<uchar>(y0, x0);
			const double v01 = img.at<uchar>(y0, x1);
			const double v10 = img.at<uchar>(y1, x0);
			const double v11 = img.at<uchar>(y1, x1);

			return (1.0 - fx) * (1.0 - fy) * v00 + fx * (1.0 - fy) * v01 +
				(1.0 - fx) * fy * v10 + fx * fy * v11;
		}

		// Convert an image to 8-bit grayscale if needed.
		cv::Mat toGray(const cv::Mat& img)
		{
			if (img.empty())
			{
				throw std::invalid_argument(
					"QuMatch::match() received an empty image");
			}

			cv::Mat gray;
			if (img.channels() == 1)
			{
				gray = img;
			}
			else if (img.channels() == 3)
			{
				cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
			}
			else if (img.channels() == 4)
			{
				cv::cvtColor(img, gray, cv::COLOR_BGRA2GRAY);
			}
			else
			{
				throw std::invalid_argument(
					"QuMatch::match() requires 1, 3, or 4 image channels");
			}

			if (gray.depth() != CV_8U)
			{
				throw std::invalid_argument(
					"QuMatch::match() requires 8-bit images");
			}

			return gray;
		}

		// Validate the matching parameters.
		void validateParams(const MatchParams& params)
		{
			if (params.patch_size < 2 || params.patch_size > 32)
			{
				throw std::invalid_argument(
					"MatchParams::patch_size must be in [2, 32]");
			}
			if (params.max_database == 0 ||
				params.max_database > (std::size_t(1) << 20))
			{
				throw std::invalid_argument(
					"MatchParams::max_database must be in (0, 2^20]");
			}
			if (!(params.ratio_threshold > 0.0f &&
				params.ratio_threshold <= 1.0f))
			{
				throw std::invalid_argument(
					"MatchParams::ratio_threshold must be in (0, 1]");
			}
			if (params.min_ncc < 0.0 || params.min_ncc > 1.0)
			{
				throw std::invalid_argument(
					"MatchParams::min_ncc must be in [0, 1]");
			}
			if (params.min_swap_similarity < 0.0 ||
				params.min_swap_similarity > 1.0)
			{
				throw std::invalid_argument(
					"MatchParams::min_swap_similarity must be in [0, 1]");
			}

			const int k = params.candidates;
			if (k < 2 || (k & (k - 1)) != 0 || k > 256)
			{
				throw std::invalid_argument(
					"MatchParams::candidates must be a power of two in "
					"[2, 256]");
			}
			if (params.proj_dims < 1 || params.proj_dims > 10 ||
				params.bits_per_dim < 1 || params.bits_per_dim > 10)
			{
				throw std::invalid_argument(
					"MatchParams::proj_dims and bits_per_dim must be in "
					"[1, 10]");
			}
			const int data_bits =
				params.proj_dims * params.bits_per_dim;
			if (data_bits < 2 || data_bits > 10)
			{
				throw std::invalid_argument(
					"MatchParams coherent descriptor must use 2..10 data "
					"qubits");
			}
			int index_bits = 0;
			while ((1 << index_bits) < k)
			{
				++index_bits;
			}
			if (index_bits + data_bits > 24)
			{
				throw std::invalid_argument(
					"MatchParams coherent oracle would need more than 24 "
					"simulated qubits");
			}
			if (params.grover_shots <= 0 || params.max_searches <= 0)
			{
				throw std::invalid_argument(
					"MatchParams::grover_shots and max_searches must be "
					"positive");
			}
		}

	} // namespace

	// ============================================================================
	// Coherent squared-L2 oracle (internal)
	//
	// The oracle is a real quantum circuit:
	//   - an index register selects one candidate neighbor,
	//   - multi-controlled X gates load that candidate's quantized descriptor,
	//   - a coherent truth-table circuit evaluates the squared L2 distance to
	//     the query and flips the phase when it is below the threshold,
	//   - the data register is uncomputed, leaving the phase on the index,
	//   - Durr-Hoyer minimum finding uses this oracle.
	//
	// SIFT uses Euclidean (L2) distance; the full 128-D float vector cannot be
	// basis-encoded in a state-vector simulator, so the oracle operates on a
	// PCA-reduced, uniformly quantized descriptor. Candidate distances are
	// evaluated classically only for the Durr-Hoyer threshold/iteration
	// schedule and stopping rule; the oracle circuit itself never encodes the
	// answer.
	// ============================================================================
	namespace
	{

		struct CoherentL2Params
		{
			int num_candidates = 8;   // power of two in [2, 256]
			int proj_dims = 4;        // PCA dimensions kept
			int bits_per_dim = 2;     // quantization levels per dimension
			int grover_shots = 32;    // measurement shots per Grover search
			int max_searches = 8;     // Durr-Hoyer iteration limit
		};

		struct CoherentL2Result
		{
			std::size_t index = 0;    // selected candidate index
			int oracle_calls = 0;     // number of Grover searches performed
			int qubits = 0;           // total simulated qubits
			bool success = false;     // selected result equals classical min
		};

		struct QuantizedDescriptors
		{
			std::vector<std::vector<int>> candidates; // K x (proj_dims*bits)
			std::vector<int> query;                   // proj_dims*bits
			int proj_dims = 0;
			int bits_per_dim = 0;
		};

		bool isPowerOfTwo(int x)
		{
			return x > 0 && (x & (x - 1)) == 0;
		}

		int log2Int(int x)
		{
			int n = 0;
			while ((1 << n) < x)
			{
				++n;
			}
			return n;
		}

		int bitsToValue(const std::vector<int>& bits, int offset,
			int n_bits)
		{
			int value = 0;
			for (int b = 0; b < n_bits; ++b)
			{
				value |= (bits[offset + b] & 1) << b;
			}
			return value;
		}

		int squaredDistance(const std::vector<int>& a,
			const std::vector<int>& b, int proj_dims, int bits_per_dim)
		{
			int sum = 0;
			for (int d = 0; d < proj_dims; ++d)
			{
				const int va = bitsToValue(a, d * bits_per_dim,
					bits_per_dim);
				const int vb = bitsToValue(b, d * bits_per_dim,
					bits_per_dim);
				const int diff = va - vb;
				sum += diff * diff;
			}
			return sum;
		}

		// Squared L2 distance between the query code and a data-register
		// pattern. The pattern's big-endian digits come from n2multiidx().
		int patternDistance(int pattern, const std::vector<int>& query,
			int num_bits, int proj_dims, int bits_per_dim)
		{
			const std::vector<idx> dims(num_bits, 2);
			const std::vector<idx> digits =
				n2multiidx(static_cast<idx>(pattern), dims);

			int sum = 0;
			for (int d = 0; d < proj_dims; ++d)
			{
				int vp = 0;
				int vq = 0;
				for (int b = 0; b < bits_per_dim; ++b)
				{
					const int k = d * bits_per_dim + b;
					vp |= static_cast<int>(digits[k] & 1) << b;
					vq |= (query[k] & 1) << b;
				}
				const int diff = vp - vq;
				sum += diff * diff;
			}
			return sum;
		}

		// Load candidate descriptors into the data register, controlled by the
		// index register: for each basis index i, the data becomes code[i].
		// With invert=true the same routine uncomputes the load.
		void applyDataLoad(ket& state,
			const std::vector<std::vector<int>>& codes,
			int num_index_bits, int num_data_bits,
			const std::vector<idx>& index_qubits,
			const std::vector<idx>& data_qubits,
			bool invert)
		{
			std::vector<std::pair<int, int>> ops;
			for (int i = 0; i < static_cast<int>(codes.size()); ++i)
			{
				for (int b = 0; b < num_data_bits; ++b)
				{
					if (codes[i][b] != 0)
					{
						ops.emplace_back(i, b);
					}
				}
			}
			if (invert)
			{
				std::reverse(ops.begin(), ops.end());
			}

			const std::vector<idx> dims_index(num_index_bits, 2);
			for (const auto& op : ops)
			{
				const int i = op.first;
				const int b = op.second;
				const std::vector<idx> digits =
					n2multiidx(static_cast<idx>(i), dims_index);

				// Mask |i> to |1...1> so the multi-controlled X fires.
				for (int j = 0; j < num_index_bits; ++j)
				{
					if (digits[j] == 0)
					{
						apply_inplace(state, gt.X, { index_qubits[j] });
					}
				}

				applyCTRL_inplace(state, gt.X, index_qubits,
					{ data_qubits[b] });

				for (int j = 0; j < num_index_bits; ++j)
				{
					if (digits[j] == 0)
					{
						apply_inplace(state, gt.X, { index_qubits[j] });
					}
				}
			}
		}

		// Phase-flip every data pattern whose squared L2 distance to the query
		// is strictly below the threshold. The table is a generic distance
		// function, not a database answer.
		void applyPhasePredicate(ket& state,
			const std::vector<int>& pattern_distance,
			int threshold, int num_data_bits,
			const std::vector<idx>& data_qubits)
		{
			const int num_patterns = 1 << num_data_bits;

			auto apply_mcz = [&]()
			{
				std::vector<idx> controls;
				for (int b = 1; b < num_data_bits; ++b)
				{
					controls.push_back(data_qubits[b]);
				}
				applyCTRL_inplace(state, gt.Z, controls,
					{ data_qubits[0] });
			};

			// Map pattern 0 to all-ones: apply X to every data qubit.
			for (int b = 0; b < num_data_bits; ++b)
			{
				apply_inplace(state, gt.X, { data_qubits[b] });
			}

			if (pattern_distance[0] < threshold)
			{
				apply_mcz();
			}

			// Walk all patterns in Gray-code order: consecutive patterns
			// differ by one data qubit, so one X toggle per step suffices.
			int previous_gray = 0;
			for (int i = 1; i < num_patterns; ++i)
			{
				const int gray = i ^ (i >> 1);
				const int changed = gray ^ previous_gray;
				int low_bit = 0;
				while (((changed >> low_bit) & 1) == 0)
				{
					++low_bit;
				}
				// Decimal bits are big-endian in the data-register digits.
				const int data_bit = num_data_bits - 1 - low_bit;
				apply_inplace(state, gt.X, { data_qubits[data_bit] });

				if (pattern_distance[gray] < threshold)
				{
					apply_mcz();
				}
				previous_gray = gray;
			}

			// Remove the current X mask.
			const std::vector<idx> dims(num_data_bits, 2);
			const std::vector<idx> digits =
				n2multiidx(static_cast<idx>(previous_gray), dims);
			for (int b = 0; b < num_data_bits; ++b)
			{
				if (digits[b] == 0)
				{
					apply_inplace(state, gt.X, { data_qubits[b] });
				}
			}
		}

		// Grover diffusion on the index register. The data register is |0>
		// because the oracle uncomputes its load.
		void applyDiffusion(ket& state, int num_index_bits,
			const std::vector<idx>& index_qubits)
		{
			for (idx q : index_qubits)
			{
				apply_inplace(state, gt.H, { q });
			}

			ket reflection = ket::Ones(static_cast<idx>(1) <<
				num_index_bits);
			reflection[0] = -1.0;
			apply_diag_inplace(state, reflection, index_qubits);

			for (idx q : index_qubits)
			{
				apply_inplace(state, gt.H, { q });
			}
		}

		// Build the oracle state, apply the Grover iterations, and measure the
		// index register.
		std::size_t runGroverThreshold(
			const std::vector<std::vector<int>>& codes,
			const std::vector<int>& pattern_distance,
			int threshold, int iterations, int shots,
			int num_index_bits, int num_data_bits)
		{
			const idx total_qubits =
				static_cast<idx>(num_index_bits + num_data_bits);
			const std::size_t num_candidates = codes.size();

			ket state = st.zero(total_qubits);

			std::vector<idx> index_qubits(num_index_bits);
			std::vector<idx> data_qubits(num_data_bits);
			std::iota(index_qubits.begin(), index_qubits.end(), idx(0));
			std::iota(data_qubits.begin(), data_qubits.end(),
				static_cast<idx>(num_index_bits));

			for (idx q : index_qubits)
			{
				apply_inplace(state, gt.H, { q });
			}

			for (int k = 0; k < iterations; ++k)
			{
				applyDataLoad(state, codes, num_index_bits, num_data_bits,
					index_qubits, data_qubits, false);
				applyPhasePredicate(state, pattern_distance, threshold,
					num_data_bits, data_qubits);
				applyDataLoad(state, codes, num_index_bits, num_data_bits,
					index_qubits, data_qubits, true);
				applyDiffusion(state, num_index_bits, index_qubits);
			}

			const std::vector<idx> dims_all(total_qubits, 2);
			const std::vector<idx> dims_index(num_index_bits, 2);
			std::vector<int> counts(num_candidates, 0);
			for (int s = 0; s < shots; ++s)
			{
				const auto sampled =
					sample(state, index_qubits, dims_all);
				const idx r = multiidx2n(std::get<0>(sampled),
					dims_index);
				++counts[r];
			}

			return static_cast<std::size_t>(
				std::max_element(counts.begin(), counts.end()) -
				counts.begin());
		}

		// PCA-project the query and candidates, then uniformly quantize each
		// projected dimension to [0, 2^bits_per_dim - 1].
		QuantizedDescriptors projectAndQuantize(
			const cv::Mat& query_desc,
			const cv::Mat& candidate_descs,
			int proj_dims, int bits_per_dim)
		{
			if (query_desc.empty() || candidate_descs.empty())
			{
				throw std::invalid_argument(
					"projectAndQuantize(): empty descriptor matrix");
			}
			if (query_desc.type() != CV_32F ||
				candidate_descs.type() != CV_32F)
			{
				throw std::invalid_argument(
					"projectAndQuantize(): descriptors must be CV_32F");
			}
			if (query_desc.rows != 1 ||
				query_desc.cols != candidate_descs.cols)
			{
				throw std::invalid_argument(
					"projectAndQuantize(): query must be one row with "
					"the same descriptor dimension");
			}
			if (proj_dims < 1 || proj_dims > 10 ||
				bits_per_dim < 1 || bits_per_dim > 10)
			{
				throw std::invalid_argument(
					"projectAndQuantize(): projection and quantization "
					"must be in [1, 10]");
			}
			const int num_bits = proj_dims * bits_per_dim;
			if (num_bits < 2 || num_bits > 10)
			{
				throw std::invalid_argument(
					"projectAndQuantize(): data register must have 2..10 "
					"bits");
			}

			const int num_candidates = candidate_descs.rows;
			if (num_candidates < 2)
			{
				throw std::invalid_argument(
					"projectAndQuantize(): at least two candidates are "
					"required");
			}

			cv::Mat data(num_candidates + 1, query_desc.cols, CV_32F);
			query_desc.copyTo(data.row(0));
			candidate_descs.copyTo(
				data.rowRange(1, num_candidates + 1));

			// PCA on the query + candidate set. At most num_candidates
			// components are available from num_candidates + 1 samples.
			const int max_components = std::min(proj_dims,
				std::min(query_desc.cols, num_candidates));
			cv::PCA pca(data, cv::noArray(), cv::PCA::DATA_AS_ROW,
				max_components);
			const cv::Mat projected = pca.project(data);

			QuantizedDescriptors result;
			result.proj_dims = proj_dims;
			result.bits_per_dim = bits_per_dim;
			const int levels = 1 << bits_per_dim;

			result.query.assign(num_bits, 0);
			result.candidates.assign(num_candidates,
				std::vector<int>(num_bits, 0));

			for (int d = 0; d < proj_dims; ++d)
			{
				float vmin = 0.0f;
				float vmax = 0.0f;
				if (d < projected.cols)
				{
					double lo = 0.0;
					double hi = 0.0;
					cv::minMaxLoc(projected.col(d), &lo, &hi);
					vmin = static_cast<float>(lo);
					vmax = static_cast<float>(hi);
				}

				const float span = (vmax - vmin) > 1e-12f
					? (vmax - vmin)
					: 1.0f;

				for (int row = 0; row < num_candidates + 1; ++row)
				{
					const float value = d < projected.cols
						? projected.at<float>(row, d)
						: 0.0f;
					int code = static_cast<int>(std::lround(
						(value - vmin) / span * (levels - 1)));
					code = std::clamp(code, 0, levels - 1);

					std::vector<int>& dst = row == 0
						? result.query
						: result.candidates[row - 1];
					for (int b = 0; b < bits_per_dim; ++b)
					{
						dst[d * bits_per_dim + b] = (code >> b) & 1;
					}
				}
			}

			return result;
		}

		// Durr-Hoyer minimum finding over the quantized squared-L2 oracle.
		CoherentL2Result coherentGroverArgMin(
			const std::vector<std::vector<int>>& candidate_bits,
			const std::vector<int>& query_bits,
			const CoherentL2Params& params)
		{
			const int num_candidates =
				static_cast<int>(candidate_bits.size());
			if (!isPowerOfTwo(num_candidates) || num_candidates < 2 ||
				num_candidates > 256)
			{
				throw std::invalid_argument(
					"coherentGroverArgMin(): candidate count must be a "
					"power of two in [2, 256]");
			}
			if (params.proj_dims < 1 || params.proj_dims > 10 ||
				params.bits_per_dim < 1 || params.bits_per_dim > 10)
			{
				throw std::invalid_argument(
					"coherentGroverArgMin(): projection and quantization "
					"must be in [1, 10]");
			}
			if (params.num_candidates != num_candidates)
			{
				throw std::invalid_argument(
					"coherentGroverArgMin(): num_candidates does not match "
					"the candidate list");
			}

			const int num_data_bits =
				params.proj_dims * params.bits_per_dim;
			if (num_data_bits < 2 || num_data_bits > 10)
			{
				throw std::invalid_argument(
					"coherentGroverArgMin(): data register must have "
					"2..10 qubits for the truth-table oracle");
			}
			if (static_cast<int>(query_bits.size()) != num_data_bits)
			{
				throw std::invalid_argument(
					"coherentGroverArgMin(): query descriptor bit width "
					"mismatch");
			}
			for (const auto& code : candidate_bits)
			{
				if (static_cast<int>(code.size()) != num_data_bits)
				{
					throw std::invalid_argument(
						"coherentGroverArgMin(): candidate bit width "
						"mismatch");
				}
			}

			const int num_index_bits = log2Int(num_candidates);
			const int total_qubits = num_index_bits + num_data_bits;
			if (total_qubits > 24)
			{
				throw std::invalid_argument(
					"coherentGroverArgMin(): state-vector simulation "
					"would need too many qubits");
			}

			const int shots = std::max(1, params.grover_shots);
			const int max_searches = std::max(1, params.max_searches);

			RandomDevices::get_instance().get_prng().seed(12345);

			// Generic distance table over all quantized data patterns.
			const int num_patterns = 1 << num_data_bits;
			std::vector<int> pattern_distance(num_patterns, 0);
			for (int p = 0; p < num_patterns; ++p)
			{
				pattern_distance[p] = patternDistance(p, query_bits,
					num_data_bits, params.proj_dims,
					params.bits_per_dim);
			}

			// Candidate distances are used only for threshold bookkeeping
			// and the Durr-Hoyer iteration schedule; the oracle never uses
			// them.
			std::vector<int> candidate_distance(num_candidates, 0);
			for (int i = 0; i < num_candidates; ++i)
			{
				candidate_distance[i] = squaredDistance(
					candidate_bits[i], query_bits, params.proj_dims,
					params.bits_per_dim);
			}

			int best_index = 0;
			int best_distance = candidate_distance[0];
			CoherentL2Result result;
			result.qubits = total_qubits;

			for (int search = 0; search < max_searches; ++search)
			{
				int num_marked = 0;
				for (int i = 0; i < num_candidates; ++i)
				{
					if (candidate_distance[i] < best_distance)
					{
						++num_marked;
					}
				}
				if (num_marked == 0)
				{
					break; // the current best is the minimum
				}

				// When most candidates are marked, zero Grover iterations
				// (uniform superposition) is optimal.
				const int iterations = std::max(0,
					static_cast<int>(std::floor(kPi / 4.0 *
						std::sqrt(static_cast<double>(num_candidates) /
							num_marked))));

				const std::size_t measured = runGroverThreshold(
					candidate_bits, pattern_distance, best_distance,
					iterations, shots, num_index_bits, num_data_bits);
				++result.oracle_calls;

				if (candidate_distance[measured] < best_distance)
				{
					best_distance = candidate_distance[measured];
					best_index = static_cast<int>(measured);
				}
			}

			result.index = static_cast<std::size_t>(best_index);
			result.success = best_distance ==
				*std::min_element(candidate_distance.begin(),
					candidate_distance.end());
			return result;
		}

	} // namespace

	// ============================================================================
	// Synthetic self-test
	// ============================================================================
	bool QuMatch::selfTest(int trials)
	{
		if (trials <= 0)
		{
			return false;
		}

		// Use the class defaults so the self-test tracks MatchParams.
		const MatchParams defaults;
		CoherentL2Params params;
		params.num_candidates = defaults.candidates;
		params.proj_dims = defaults.proj_dims;
		params.bits_per_dim = defaults.bits_per_dim;
		params.grover_shots = defaults.grover_shots;
		params.max_searches = defaults.max_searches;

		const int num_candidates = params.num_candidates;
		const int num_bits = params.proj_dims * params.bits_per_dim;

		std::mt19937 rng(12345);
		std::uniform_int_distribution<int> bit(0, 1);

		int passed = 0;
		for (int t = 0; t < trials; ++t)
		{
			std::vector<std::vector<int>> codes(
				num_candidates, std::vector<int>(num_bits));
			std::vector<int> query(num_bits);
			for (auto& code : codes)
			{
				for (int& b : code)
				{
					b = bit(rng);
				}
			}
			for (int& b : query)
			{
				b = bit(rng);
			}

			std::vector<int> distances(num_candidates);
			for (int i = 0; i < num_candidates; ++i)
			{
				distances[i] = squaredDistance(codes[i], query,
					params.proj_dims, params.bits_per_dim);
			}
			const int classical_best = static_cast<int>(
				std::min_element(distances.begin(), distances.end()) -
				distances.begin());

			const CoherentL2Result res =
				coherentGroverArgMin(codes, query, params);
			const int selected_distance =
				distances[static_cast<std::size_t>(res.index)];
			const int classical_distance = distances[classical_best];
			if (selected_distance == classical_distance)
			{
				++passed;
			}
		}

		std::cout << "Coherent L2 oracle self-test: " << passed << "/"
			<< trials << " trials matched the classical minimum"
			<< std::endl;
		return passed == trials;
	}

	// ============================================================================
	// Construction
	// ============================================================================
	QuMatch::QuMatch(const MatchParams& params) : params_(params) {}

	// ============================================================================
	// Main matching pipeline
	// ============================================================================
	void QuMatch::match(
		const cv::Mat& img1,
		const std::vector<cv::KeyPoint>& kp1,
		const cv::Mat& desc1,
		const cv::Mat& img2,
		const std::vector<cv::KeyPoint>& kp2,
		const cv::Mat& desc2)
	{
		classical_matches_.clear();
		verified_matches_.clear();
		inlier_matches_.clear();
		stats_ = MatchStatistics{};
		stats_.keypoints1 = kp1.size();
		stats_.keypoints2 = kp2.size();
		stats_.oracle_candidates =
			static_cast<std::size_t>(params_.candidates);

		validateParams(params_);

		if (kp1.empty() || kp2.empty())
		{
			return; // no features to match
		}
		if (desc1.empty() || desc2.empty())
		{
			throw std::invalid_argument(
				"QuMatch::match() received empty descriptors");
		}

		if (desc1.type() != CV_32F || desc2.type() != CV_32F)
		{
			throw std::invalid_argument(
				"QuMatch::match() requires CV_32F descriptors (e.g. SIFT)");
		}
		if (desc1.rows != static_cast<int>(kp1.size()) ||
			desc2.rows != static_cast<int>(kp2.size()))
		{
			throw std::invalid_argument(
				"QuMatch::match() requires one descriptor row per keypoint");
		}
		if (desc1.cols != desc2.cols)
		{
			throw std::invalid_argument(
				"QuMatch::match() requires descriptors with equal dimensions");
		}

		cv::setRNGSeed(12345); // make RANSAC reproducible
		const auto t_start = std::chrono::steady_clock::now();
		auto elapsed_ms = [](auto a, auto b) {
			return std::chrono::duration<double, std::milli>(b - a).count();
			};

		// ------------------------------------------------------------------
		// kNN prefilter database: the first max_database keypoints of image 2
		// ------------------------------------------------------------------
		const std::size_t db_size =
			std::min<std::size_t>(kp2.size(), params_.max_database);
		stats_.database_size = db_size;
		if (db_size == 0)
		{
			return;
		}

		const cv::Mat gray1 = toGray(img1);
		const cv::Mat gray2 = toGray(img2);

		// Database patch features are computed lazily, only for the entries
		// actually selected by the coherent oracle.
		std::vector<std::vector<double>> db_features(db_size);
		std::vector<char> db_feature_ready(db_size, 0);
		auto get_db_feature = [&](std::size_t index) -> const std::vector<double>&
		{
			if (!db_feature_ready[index])
			{
				db_features[index] =
					QuMatch::extractPatchFeature(gray2, kp2[index],
						params_.patch_size);
				db_feature_ready[index] = 1;
			}
			return db_features[index];
		};

		// ------------------------------------------------------------------
		// Stage 1: Lowe ratio test -> candidate queries
		// BFMatcher keeps the top-K SIFT-L2 candidates for the coherent
		// oracle; the ratio test uses the first two.
		// ------------------------------------------------------------------
		const cv::Mat db_desc = desc2.rowRange(0, static_cast<int>(db_size));
		cv::BFMatcher matcher(cv::NORM_L2);
		std::vector<std::vector<cv::DMatch>> knn_matches;
		matcher.knnMatch(desc1, db_desc, knn_matches, params_.candidates);

		for (const auto& m : knn_matches)
		{
			if (m.size() >= 2 &&
				m[0].distance < params_.ratio_threshold * m[1].distance)
			{
				classical_matches_.push_back(m[0]);
			}
		}

		// Cache patch features only for candidate query keypoints.
		std::vector<std::vector<double>> query_features(kp1.size());
		std::vector<char> query_feature_needed(kp1.size(), 0);
		for (const cv::DMatch& m : classical_matches_)
		{
			query_feature_needed[static_cast<std::size_t>(m.queryIdx)] = 1;
		}
		for (std::size_t i = 0; i < kp1.size(); ++i)
		{
			if (query_feature_needed[i])
			{
				query_features[i] =
					extractPatchFeature(gray1, kp1[i], params_.patch_size);
			}
		}

		stats_.classical_candidates = classical_matches_.size();
		stats_.candidate_ms =
			elapsed_ms(t_start, std::chrono::steady_clock::now());

		// ------------------------------------------------------------------
		// Stage 2 & 3: coherent L2 selection + swap-test verification
		// ------------------------------------------------------------------
		const std::size_t n_candidates = classical_matches_.size();
		std::vector<std::size_t> selected_train(n_candidates, 0);
		std::vector<double> selected_ncc(n_candidates, 0.0);
		std::vector<double> selected_sim(n_candidates, 0.0);
		std::vector<double> sift_ncc(n_candidates, 0.0);

		const int k = params_.candidates;
		CoherentL2Params coherent_params;
		coherent_params.num_candidates = k;
		coherent_params.proj_dims = params_.proj_dims;
		coherent_params.bits_per_dim = params_.bits_per_dim;
		coherent_params.grover_shots = params_.grover_shots;
		coherent_params.max_searches = params_.max_searches;

		for (std::size_t ci = 0; ci < n_candidates; ++ci)
		{
			const cv::DMatch& qm = classical_matches_[ci];
			const std::vector<double>& qfeat = query_features[qm.queryIdx];

			const auto ts0 = std::chrono::steady_clock::now();

			const std::vector<cv::DMatch>& query_matches =
				knn_matches[static_cast<std::size_t>(qm.queryIdx)];
			if (query_matches.empty())
			{
				throw std::runtime_error(
					"QuMatch::match() received an empty kNN list");
			}

			// Pad to exactly k candidates by repeating the last match.
			std::vector<cv::DMatch> candidate_matches(k);
			for (int j = 0; j < k; ++j)
			{
				const std::size_t src = std::min<std::size_t>(
					static_cast<std::size_t>(j),
					query_matches.size() - 1);
				candidate_matches[j] = query_matches[src];
			}

			cv::Mat candidate_desc(k, desc2.cols, CV_32F);
			for (int j = 0; j < k; ++j)
			{
				desc2.row(candidate_matches[j].trainIdx)
					.copyTo(candidate_desc.row(j));
			}
			const cv::Mat query_desc = desc1.row(qm.queryIdx);

			const QuantizedDescriptors quantized =
				projectAndQuantize(query_desc, candidate_desc,
					params_.proj_dims, params_.bits_per_dim);

			const CoherentL2Result coherent_result =
				coherentGroverArgMin(quantized.candidates,
					quantized.query, coherent_params);

			selected_train[ci] = static_cast<std::size_t>(
				candidate_matches[coherent_result.index].trainIdx);

			stats_.oracle_qubits = coherent_result.qubits;
			stats_.oracle_calls +=
				static_cast<std::size_t>(coherent_result.oracle_calls);
			if (coherent_result.oracle_calls > 0)
			{
				++stats_.search_queries;
			}

			stats_.selection_ms +=
				elapsed_ms(ts0, std::chrono::steady_clock::now());

			sift_ncc[ci] = dot(qfeat,
				get_db_feature(static_cast<std::size_t>(qm.trainIdx)));
			selected_ncc[ci] = dot(qfeat,
				get_db_feature(selected_train[ci]));

			// Swap-test verification
			const auto tw0 = std::chrono::steady_clock::now();
			selected_sim[ci] = swapTestSimilarity(qfeat,
				get_db_feature(selected_train[ci]));
			stats_.swap_ms +=
				elapsed_ms(tw0, std::chrono::steady_clock::now());

#ifndef NDEBUG
			// The swap test must reproduce |<a|b>|^2 (NCC^2) exactly for
			// unit-norm patches. This catches circuit/indexing regressions.
			const double expected_sim =
				selected_ncc[ci] * selected_ncc[ci];
			assert(std::abs(selected_sim[ci] - expected_sim) < 1e-9);
#endif
		}

		for (std::size_t ci = 0; ci < n_candidates; ++ci)
		{
			if (selected_train[ci] ==
				static_cast<std::size_t>(classical_matches_[ci].trainIdx))
			{
				++stats_.selection_agreement;
			}
		}

		// ------------------------------------------------------------------
		// Accept matches: NCC and swap-test similarity must pass. Keep only
		// the best match for each train keypoint.
		// ------------------------------------------------------------------
		std::vector<int> best_query_for_train(db_size, -1);
		std::vector<double> best_sim_for_train(db_size, -1.0);

		for (std::size_t ci = 0; ci < n_candidates; ++ci)
		{
			if (selected_ncc[ci] < params_.min_ncc ||
				selected_sim[ci] < params_.min_swap_similarity)
			{
				continue;
			}

			const std::size_t train_idx = selected_train[ci];
			if (selected_sim[ci] > best_sim_for_train[train_idx])
			{
				best_sim_for_train[train_idx] = selected_sim[ci];
				best_query_for_train[train_idx] =
					classical_matches_[ci].queryIdx;
			}
		}

		for (std::size_t t = 0; t < db_size; ++t)
		{
			if (best_query_for_train[t] >= 0)
			{
				// DMatch uses a distance, so store 1 - similarity.
				verified_matches_.emplace_back(best_query_for_train[t],
					static_cast<int>(t),
					static_cast<float>(1.0 - best_sim_for_train[t]));
			}
		}

		std::sort(verified_matches_.begin(), verified_matches_.end(),
			[](const cv::DMatch& a, const cv::DMatch& b) {
				return a.distance < b.distance; // higher similarity first
			});
		stats_.verified_matches = verified_matches_.size();

		// ------------------------------------------------------------------
		// RANSAC geometric consistency filter
		// ------------------------------------------------------------------
		if (verified_matches_.size() >= 4)
		{
			std::vector<cv::Point2f> src, dst;
			src.reserve(verified_matches_.size());
			dst.reserve(verified_matches_.size());
			for (const cv::DMatch& m : verified_matches_)
			{
				src.push_back(kp1[m.queryIdx].pt);
				dst.push_back(kp2[m.trainIdx].pt);
			}

			std::vector<char> mask;
			cv::Mat H = cv::findHomography(src, dst, cv::RANSAC, 5.0, mask);
			if (!H.empty())
			{
				for (std::size_t i = 0; i < verified_matches_.size(); ++i)
				{
					if (mask[i])
					{
						inlier_matches_.push_back(verified_matches_[i]);
					}
				}
			}
		}
		stats_.geometric_inliers = inlier_matches_.size();

		// ------------------------------------------------------------------
		// Average NCC
		// ------------------------------------------------------------------
		double sum_sift_ncc = 0.0;
		double sum_selected_ncc = 0.0;
		for (std::size_t ci = 0; ci < n_candidates; ++ci)
		{
			sum_sift_ncc += sift_ncc[ci];
			sum_selected_ncc += selected_ncc[ci];
		}
		if (n_candidates > 0)
		{
			stats_.avg_sift_ncc = sum_sift_ncc / n_candidates;
			stats_.avg_selected_ncc = sum_selected_ncc / n_candidates;
		}
		stats_.total_ms = elapsed_ms(t_start, std::chrono::steady_clock::now());
	}

	// ============================================================================
	// Swap test
	//
	//   |0>_anc |psi>_A |phi>_B
	//     --H(anc)--> CSWAP(anc, A_i, B_i) --H(anc)--> measure anc
	//
	//   P(anc = 0) = (1 + |<psi|phi>|^2) / 2
	//
	// P(0) is computed exactly from the simulated QPP state, and the function
	// returns |<psi|phi>|^2 (i.e. NCC^2).
	// ============================================================================
	double QuMatch::swapTestSimilarity(
		const std::vector<double>& vecA,
		const std::vector<double>& vecB) const
	{
		if (vecA.empty() || vecA.size() != vecB.size())
		{
			return 0.0;
		}

		idx n_qubits = 0;
		while ((idx(1) << n_qubits) < vecA.size())
		{
			++n_qubits;
		}

		const idx state_dim = idx(1) << n_qubits;

		std::vector<double> a(state_dim, 0.0), b(state_dim, 0.0);
		for (idx i = 0; i < vecA.size(); ++i)
		{
			a[i] = vecA[i];
			b[i] = vecB[i];
		}

		double normA = 0.0, normB = 0.0;
		for (double x : a)
		{
			normA += x * x;
		}

		for (double x : b)
		{
			normB += x * x;
		}

		if (normA < 1e-12 || normB < 1e-12)
		{
			return 0.0;
		}

		for (double& x : a)
		{
			x /= std::sqrt(normA);
		}

		for (double& x : b)
		{
			x /= std::sqrt(normB);
		}

		// Amplitude encoding
		ket psi(state_dim);
		ket phi(state_dim);
		for (idx i = 0; i < state_dim; ++i)
		{
			psi[i] = a[i];
			phi[i] = b[i];
		}

		// |0>_anc (x) |psi>_A (x) |phi>_B, ancilla is subsystem 0
		ket state = kron(st.z0, kron(psi, phi));

		apply_inplace(state, gt.H, { 0 });

		// Controlled SWAP: the ancilla controls the swap of A and B qubit pairs
		for (idx i = 0; i < n_qubits; ++i)
		{
			applyCTRL_inplace(state, gt.SWAP, { 0 },
				{ 1 + i, 1 + n_qubits + i });
		}

		apply_inplace(state, gt.H, { 0 });

		// With the kron ordering, ancilla = 0 occupies the first half
		const idx half = state_dim * state_dim;
		double p0 = 0.0;
		for (idx i = 0; i < half; ++i)
		{
			p0 += std::norm(state[i]);
		}

		return std::clamp(2.0 * p0 - 1.0, 0.0, 1.0);
	}

	// ============================================================================
	// Sample a patch_size x patch_size patch centered at the keypoint, oriented
	// and scaled according to the keypoint, then normalize it to zero mean and
	// unit norm. The inner product of two such patches is the normalized
	// cross-correlation (NCC), which can be directly used for amplitude encoding.
	// ============================================================================
	std::vector<double> QuMatch::extractPatchFeature(
		const cv::Mat& gray,
		const cv::KeyPoint& kp, int patch_size)
	{
		std::vector<double> feat(patch_size * patch_size, 0.0);

		const double angle = kp.angle * kPi / 180.0;
		const double ca = std::cos(angle);
		const double sa = std::sin(angle);
		// Follow the keypoint scale so that the patch is scale invariant
		const double side = std::max(8.0, static_cast<double>(kp.size));

		for (int r = 0; r < patch_size; ++r)
		{
			for (int c = 0; c < patch_size; ++c)
			{
				const double u = ((c + 0.5) / patch_size - 0.5) * side;
				const double v = ((r + 0.5) / patch_size - 0.5) * side;
				const double x = kp.pt.x + ca * u - sa * v;
				const double y = kp.pt.y + sa * u + ca * v;
				feat[r * patch_size + c] = bilinearSample(gray, x, y) / 255.0;
			}
		}

		const double mean =
			std::accumulate(feat.begin(), feat.end(), 0.0) / feat.size();
		double norm = 0.0;
		for (double& v : feat)
		{
			v -= mean;
			norm += v * v;
		}

		norm = std::sqrt(norm);
		if (norm < 1e-12) { // flat region, no usable structure
			std::fill(feat.begin(), feat.end(), 0.0);
			return feat;
		}

		for (double& v : feat)
		{
			v /= norm;
		}

		return feat;
	}

	double QuMatch::dot(
		const std::vector<double>& a,
		const std::vector<double>& b)
	{
		double s = 0.0;
		for (std::size_t i = 0; i < a.size(); ++i)
		{
			s += a[i] * b[i];
		}

		return s;
	}

	// ============================================================================
	// Statistics report
	// ============================================================================
	std::string QuMatch::report() const
	{
		std::ostringstream os;
		os << "===== QuMatch statistics =====\n";
		os << std::fixed << std::setprecision(2);
		os << "Parameters: patch " << params_.patch_size << "x"
			<< params_.patch_size << ", ratio " << params_.ratio_threshold
			<< ", min NCC " << params_.min_ncc << ", min swap similarity "
			<< params_.min_swap_similarity << ", candidates "
			<< params_.candidates << ", PCA dims " << params_.proj_dims
			<< ", bits/dim " << params_.bits_per_dim << ", Grover shots "
			<< params_.grover_shots << ", max searches "
			<< params_.max_searches << "\n";

		os << "Keypoints: image1 = " << stats_.keypoints1
			<< ", image2 = " << stats_.keypoints2 << "\n";
		os << "Database (kNN prefilter): " << stats_.database_size
			<< " / " << stats_.keypoints2 << " keypoints of image2\n";
		os << "Selection: coherent L2 oracle over "
			<< stats_.oracle_candidates << " candidates ("
			<< stats_.oracle_qubits << " qubits, "
			<< stats_.oracle_calls << " Grover searches over "
			<< stats_.search_queries << " queries)\n";
		os << "Classical ratio-test candidates (in database): "
			<< stats_.classical_candidates << "\n";
		os << "Selection agreement with SIFT nearest (in database): "
			<< stats_.selection_agreement << " / "
			<< stats_.classical_candidates << "\n";
		os << "Verified matches (NCC + swap test): "
			<< stats_.verified_matches << "\n";
		os << "Geometric inliers (RANSAC): " << stats_.geometric_inliers
			<< " / " << stats_.verified_matches << "\n";
		os << std::setprecision(4);
		os << "Average patch NCC over candidates: SIFT nearest = "
			<< stats_.avg_sift_ncc << ", selected = "
			<< stats_.avg_selected_ncc << "\n";
		os << "Timing: feature prep + ratio test " << std::setprecision(1)
			<< stats_.candidate_ms << " ms, selection "
			<< stats_.selection_ms << " ms, swap test " << stats_.swap_ms
			<< " ms, total matching (excl. SIFT) " << stats_.total_ms
			<< " ms\n";

		return os.str();
	}

} // namespace qumatch
