# QuMatch

QuMatch: Image Matching with **Quantum Grover Search and Swap Tes**t, built with [OpenCV](https://opencv.org/),
[Quantum++ (QPP)](https://github.com/softwareQinc/qpp), and Eigen.

## Code structure

| File | Responsibility |
| --- | --- |
| `main.cpp` | Loads color images, detects SIFT keypoints / computes descriptors, runs `QuMatch`, and saves results through `QuMatchUtils` |
| `QuMatch.h/.cpp` | `QuMatch` class: kNN prefilter, PCA/quantization, end-to-end coherent L2 oracle (multi-controlled data load, Gray-code truth-table phase oracle, Durr-Hoyer search), swap test, RANSAC, self-test, statistics report |
| `QuMatchUtils.h/.cpp` | All result saving / visualization: colored match drawing, image and text output |

Match visualization colors (only matched keypoints are drawn):

- matched keypoints in the left image: **red**
- matched keypoints in the right image: **yellow**
- match connections: **green**

The original color images are used for the match drawing.

## Pipeline

1. **Classical feature extraction (OpenCV SIFT, in `main.cpp`)**
   - SIFT feature limit: **30000**.
   - Descriptors are computed on grayscale versions of the color inputs.
   - `QuMatch::match()` applies a Lowe ratio test
     (`ratio_threshold = 0.80`) to obtain candidate queries.

2. **Candidate prefilter (OpenCV, full 128-D SIFT L2)**
   - The first `max_database` keypoints of image 2 form the kNN database
     (`max_database = 32768`).
   - BFMatcher returns up to `K = 8` SIFT-L2 candidates for each query
     (fewer if the database is smaller). This prefilter uses the full 128-D
     Euclidean distance.

3. **Coherent L2 oracle (QPP, end-to-end)**
   - For each query, the 128-D SIFT descriptors of the query and its K
     candidates are PCA-reduced to 4 dimensions and uniformly quantized to
     2 bits per dimension.
   - The oracle is a real quantum circuit with **11 qubits**:
     - 3 index qubits select a candidate;
     - multi-controlled X gates load the selected candidate's 8-bit
       quantized descriptor into an 8-qubit data register;
     - a generic truth-table circuit evaluates the **squared Euclidean (L2)**
       distance to the query and phase-flips indices below the threshold;
     - the data register is uncomputed, leaving the phase on the index.
   - Durr-Hoyer minimum finding uses this oracle with
     `floor(pi/4 * sqrt(K/M))` Grover iterations, where `M` is the number of
     candidates closer than the current threshold. Candidate distances are
     evaluated classically only to update thresholds, the stopping rule, and
     the iteration count; the oracle circuit never encodes the answer.
   - The coherent descriptor is a 4-D / 2-bit approximation of full SIFT L2;
     the full 128-D float vector cannot be basis-encoded in a state-vector
     simulator.

4. **Swap-test verification (QPP)**
   - An 8x8 patch is sampled at each keypoint, oriented and scaled by the
     keypoint, and normalized to zero mean / unit norm.
   - One ancilla plus two patch registers (12 data qubits for 8x8 patches)
     implement the swap test:
     `P(anc=0) = (1 + |<a|b>|^2) / 2`, returning `|<a|b>|^2 = NCC^2`.
   - A match is accepted when `NCC >= 0.20` and the swap-test similarity is
     `>= 0.10`.

5. **Post-processing**
   - Keep the best match for each train keypoint.
   - Filter geometrically inconsistent matches with `findHomography` + RANSAC
     (for evaluation / visualization).
   - `QuMatchUtils` writes three colored match images and a text report.

> **Note**: the oracle circuit is coherent, but image-scale 128-D float L2
> remains infeasible on a state-vector simulator. The kNN prefilter, PCA /
> quantization, and Durr-Hoyer threshold bookkeeping are classical; the
> Grover oracle and measurement are quantum. Real hardware would additionally
> need QRAM and deep arithmetic circuits.

## Build

Requirements:

- Visual Studio 2026 (`PlatformToolset v145`), **x64**
- OpenCV 4.13 (`3rdParty/OpenCV-SDK`)
- QPP 7.1.1 (`3rdParty/qpp-sdk`)
- Eigen 5.0.1 (`3rdParty/eigen-sdk`)

Open `QuMatch.slnx` and build `x64 | Debug` or `x64 | Release`. The property
sheets `OpenCVX64_*.props` configure include/library paths, `/utf-8`,
`QPP_ENABLE_QUBIT_OPTIMIZATIONS`, and copy the matching OpenCV DLL to the
output directory after each build.

Command line build:

```bat
msbuild QuMatch.slnx -p:Configuration=Release -p:Platform=x64
```

> `QPP_ENABLE_QUBIT_OPTIMIZATIONS` is important: without it the generic QPP
> kernels make the quantum simulation much slower.

Debug builds include an assertion that the swap-test result equals `NCC^2`,
which validates the QPP circuit against the classical dot product.
`--selftest` runs 20 synthetic trials of the coherent L2 oracle against the
classical minimum (currently 20/20).

## Run

With no image arguments, both built-in test pairs are processed independently:

```bat
cd QuMatch
..\x64\Release\QuMatch.exe
```

This creates:

- `results/img1_img4/` for `img1.ppm` / `img4.ppm`
- `results/adam1_adam2/` for `adam1.jpg` / `adam2.jpg`

A custom pair is saved under `results/<stem1>_<stem2>/`:

```bat
..\x64\Release\QuMatch.exe adam1.jpg adam2.jpg
```

Oracle self-test:

```bat
..\x64\Release\QuMatch.exe --selftest
```

Each pair's output directory contains:

| File | Content |
| --- | --- |
| `quantum_verified_matches.jpg` | Matches selected by the coherent L2 oracle and verified by NCC + swap test |
| `quantum_inlier_matches.jpg` | RANSAC geometric inliers (recommended view) |
| `classical_ratio_matches.jpg` | Classical SIFT + ratio-test baseline |
| `match_report.txt` | Qubits, Grover searches, agreement, match counts, timings |

`QuMatchUtils` creates the output directory automatically. The built-in pairs
are skipped with a message if their image files are not present.

All match images use the same colors: matched left keypoints are red, matched
right keypoints are yellow, and connections are green.

## API notes

- `QuMatch::classicalMatches()` stores the descriptor L2 distance in
  `DMatch::distance`.
- `QuMatch::verifiedMatches()` and `QuMatch::geometricInliers()` store
  `1 - swap-test similarity` in `DMatch::distance` (smaller is better).
- Input images must be 8-bit; grayscale, BGR, and BGRA are accepted.
- Descriptor rows must match the keypoint counts, descriptors must be CV_32F,
  and both descriptor sets must have the same number of columns.
- `patch_size` must be in [2, 32] and `max_database` in (0, 2^20].
- `candidates` must be a power of two in [2, 256]. `proj_dims` and
  `bits_per_dim` must each be in [1, 10], and their product in [2, 10]. The
  simulated coherent oracle must use at most 24 qubits.
- `grover_shots` and `max_searches` must be positive.
- Invalid parameters or inputs cause `QuMatch::match()` to throw
  `std::invalid_argument`.
- `QuMatch::selfTest()` (exposed as `--selftest`) validates the coherent
  oracle against the classical minimum on synthetic data.

## Measured results (Release x64)

`img1.ppm` / `img4.ppm`:

- SIFT keypoints 2674 / 3668; kNN database 3668
- Ratio-test candidates 231; coherent oracle: 8 candidates, 11 qubits
- 23 Grover searches over 23 of the 231 queries
- Selection agrees with the full-128D SIFT nearest for **208 / 231** queries
- Swap test passes 194 pairs; RANSAC inliers **69**
- Matching time ~0.37 s

`adam1.jpg` / `adam2.jpg`:

- SIFT keypoints 710 / 322; kNN database 322
- Ratio-test candidates 191; coherent oracle: 8 candidates, 11 qubits
- 12 Grover searches over 11 of the 191 queries
- Selection agrees with the full-128D SIFT nearest for 180 / 191 queries
- Swap test passes 151 pairs; RANSAC inliers **126**
- Matching time ~0.20 s
