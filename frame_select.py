import json
import numpy as np
import argparse
import os
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


REQUIRED_WEIGHT_KEYS = [
    'peak_similarity',
    'slope_abs',
    'rising_slope',
    'falling_slope',
    'boundary_change',
    'context_density',
]


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='RIDGE: region-informed derivative-guided frame selection'
    )

    parser.add_argument('--dataset_name', type=str, default='videomme/longvideobench/mlvu/LVbench')
    parser.add_argument('--extract_feature_model', type=str, default='clip/blip1/siglip/blip2')

    parser.add_argument('--score_path', type=str, default='your_path/scores.json')
    parser.add_argument('--frame_path', type=str, default='your_path/frames.json')
    parser.add_argument('--query_path', type=str, default='your_path/weight.json')
    parser.add_argument('--output_file', type=str, default='your_path/selected_frames')

    parser.add_argument('--max_num_frames', type=int, default=8)
    parser.add_argument('--ratio', type=int, default=1)

    parser.add_argument('--sigma', type=float, default=2.0)
    parser.add_argument('--adaptive_sigma', action='store_true')
    parser.add_argument('--peak_prominence', type=float, default=0.15)

    parser.add_argument('--base_peak_expand', type=int, default=3)
    parser.add_argument('--max_peak_expand', type=int, default=10)

    parser.add_argument('--slope_extend_k', type=float, default=0.5)
    parser.add_argument('--max_extend_factor', type=float, default=4.0)
    parser.add_argument('--boundary_k', type=float, default=2.0)

    return parser.parse_args()


def validate_weight_dict(weight_dict, idx):
    if not isinstance(weight_dict, dict):
        raise TypeError(f'queries[{idx}] must be a dict, got {type(weight_dict)}.')

    missing = [k for k in REQUIRED_WEIGHT_KEYS if k not in weight_dict]
    if missing:
        raise KeyError(f'queries[{idx}] missing required weight keys: {missing}')


def compute_coverage_intent(weight_dict):
    ps = float(weight_dict['peak_similarity'])
    rs = float(weight_dict['rising_slope'])
    fs = float(weight_dict['falling_slope'])
    cd = float(weight_dict['context_density'])

    total = ps + rs + fs + cd + 1e-9
    return (rs + fs + cd) / total


def analyze_curve(scores, args, coverage_intent):
    scores = np.array(scores, dtype=np.float64)
    T = len(scores)
    region_labels = ['background'] * T

    if T < 5:
        return region_labels, {'scores': scores, 'peaks': []}

    sigma = (
        float(np.clip(T / 500.0, 2.0, 10.0))
        if args.adaptive_sigma
        else args.sigma
    )

    s_smooth = gaussian_filter1d(scores, sigma=sigma) if sigma > 0 else scores.copy()
    s_prime = np.gradient(s_smooth)
    s_double_prime = np.gradient(s_prime)

    abs_s1 = np.abs(s_prime)
    abs_s2 = np.abs(s_double_prime)

    peaks, _ = find_peaks(
        s_smooth,
        prominence=args.peak_prominence,
        distance=5,
    )

    if len(peaks) == 0:
        peaks = np.array([int(np.argmax(s_smooth))])

    peak_widths = {}
    ref_curvature = float(np.median(abs_s2)) + 1e-12

    for p in peaks:
        rel = abs_s2[p] / ref_curvature
        width = args.base_peak_expand / np.sqrt(max(rel, 0.25))
        width = int(np.clip(round(width), 2, args.max_peak_expand))
        peak_widths[int(p)] = width

    for p in peaks:
        w = peak_widths[int(p)]
        left = max(0, p - w)
        right = min(T, p + w + 1)

        for t in range(left, right):
            region_labels[t] = 'peak'

    slope_std = float(np.std(s_prime)) + 1e-12

    effective_k = args.slope_extend_k * (1.0 - coverage_intent) + 0.1 * coverage_intent
    slope_threshold = effective_k * slope_std

    base_max_extend = int(round(args.max_extend_factor * max(sigma, 1.0)))
    effective_max_extend = int(round(base_max_extend * (1.0 + coverage_intent)))
    effective_max_extend = min(effective_max_extend, T // 4)

    for p in peaks:
        w = peak_widths[int(p)]

        t = p - w - 1
        extended = 0

        while (
            t >= 0
            and region_labels[t] == 'background'
            and extended < effective_max_extend
            and s_prime[t] > slope_threshold
        ):
            region_labels[t] = 'rising'
            t -= 1
            extended += 1

        t = p + w + 1
        extended = 0

        while (
            t < T
            and region_labels[t] == 'background'
            and extended < effective_max_extend
            and s_prime[t] < -slope_threshold
        ):
            region_labels[t] = 'falling'
            t += 1
            extended += 1

    bg_indices = [
        t for t in range(T)
        if region_labels[t] == 'background'
    ]

    if len(bg_indices) > 0:
        bg_slopes = abs_s1[bg_indices]
        med = float(np.median(bg_slopes))
        std = float(np.std(bg_slopes)) + 1e-12
        threshold = med + args.boundary_k * std

        for t in bg_indices:
            if abs_s1[t] > threshold:
                region_labels[t] = 'boundary'

    curve_info = {
        'scores': scores,
        's_smooth': s_smooth,
        's_prime': s_prime,
        's_double_prime': s_double_prime,
        'peaks': peaks.tolist(),
        'peak_widths': peak_widths,
        'slope_threshold': slope_threshold,
        'coverage_intent': coverage_intent,
        'sigma': sigma,
    }

    return region_labels, curve_info


def allocate_frames(weight_dict, region_labels, num_frames):
    region_counts = {
        label: sum(1 for r in region_labels if r == label)
        for label in ['peak', 'rising', 'falling', 'boundary', 'background']
    }

    raw_weights = {
        'peak': float(weight_dict['peak_similarity']),
        'rising': float(weight_dict['rising_slope']),
        'falling': float(weight_dict['falling_slope']),
        'boundary': float(weight_dict['boundary_change']) + float(weight_dict['slope_abs']),
        'background': float(weight_dict['context_density']),
    }

    active_regions = {
        region: weight
        for region, weight in raw_weights.items()
        if region_counts[region] > 0 and weight > 0
    }

    if not active_regions:
        fallback = {
            region: 1.0
            for region in raw_weights
            if region_counts[region] > 0
        }

        if not fallback:
            return {'background': num_frames}

        active_regions = fallback

    total_weight = sum(active_regions.values())

    allocation = {}
    allocated = 0

    sorted_regions = sorted(
        active_regions.keys(),
        key=lambda region: active_regions[region],
        reverse=True,
    )

    for region in sorted_regions[:-1]:
        n = int(round(num_frames * active_regions[region] / total_weight))
        n = min(n, region_counts[region], num_frames - allocated)

        allocation[region] = max(n, 0)
        allocated += allocation[region]

    last_region = sorted_regions[-1]
    allocation[last_region] = max(
        0,
        min(num_frames - allocated, region_counts[last_region]),
    )
    allocated += allocation[last_region]

    remaining = num_frames - allocated

    if remaining > 0:
        for region in sorted_regions:
            can_add = region_counts[region] - allocation.get(region, 0)
            add = min(remaining, can_add)

            allocation[region] = allocation.get(region, 0) + add
            remaining -= add

            if remaining <= 0:
                break

    return allocation


def select_by_score(region_indices, ranking_values, num_select, min_gap=2):
    if num_select <= 0 or len(region_indices) == 0:
        return []

    num_select = min(num_select, len(region_indices))

    candidates = sorted(
        region_indices,
        key=lambda t: ranking_values[t],
        reverse=True,
    )

    selected = []

    for t in candidates:
        if len(selected) >= num_select:
            break

        if min_gap > 0 and any(abs(t - s) < min_gap for s in selected):
            continue

        selected.append(t)

    if len(selected) < num_select:
        for t in candidates:
            if len(selected) >= num_select:
                break

            if t not in selected:
                selected.append(t)

    return selected


def select_by_temporal_bins(region_indices, s_smooth, num_select):
    if num_select <= 0 or len(region_indices) == 0:
        return []

    sorted_indices = sorted(region_indices)

    if num_select >= len(sorted_indices):
        return sorted_indices

    bins = np.array_split(sorted_indices, num_select)

    selected = []

    for bin_frames in bins:
        if len(bin_frames) == 0:
            continue

        best = max(bin_frames, key=lambda t: s_smooth[t])
        selected.append(int(best))

    return selected


def gradselect(scores, num_frames, weight_dict, args):
    scores = np.array(scores, dtype=np.float64)
    T = len(scores)

    if T <= num_frames:
        return list(range(T))

    smin, smax = scores.min(), scores.max()

    if smax - smin > 1e-9:
        normalized = (scores - smin) / (smax - smin)
    else:
        normalized = scores.copy()

    coverage_intent = compute_coverage_intent(weight_dict)

    region_labels, curve_info = analyze_curve(
        normalized,
        args,
        coverage_intent,
    )

    allocation = allocate_frames(
        weight_dict,
        region_labels,
        num_frames,
    )

    region_frame_map = {
        label: []
        for label in ['peak', 'rising', 'falling', 'boundary', 'background']
    }

    for t, label in enumerate(region_labels):
        region_frame_map[label].append(t)

    abs_s1 = np.abs(curve_info['s_prime'])
    s_smooth = curve_info['s_smooth']

    selected = []

    for region, n_alloc in allocation.items():
        if n_alloc <= 0:
            continue

        indices = region_frame_map.get(region, [])

        if len(indices) == 0:
            continue

        if region in ('rising', 'falling'):
            if n_alloc >= 3 and len(indices) >= 3:
                chosen = select_by_temporal_bins(indices, s_smooth, n_alloc)
            else:
                chosen = select_by_score(indices, s_smooth, n_alloc, min_gap=2)

        elif region == 'boundary':
            chosen = select_by_score(indices, abs_s1, n_alloc, min_gap=2)

        else:
            chosen = select_by_score(indices, s_smooth, n_alloc, min_gap=2)

        selected.extend(chosen)

    selected = sorted(set(selected))

    if len(selected) < num_frames:
        chosen_set = set(selected)

        remaining = [
            t for t in range(T)
            if t not in chosen_set
        ]

        remaining.sort(
            key=lambda t: s_smooth[t],
            reverse=True,
        )

        for t in remaining:
            if len(selected) >= num_frames:
                break

            selected.append(t)

        selected.sort()

    return selected[:num_frames]


def process_single_video(score, fn, weight_dict, num_frames, args):
    selected_indices = gradselect(
        score,
        num_frames,
        weight_dict,
        args,
    )

    return [fn[i] for i in selected_indices]


def main(args):
    if args.ratio <= 0:
        raise ValueError(f'ratio must be positive, got {args.ratio}.')

    with open(args.score_path) as f:
        itm_outs = json.load(f)

    with open(args.frame_path) as f:
        fn_outs = json.load(f)

    with open(args.query_path) as f:
        queries = json.load(f)

    assert len(itm_outs) == len(fn_outs), (
        f'Length mismatch: scores={len(itm_outs)}, frames={len(fn_outs)}.'
    )

    assert len(queries) == len(itm_outs), (
        f'Length mismatch: weights={len(queries)}, videos={len(itm_outs)}.'
    )

    print(f'[INFO] Loaded {len(queries)} weight dicts from {args.query_path}')

    out_dir = os.path.join(
        args.output_file,
        args.dataset_name,
        args.extract_feature_model,
    )

    os.makedirs(out_dir, exist_ok=True)

    outs = []

    for idx, (itm_out, fn_out, weight_dict) in enumerate(zip(itm_outs, fn_outs, queries)):
        validate_weight_dict(weight_dict, idx)

        nums = len(itm_out) // args.ratio

        new_score = [
            itm_out[num * args.ratio]
            for num in range(nums)
        ]

        new_fnum = [
            fn_out[num * args.ratio]
            for num in range(nums)
        ]

        if len(new_score) >= args.max_num_frames:
            selected_fn = process_single_video(
                new_score,
                new_fnum,
                weight_dict,
                args.max_num_frames,
                args,
            )
            outs.append(selected_fn)
        else:
            outs.append(new_fnum)

        if (idx + 1) % 100 == 0:
            print(f'[INFO] Processed {idx + 1}/{len(itm_outs)} videos')

    output_path = os.path.join(
        out_dir,
        'selected_frames_GradSelect.json',
    )

    with open(output_path, 'w') as f:
        json.dump(outs, f)

    print(f'[DONE] Saved {len(outs)} entries to {output_path}')


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
