def process(root, base_str):
    targets = [0, 4]
    aggr = {}
    ignore_target = False
    ignore_ve = True
    ignore_mode = False
    raw_stats = False
    for target in targets:
        target_path = f'{root}{os.sep}world{target}_seq_timing{base_str}.txt'
        with open(target_path, 'r') as f:
            for line in f:
                line = line.replace('[SEQ_TIMING ', '[')
                line = line.strip().split()
                if not line:
                    continue
                mode = line[0].strip('[]')
                line_dict = {k: v for k, v in (item.split('=', 1) for item in line[1:])}
                line_dict = {k: v for k, v in line_dict.items() if str(v) != 'None'}
                line_dict = {k: float(v.strip('ms%')) for k, v in line_dict.items()}
                

                world_rank = int(line_dict.pop('world_rank'))
                tp_rank = int(line_dict.pop('tp_rank'))
                pp_rank = int(line_dict.pop('pp_rank'))
                ve = int(line_dict.pop('ve'))
                seq_id = int(line_dict.pop('seq_id'))
                stage_step = int(line_dict.pop('stage_step'))
                total_steps = int(line_dict.pop('total_steps'))

                tgt_dict = aggr.setdefault('NA' if ignore_target else target, {})
                ve_dict = tgt_dict.setdefault('NA' if ignore_ve else ve, {})
                mode_dict = ve_dict.setdefault('NA' if ignore_mode else mode, {})
                for key, val in line_dict.items():
                    steps_list = mode_dict.setdefault(key, [])
                    # Ensure list length >= stage_step
                    while len(steps_list) < stage_step:
                        steps_list.append([])
                    steps_list[stage_step - 1].append(val)


    # Write CSV
    import csv, statistics

    out_path = f'{root}{os.sep}seq_timing_aggregated{base_str}.csv'
    with open(out_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Determine maximum number of steps across all keys to build header
        max_steps = 0
        for target in aggr.values():
            for mode in target.values():
                for steps in mode.values():
                    max_steps = max(max_steps, len(steps))
        header = ['Target', 'VE', 'Mode', 'Key', 'Analysis',
                'GlobalMin', 'GlobalMax', 'GlobalMedian', 'GlobalMean', 'GlobalCount']
        header.extend([f'Step{idx+1}' for idx in range(max_steps)])
        writer.writerow(header)

        for target in sorted(aggr.keys()):
            for ve in sorted(aggr[target].keys()):
                for mode in sorted(aggr[target][ve].keys()):
                    for key in aggr[target][ve][mode].keys():  # insertion order preserved
                        steps = aggr[target][ve][mode][key]  # list of lists
                        # Flatten all values
                        flat = [v for step in steps for v in step if isinstance(v, (int, float))]
                        if not flat:
                            continue
                        global_min = min(flat)
                        global_max = max(flat)
                        global_mean = statistics.fmean(flat)
                        global_median = statistics.median(flat)
                        global_count = len(flat)

                        # Prepare per-step aggregations
                        def step_agg(func, default=None):
                            agg_vals = []
                            for lst in steps:
                                nums = [x for x in lst if isinstance(x, (int, float))]
                                if nums:
                                    agg_vals.append(func(nums))
                                else:
                                    agg_vals.append(default)
                            # Pad missing steps with default if any
                            while len(agg_vals) < max_steps:
                                agg_vals.append(default)
                            return agg_vals

                        step_mins = step_agg(min)
                        step_maxs = step_agg(max)
                        step_means = step_agg(statistics.fmean)
                        step_medians = step_agg(statistics.median)

                        # Emit four rows: Min/Max/Median/Mean (each row contains step-wise values of that metric)
                        if raw_stats:
                            writer.writerow([target, ve, mode, key, 'Min',
                                            global_min, global_max, global_median, global_mean, global_count,
                                            *step_mins])
                            writer.writerow([target, ve, mode, key, 'Max',
                                            global_min, global_max, global_median, global_mean, global_count,
                                            *step_maxs])
                            writer.writerow([target, ve, mode, key, 'Median',
                                            global_min, global_max, global_median, global_mean, global_count,
                                            *step_medians])
                        writer.writerow([target, ve, mode, key, 'Mean',
                                        global_min, global_max, global_median, global_mean, global_count,
                                        *step_means])
                        
        # ---- Derived pct_idle + ratio ----
        if not ignore_target:
            # Build helper maps: derived_idle[target][ve][mode] = {'steps': [...], stats...}
            derived_idle = {}
            for target in sorted(aggr.keys()):
                for ve in sorted(aggr[target].keys()):
                    for mode in sorted(aggr[target][ve].keys()):
                        mode_dict = aggr[target][ve][mode]
                        pct_gap_steps = mode_dict.get('pct_gap')
                        pct_recv_steps = mode_dict.get('pct_recv')
                        if pct_gap_steps is None:
                            continue
                        # Step mean lists for underlying keys
                        def mean_per_step(step_lists):
                            out = []
                            for lst in step_lists:
                                nums = [x for x in lst if isinstance(x, (int, float))]
                                out.append(statistics.fmean(nums) if nums else 0.0)
                            while len(out) < max_steps:
                                out.append(0.0)
                            return out
                        gap_means = mean_per_step(pct_gap_steps)
                        recv_means = mean_per_step(pct_recv_steps) if pct_recv_steps else [0.0]*max_steps
                        if target == 0:
                            idle_step_means = gap_means
                        else:
                            # pct_idle = pct_gap + pct_recv(target) - pct_recv(target0)
                            recv_means_t0 = [0.0]*max_steps
                            if 0 in aggr and ve in aggr[0] and mode in aggr[0][ve] and 'pct_recv' in aggr[0][ve][mode]:
                                recv_means_t0 = mean_per_step(aggr[0][ve][mode]['pct_recv'])
                            idle_step_means = [gap_means[i] + recv_means[i] - recv_means_t0[i] for i in range(max_steps)]
                        flat_idle = [v for v in idle_step_means if isinstance(v, (int, float))]
                        if not flat_idle:
                            continue
                        derived_idle.setdefault(target, {}).setdefault(ve, {})[mode] = {
                            'steps': idle_step_means,
                            'global_min': min(flat_idle),
                            'global_max': max(flat_idle),
                            'global_mean': statistics.fmean(flat_idle),
                            'global_median': statistics.median(flat_idle),
                            'global_count': len(flat_idle),
                        }
                        # Emit pct_idle row
                        writer.writerow([
                            target, ve, mode, 'pct_true_idle', 'Mean',
                            min(flat_idle), max(flat_idle),
                            statistics.median(flat_idle), statistics.fmean(flat_idle),
                            len(flat_idle), *idle_step_means
                        ])
            # Idle ratio rows (target0 / target4) per ve/mode
            if 0 in derived_idle and 4 in derived_idle:
                common_ves = set(derived_idle[0].keys()) & set(derived_idle[4].keys())
                for ve in sorted(common_ves):
                    common_modes = set(derived_idle[0][ve].keys()) & set(derived_idle[4][ve].keys())
                    for mode in sorted(common_modes):
                        idle0 = derived_idle[0][ve][mode]
                        idle4 = derived_idle[4][ve][mode]
                        steps_ratio = []
                        for i in range(max_steps):
                            v4 = idle4['steps'][i] if i < len(idle4['steps']) else 0.0
                            v0 = idle0['steps'][i] if i < len(idle0['steps']) else 0.0
                            steps_ratio.append(v0 / v4 if v4 != 0 else 0.0)
                        flat_ratio = [v for v in steps_ratio if isinstance(v, (int, float))]
                        if not flat_ratio:
                            continue
                        writer.writerow([
                            'NA', ve, mode, 'idle_ratio', 'Mean',
                            min(flat_ratio), max(flat_ratio),
                            statistics.median(flat_ratio), statistics.fmean(flat_ratio),
                            len(flat_ratio), *steps_ratio
                        ])
    print(f"Wrote aggregated CSV to {out_path}")

if __name__ == "__main__":
    import os, re
    targets = {0, 4}
    regex = re.compile(r"world([04])_seq_timing(.*)\.txt$")
    
    for root, dirs, files in os.walk("."):
        base_strs = set()
        for file in files:
            m = regex.match(file)
            if not m:
                continue
            tgt = int(m.group(1))
            if tgt in targets:
                base_strs.add(m.group(2))
        for b in sorted(base_strs):
            if b != '':
                continue
            process(root, b)