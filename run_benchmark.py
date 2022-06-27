#
import os
import ast
import subprocess as sp
import re
from argparse import ArgumentParser
from copy import deepcopy
import numpy as np


def get_args():
    """Read in cmd arguments"""
    parser = ArgumentParser(description='Run specified model on particular benchmark')
    parser.add_argument('--version', '-v', action='version', version='%(prog)s 0.1')
    parser.add_argument('--venv', type=str, default='l2o',
                        help='virtual conda environment name to activate.')
    parser.add_argument('--run_file', '-r', type=str)
    parser.add_argument('--data_dir', '-d', type=str,
                        default='./data/CVRP/benchmark/test')
    parser.add_argument('--group_id', '-g', nargs='+', default='')
    parser.add_argument('--instance_pth', type=str, default=None)
    parser.add_argument('--problem', '-p', type=str, default='jssp')
    parser.add_argument('--model', '-m', type=str, default='meta')
    parser.add_argument('--policy', type=str, default='')
    parser.add_argument('--num_steps', '-n', type=int, default=0)
    parser.add_argument('--eval_cfg', '-e', type=str, default='')
    parser.add_argument('--dry_run', action='store_true',
                        help='only run outer loop without executing.')
    parser.add_argument('--args', type=str, default='',
                        help='additional command line arguments as one string.')
    args = vars(parser.parse_args())  # parse to dict
    return args


def dict_from_str(s):
    idx = s.find("{")
    return ast.literal_eval(s[idx:])


def run():
    """Run specified solver on provided benchmark."""
    args = get_args()
    grps = deepcopy(args['group_id'])
    inst_pth = args['instance_pth']
    if inst_pth is not None:
        grps = [os.path.split(os.path.dirname(inst_pth))[-1]]
    for gid in grps:
        cwd = os.getcwd()
        path = os.path.join(cwd, f"outputs_benchmark/")
        os.makedirs(path, exist_ok=True)

        args['group_id'] = gid
        rfile = args['run_file']
        # manage directories and files
        if inst_pth is not None:
            ddir = os.path.split(os.path.dirname(inst_pth))[0]
            inst_dir = os.path.dirname(inst_pth)
            file_names = [os.path.split(inst_pth)[-1]]
        else:
            ddir = args['data_dir']
            inst_dir = os.path.join(ddir, args['group_id'])
            file_names = os.listdir(inst_dir)
            file_names.sort()
            print(f"Loading all instances in {inst_dir}")
            print(file_names)

        d_str = os.path.basename(os.path.dirname(ddir))
        path = os.path.join(path, f"{d_str}_{args['group_id']}/")
        os.makedirs(path, exist_ok=True)

        if len(args['args']) > 0:
            print(f"specified additional args: {args['args']}")
        prob = args['problem'].lower()
        m = args['model'].lower()
        print(prob, ddir.lower())
        assert prob in ddir.lower()
        assert m in rfile.lower()
        steps = args['num_steps']
        pol = args['policy'] if m in ["meta", "pdr"] else ""

        out_fname = f"results_{m}{'-' + pol if len(pol)>0 else ''}.log"
        out_path = os.path.join(path, out_fname)
        print(f"writing to: {out_path}")

        results = {
            "cost_mean": [],
            "cost_std": [],
            "num_vehicles_mean": [],
            "num_vehicles_std": [],
            "num_vehicles_median": [],
            "run_time_mean": [],
            "run_time_total": [],
            "num_infeasible": [],
        }
        n_inf = 0

        for i, fname in enumerate(file_names):
            data_pth = os.path.join(inst_dir, fname)

            if m == "nls":
                run_args = f"meta={args['eval_cfg']} " \
                           f"tester_cfg.test_env_cfg.data_file_path={data_pth} " \
                           f"tester_cfg.test_batch_size=1 " \
                           f"tester_cfg.test_dataset_size=1 " \
                           f"tester_cfg.env_kwargs.num_steps={steps}"
            elif m == "meta":
                run_args = f"policy={args['policy']} " \
                           f"env_cfg.data_file_path={data_pth} " \
                           f"batch_size=1 " \
                           f"dataset_size=1 " \
                           f"env_kwargs.num_steps={steps}"
            elif m == "pdr":
                run_args = f"policy_cfg.method={pol.upper()} " \
                           f"data_file_path={data_pth} " \
                           f"batch_size=1 " \
                           f"dataset_size=1"
                steps = 0
            elif m == "dact":
                run_args = f"data_file_path={data_pth} " \
                           f"batch_size=1 " \
                           f"dataset_size=1 " \
                           f"T_max={steps}"
            elif m == "gort":
                run_args = f"policy={args['policy']} " \
                           f"data_file_path={data_pth} " \
                           f"batch_size=1 " \
                           f"dataset_size=1 " \
                           f"policy_cfg.solution_limit={steps}"
            else:
                run_args = f"data_file_path={data_pth} " \
                           f"batch_size=1 " \
                           f"dataset_size=1"
                if len(args['policy']) > 0:
                    run_args += f" policy={args['policy']}"

            if prob == "cvrp":
                gsize = int(re.search('X-n(.+?)-k', data_pth).group(1))-1
                run_args += f" graph_size={gsize}"

            inst_str = os.path.splitext(fname)[0]
            print(f"running {rfile} for {inst_str}")
            out = None
            if not args['dry_run']:
                try:
                    out = sp.run([
                            f"python",
                            rfile,
                            *run_args.split(),
                            *args['args'].split()
                        ],
                        universal_newlines=True,
                        capture_output=True,
                        check=True
                    )
                except sp.CalledProcessError as e:
                    print(f"encountered error for call: {e.cmd}\n")
                    print(e.stderr)

            res_str = f"{i}: {m}{'-' + pol if len(pol) > 0 else ''}, {inst_str}, steps={steps}"
            if out is not None:
                lines = out.stdout.splitlines()
                if len(lines[-1]) < 10:     # exit code sometimes displayed in interpreter
                    del lines[-1]
                #res_info = dict_from_str(lines[-2])
                if "nan" in lines[-1].lower():
                    res_str += f", NOT_SOLVED"
                    n_inf += 1
                    print(f"nan in {i}: {lines[-1]}")
                else:
                    try:
                        res = dict_from_str(lines[-1])
                        for k, v in res.items():
                            if "cost" in k:
                                if prob != "cvrp":
                                    v = round(v, 0)
                                else:
                                    v *= 1000
                                results[k].append(v)
                            else:
                                results[k].append(v)
                        res_str += f", {res}"
                    except ValueError as e:
                        print(e)
                        res_str += f", ERROR"
            else:
                print("none")
                res_str += f", ERROR"
            with open(out_path, "a") as o:
                o.write(f"{res_str}\n")

        results_str = {
            "cost_mean": np.mean(results["cost_mean"]),
            "cost_std": np.std(results["cost_mean"]),
            "num_vehicles_mean": np.mean(results["num_vehicles_mean"])
            if len(results["num_vehicles_mean"]) > 0 else None,
            "num_vehicles_std": np.std(results["num_vehicles_mean"])
            if len(results["num_vehicles_mean"]) > 0 else None,
            "num_vehicles_median": np.median(results["num_vehicles_mean"])
            if len(results["num_vehicles_mean"]) > 0 else None,
            "run_time_mean": np.mean(results["run_time_mean"]),
            "run_time_total": np.sum(results["run_time_total"]),
            "num_infeasible": np.sum(results["num_infeasible"]) + n_inf,
        }

        with open(out_path, "a") as o:
            o.write(f"{results_str}\n")

        print(f"\ndone.\n\n")

        #
        if prob == "cvrp":
            for cost, k, runtime in zip(results["cost_mean"], results["num_vehicles_mean"], results["run_time_mean"]):
                print(cost)
                print(k)
                print(runtime)
        else:
            for cost, runtime in zip(results["cost_mean"], results["run_time_mean"]):
                print(cost)
                print(runtime)
        print("-------------------")
        print(results_str["cost_mean"])
        print(results_str["num_vehicles_mean"])
        print(results_str["run_time_mean"])
        print(results_str["run_time_total"])


if __name__ == "__main__":
    run()
