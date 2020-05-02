#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from change_point_identification import Synthesizer, ChangePointLearner, sample_commits, construct_sample, query_configurations
import sys

import json
import os
import numpy as np

if __name__ == "__main__":

    seed = int(sys.argv[1])
    n_options = int(sys.argv[2])
    n_commits = int(sys.argv[3])
    n_changepoints = int(sys.argv[4])
    p_interactions = float(sys.argv[5])
    p_interaction_degree = float(sys.argv[6])

    commit_srate = float(sys.argv[7])
    measurements_per_iteration = int(sys.argv[8])
    n_configs = int(sys.argv[9]) * n_options

    noise = 0.0

    synth = Synthesizer(
        n_commits,
        n_options,
        n_changepoints,
        p_interactions,
        p_interaction_degree,
        noise,
        seed
    )

    # obtain random configurations

    confs = np.vstack(list(map(lambda x: np.array(x), query_configurations("", [], n_options, n_configs, ignore_vm=True, set_vary=[]))))
    revs = sample_commits(len(confs), n_commits, commit_srate)
    sample = construct_sample(confs, revs)
    learner = ChangePointLearner(synth, m_measurements_per_iteration=measurements_per_iteration)

    # log
    log = {}
    initial = {
        "n_commits": n_commits,
        "n_options": n_options,
        "n_changepoints": n_changepoints,
        "p_interactions": p_interactions,
        "p_interaction_degree": p_interaction_degree,
        "seed": seed,
        "commit_srate": commit_srate,
        "measurements_per_iteration": measurements_per_iteration,
        "n_configs": n_configs,
        "ground_truth": list(map(lambda c: {"commit": int(c[0]), "option": int(c[1])},learner.get_ground_truth()))
    }

    log["parameters"] = initial
    iterations = []
    print(learner.get_ground_truth())

    for i in range(30):
        print(i)
        if not learner.stop():
            learner.build_likelihoods(sample, threshold=0.01)
            cs = learner.calc_candidate_solution()

            learner.caching()

            learner.acquire_configurations()
            sample = learner.next_sample()

            scores = learner.score(estimated=learner.cached_solutions.keys(), return_f1=True)

            cs = list(map(lambda c: {"commit": int(c["commit"]), "option": int(c["option"])}, cs))
            scores["candidate_solution"] = cs
            iterations.append(scores)
        else:
            break
        

    log["iterations"] = iterations

    with open('/home/stefan/results_{}.json'.format(seed), 'w') as fp:
        fp.write(json.dumps(log, indent=2))

    os.system("rsync -a --remove-source-files '/home/stefan/results_{}.json' '/media/raid/stefan/synthetic_evaluation_results/results_{}.json'".format(seed, seed))
