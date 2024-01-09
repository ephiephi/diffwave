from pesq import pesq
from pystoi import stoi
import numpy
import os
import pickle
from tqdm import tqdm
import torch
import torchaudio
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import torchaudio.functional as F

import argparse
import json

from DNSMOS import dnsmos_local


def main_measure(
    enhance_dir,
    clean_dir,
    noisy_dir="/data/ephraim/datasets/known_noise/noisy_wav/",
):
    doc_file = os.path.join(enhance_dir, "args.json")


    pkl_results_file = os.path.join(enhance_dir, "stats.pickle")

    if os.path.exists(pkl_results_file):
        with open(pkl_results_file, "rb") as handle:
            df = pd.read_pickle(handle)
            stats = df.describe()
            print(stats)

    else:
        df = calc_measures(noisy_dir, clean_dir, enhance_dir)
        stats = df.describe()
        print(stats)

        with open(pkl_results_file, "wb") as f:
            pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL)

    dns_pickle = os.path.join(enhance_dir, "dnsmos.pickle")
    if os.path.exists(dns_pickle):
        with open(dns_pickle, "rb") as handle:
            dns_df = pd.read_pickle(handle)
            stats_mos = dns_df.describe()
            print(stats_mos)
    else:
        mos_args = argparse.Namespace(
            testset_dir=enhance_dir, personalized_MOS=False, csv_path=None
        )
        dns_df = dnsmos_local.main(mos_args)
        stats_mos = dns_df.describe()
        # print(stats_mos)

        with open(dns_pickle, "wb") as f:
            dns_df.to_pickle(f)

    documentation = (
        "pesq:{} stoi:{} OVRL:{} SIG:{} BAK:{} enhanced_dir:{} args: {}\n".format(
            stats["pesq_enhanced"]["mean"],
            stats["stoi_enhanced"]["mean"],
            stats_mos["OVRL"]["mean"],
            stats_mos["SIG"]["mean"],
            stats_mos["BAK"]["mean"],
            enhance_dir,
            str(args),
        )
    )

    stats_path = "stats_dns.txt"
    with open(stats_path, "a") as file1:
        file1.write(documentation)


def calc_measures(noisy_dir, clean_dir, enhance_dir):
    noises = os.listdir(noisy_dir)
    dont_calculated = []
    results = {
        "pesq_noisy": {},
        "stoi_noisy": {},
        "pesq_enhanced": {},
        "stoi_enhanced": {},
    }

    i = 0

    for ref_filename in tqdm(noises):
        if not ref_filename.endswith(".wav"):
            continue
        reference = os.path.join(clean_dir, ref_filename)
        test_noisy = os.path.join(noisy_dir, ref_filename)
        test_enhanced = os.path.join(enhance_dir, ref_filename)
        WAVEFORM_SPEECH, SAMPLE_RATE_SPEECH = torchaudio.load(reference)
        WAVEFORM_NOISE, SAMPLE_RATE_NOISE = torchaudio.load(test_noisy)
        WAVEFORM_enhanced, SAMPLE_RATE_enhanced = torchaudio.load(test_enhanced)
        # print("Computing scores for ", reference)
        try:
            pesq_noise = pesq(
                16000,
                WAVEFORM_SPEECH[0].numpy(),
                WAVEFORM_NOISE[0].numpy(),
                mode="wb",
            )
            stoi_noise = stoi(
                WAVEFORM_SPEECH[0].numpy(),
                WAVEFORM_NOISE[0].numpy(),
                16000,
                extended=False,
            )

            if WAVEFORM_SPEECH.shape[1] < WAVEFORM_enhanced.shape[1]:
                WAVEFORM_enhanced = WAVEFORM_enhanced[:, : WAVEFORM_SPEECH.shape[1]]
            else:
                WAVEFORM_SPEECH = WAVEFORM_SPEECH[:, : WAVEFORM_enhanced.shape[1]]
            pesq_enhanced = pesq(
                16000,
                WAVEFORM_SPEECH[0].numpy(),
                WAVEFORM_enhanced[0].numpy(),
                mode="wb",
            )
            stoi_enhanced = stoi(
                WAVEFORM_SPEECH[0].numpy(),
                WAVEFORM_enhanced[0].numpy(),
                16000,
                extended=False,
            )

            results["pesq_noisy"][ref_filename] = pesq_noise
            results["stoi_noisy"][ref_filename] = stoi_noise

            results["stoi_enhanced"][ref_filename] = stoi_enhanced
            results["pesq_enhanced"][ref_filename] = pesq_enhanced
            df = pd.DataFrame.from_dict(results)
            df["pesq_diff"] = df["pesq_enhanced"].sub(df["pesq_noisy"])
            df["stoi_diff"] = df["stoi_enhanced"].sub(df["stoi_noisy"])
        except:
            dont_calculated.append(ref_filename)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="measure guided")
    parser.add_argument(
        "-enhanced_dir",
        default="/data/ephraim/datasets/known_noise/enhanced_diffwave/",
    )
    parser.add_argument(
        "-clean_dir", default="/data/ephraim/datasets/known_noise/clean_wav/"
    )

    args = parser.parse_args()
    main_measure(
        args.enhanced_dir,
        args.clean_dir,
    )
