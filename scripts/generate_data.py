import numpy as np
import pandas as pd
import argparse

# for prints
DG = "[data gen]"


def generate_generative_demo(n=194, seed=0, p_cancer=0.45):
    """
    Hypothesis:
        baseline cancer risk is slightly correlated with age/BMI
        higher methylation entropy more strongly correlated with higher Gleason score
    """
    
    np.random.seed(seed) 
    # Gleason for cancer samples (multinomial / categorical)
    gleason_levels = np.array([6, 7, 8, 9, 10])
    
    # mock probabilities for Gleason among cancers
    gleason_probs = np.array([0.25, 0.45, 0.20, 0.08, 0.02])

    # entropy drawn from clipped normal with mean/sd by gleason
    entropy_means = {
        0: 0.40, 6: 0.42, 7: 0.44,
        8: 0.47, 9: 0.50, 10: 0.54
    }
    
    entropy_sds = {
        0: 0.02, 6: 0.03, 7: 0.035,
        8: 0.05, 9: 0.06, 10: 0.07
    }
    
    # age and BMI means. will increment with gleason to give slight covar
    age_mean = 55.0
    bmi_mean = 26.5
    cov = np.array([[9.0, 0.3], [0.3, 4.0]])
    
    data = []
    for obs in range(n):
        pid = f"P{1000+obs}"
        
        # first sample cancer status
        has_cancer = np.random.uniform() < p_cancer
        
        # derive a gleason score
        if not has_cancer:
            gleason = 0
        
        else:
            gleason = np.random.choice(gleason_levels, size=1, p=gleason_probs)[0]
    
        # entropy determined by gleason + noise for this demo
        mu_s = entropy_means[gleason]
        sigma_s = entropy_sds[gleason]
        
        # clipped normal
        methylation_entropy = np.clip(np.random.normal(mu_s, sigma_s), 0.0, 1.0)
        
        # slightly higher cancer risk as age increased
        age_i = age_mean + 0.5*gleason
        bmi_i = bmi_mean + 0.2*gleason
        age_bmi = np.random.multivariate_normal([age_i, bmi_i], cov)
        
        # clip and make integer
        age = int(np.clip(age_bmi[0], 40, 100))
        bmi = int(np.clip(age_bmi[1], 10, 50))

        observation = {
            "patient_id": pid,
            "cancer_status": int(has_cancer),
            "gleason_score": gleason,
            "methylation_entropy": np.round(methylation_entropy, 4),
            "age": age,
            "bmi": bmi
        }
        
        data.append(observation)
    
    df_out = pd.DataFrame(data)
    return df_out


def parse_args():
    p = argparse.ArgumentParser(description="Generate demo cancer methylation dataset")
    p.add_argument("--out", required=True, help="output CSV path, e.g. data/demo.csv")
    p.add_argument("--n", type=int, default=194, help="number of samples to generate")
    p.add_argument("--seed", type=int, default=0, help="random seed")
    p.add_argument(
        "--p-cancer", type=float, default=0.45, help="prevalence of cancer (0..1)"
    )
    
    return p.parse_args()


def main():
    args = parse_args()
    
    print(f"{DG} Generating demo dataset")
    print("Arguments:")
    for name, val in vars(args).items():
        print(f"  {name}: {val}")
    
    df = generate_generative_demo(n=args.n, seed=args.seed, p_cancer=args.p_cancer)
    df.to_csv(args.out, index=False)
    print(f"{DG} complete")


if __name__ == "__main__":
    main()