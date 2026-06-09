"""
Combine les predictions de deux pipelines entraines separement : modele A et modele B
Blend pondere : 0.25 * A + 0.75 * B  (penche vers le meilleur modele B = 0.00103).

Usage :
    python blend_final.py predictions_A.csv predictions_B.csv submission.csv
"""
import sys
import numpy as np
import pandas as pd

W_A = 0.25   # poids du modele A 
W_B = 0.75   # poids du modele B 

def main(path_a, path_b, out_path):
    a = pd.read_csv(path_a)[["filename", "FaceOcclusion"]].rename(columns={"FaceOcclusion": "pA"})
    b = pd.read_csv(path_b)[["filename", "FaceOcclusion"]].rename(columns={"FaceOcclusion": "pB"})
    m = a.merge(b, on="filename", how="inner")
    assert len(m) == len(a) == len(b), f"desalignement des filenames : {len(m)} vs {len(a)}/{len(b)}"

    blended = np.clip(W_A * m["pA"] + W_B * m["pB"], 0.0, 1.0)
    out = m[["filename"]].copy()
    out["FaceOcclusion"] = blended
    out["gender"] = "x"   # colonne factice requise par la soumission
    out.to_csv(out_path, index=False)
    print(f"result : {out_path}  ({len(out)} lignes, mean={blended.mean():.4f})")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python blend_final.py predictions_A.csv predictions_B.csv submission.csv")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3])
