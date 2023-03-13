import pandas as pd

for split_type in ["train", "test"]:
    with open(split_type + "_NLI_B.csv", "r") as f:
        lines = f.readlines()

    target_text = []
    input_text = []
    for i in range(len(lines)//25):
        opinions = []
        for j in range(25):
            line = lines[(i * 25) + j].split("\t")
            if int(line[1]) == 1:
                asp_pol = line[2].split(" - ")
                if asp_pol[0] != "none":
                    # opinions.append(f"[{asp_pol[0]}] opinion on [{asp_pol[1]}]")
                    opinions.append(f"{asp_pol[1]} ~ {asp_pol[0]}")

        input_text.append(line[-1].strip().strip("\n").strip())
        # target_text.append(f"The review expressed {', '.join(opinions)}")
        target_text.append(' ~~ '.join(opinions))

    assert len(input_text) == len(target_text)
    prefix = ["ASD"] * len(target_text)
    df = pd.DataFrame(prefix, columns=["prefix"])
    df["input_text"] = input_text
    df["target_text"] = target_text
    df.to_csv(f"{split_type}_ASD_phrase.csv", header=True, index=False)
