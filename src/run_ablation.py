import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.train_gat import train

def load_config(path):

    with open(path,"r",encoding="utf-8") as f:

        return json.load(f)
    
def run_once(config,value):

    parameter=config["parameter"]

    fixed=config["fixed"]

    args=argparse.Namespace(

        arrays_path=Path("data/processed/graph_arrays.npz"),

        model_dir=Path("models")/f"{parameter}_{value}",

        results_dir=Path("results")/f"{parameter}_{value}",

        epochs=fixed.get("epochs",50),

        hidden_channels=fixed.get("hidden_channels",64),

        heads=fixed.get("heads",4),

        dropout=fixed.get("dropout",0.2),

        lr=0.005,

        weight_decay=0.0005,

        random_state=42,

        cpu=False,

        early_stopping=True,

        patience=8,

        min_delta=0.0005,

        selection_metric="val_tuned_f1",

        message_passing_edges="all",
    )

    setattr(args,parameter,value)

    result=train(args)

    test=result["final_metrics"]["test"]

    return{

        parameter:value,

        "Accuracy":test["accuracy"],

        "Precision":test["precision"],

        "Recall":test["recall"],

        "F1":test["f1"]

    }

def run(config_path):

    cfg=load_config(config_path)

    rows=[]

    for value in cfg["values"]:

        print("="*60)

        print(cfg["parameter"],"=",value)

        print("="*60)

        rows.append(

            run_once(cfg,value)

        )

    df=pd.DataFrame(rows)

    Path("results").mkdir(exist_ok=True)

    csv_path=Path("results")/f"ablation_{cfg['parameter']}.csv"

    df.to_csv(csv_path,index=False)

    plt.figure(figsize=(7,5))

    plt.plot(

        df[cfg["parameter"]],

        df["F1"],

        marker="o"

    )

    plt.xlabel(cfg["parameter"])

    plt.ylabel("F1-score")

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(

        Path("results")/

        f"ablation_{cfg['parameter']}.png",

        dpi=300

    )

    print(df)

if __name__=="__main__":

    parser=argparse.ArgumentParser()

    parser.add_argument(

        "--config",

        required=True,

        type=Path

    )

    args=parser.parse_args()

    run(args.config)