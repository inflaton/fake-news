from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
import matplotlib.pyplot as plt
import evaluate
import numpy as np
import pandas as pd
import time
from datasets import load_dataset, concatenate_datasets

datasets = load_dataset(
    "csv",
    data_files={
        "train": [
            "dataset/train_data_1.csv",
            "dataset/train_data_2.csv",
            "dataset/train_data_3.csv",
            "dataset/train_data_4.csv",
        ],
        "test": "dataset/test_data.csv",
        "rewritten_train": [
            "dataset/rewritten_train_data_1.csv",
            "dataset/rewritten_train_data_2.csv",
            "dataset/rewritten_train_data_3.csv",
            "dataset/rewritten_train_data_4.csv",
        ],
        "rewritten_test": "dataset/rewritten_test_data.csv",
    },
)

def save_results(
    model,
    perf_original_model_val_data,
    perf_original_model_val_data_rewritten,
    perf_combined_model_val_data,
    perf_combined_model_val_data_rewritten,
    perf_rewritten_model_val_data=None,
    perf_rewritten_model_val_data_rewritten=None,
):
    perf_original_model_val_data = pd.DataFrame(perf_original_model_val_data, index=[0])
    perf_original_model_val_data_rewritten = pd.DataFrame(
        perf_original_model_val_data_rewritten, index=[0]
    )
    perf_combined_model_val_data = pd.DataFrame(perf_combined_model_val_data, index=[0])
    perf_combined_model_val_data_rewritten = pd.DataFrame(
        perf_combined_model_val_data_rewritten, index=[0]
    )
    df = pd.concat(
        [
            perf_original_model_val_data,
            perf_original_model_val_data_rewritten,
            perf_combined_model_val_data,
            perf_combined_model_val_data_rewritten,
        ]
    )
    if perf_rewritten_model_val_data is not None:
        perf_rewritten_model_val_data = pd.DataFrame(
            perf_rewritten_model_val_data, index=[0]
        )
        perf_rewritten_model_val_data_rewritten = pd.DataFrame(
            perf_rewritten_model_val_data_rewritten, index=[0]
        )
        df = pd.concat(
            [
                df,
                perf_rewritten_model_val_data,
                perf_rewritten_model_val_data_rewritten,
            ]
        )

    df["model"] = model
    df["train-data"] = [
        "original",
        "original",
        "combined",
        "combined",
        "rewritten",
        "rewritten",
    ][: len(df)]
    df["test-data"] = [
        "original",
        "rewritten",
        "original",
        "rewritten",
        "original",
        "rewritten",
    ][: len(df)]

    df2 = df[
        [
            "model",
            "train-data",
            "test-data",
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "throughput",
        ]
    ].reset_index(drop=True)
    df2.to_csv(f"results/{model}_metrics.csv", index=False)
    print(f"Results saved to results/{model}_metrics.csv")


def plot_results(
    model,
    perf_original_model_val_data,
    perf_original_model_val_data_rewritten,
    perf_combined_model_val_data,
    perf_combined_model_val_data_rewritten,
    perf_rewritten_model_val_data=None,
    perf_rewritten_model_val_data_rewritten=None,
    ylim=1.13
):
    """
    Plot performance metrics for three models (trained with original, LLM-rewritten, or combined data)
    on two test sets (Original Test Data and LLM-rewritten Test Data) in one row and three columns.
    Original Test Data results are displayed using the left y-axis and LLM-rewritten Test Data results on the right y-axis.
    Separate legends are shown: the left legend (Original Test Data) at the top left,
    and the right legend (LLM-rewritten Test Data) at the top right.
    """

    # Helper function to plot each subplot
    def plot_subplot(ax, data_left, data_right, title, left_color, right_color):
        data_left.pop("throughput", None)
        data_right.pop("throughput", None)

        categories = list(data_left.keys())
        x = np.arange(len(categories))
        bar_width = 0.4

        # Plot Original Test Data on primary axis (left)
        bars_left = ax.bar(
            x - bar_width / 2,
            list(data_left.values()),
            width=bar_width,
            color=left_color,
        )
        ax.set_ylim(0, ylim)
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        for i, val in enumerate(data_left.values()):
            ax.text(
                x[i] - bar_width / 2,
                val + 0.01,
                f"{val*100:.2f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        # Add left legend at top left for Original Test Data
        ax.legend([bars_left], ["Original Test Data"], loc="upper left", fontsize=8)

        # Create twin axis for LLM-rewritten Test Data
        ax2 = ax.twinx()
        bars_right = ax2.bar(
            x + bar_width / 2,
            list(data_right.values()),
            width=bar_width,
            color=right_color,
        )
        ax2.set_ylim(0, ylim)
        for i, val in enumerate(data_right.values()):
            ax2.text(
                x[i] + bar_width / 2,
                val + 0.01,
                f"{val*100:.2f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        # Add right legend at top right for LLM-rewritten Test Data
        ax2.legend(
            [bars_right], ["LLM-rewritten Test Data"], loc="upper right", fontsize=8
        )
        ax.set_title(title, fontsize=10)

    # Create 1 row, 3 columns of subplots
    fig, axes = plt.subplots(1, 3 if perf_rewritten_model_val_data else 2, figsize=(15, 5))
    fig.suptitle(f"{model} Model Performance Comparison", fontsize=16)

    # Subplot 1: Model Trained with Original Data
    plot_subplot(
        axes[0],
        perf_original_model_val_data,
        perf_original_model_val_data_rewritten,
        "Model Trained with Original Data",
        left_color="skyblue",
        right_color="steelblue",
    )

    i = 1
    if perf_rewritten_model_val_data is not None:
        # Subplot 2: Model Trained with LLM-rewritten Test Data
        plot_subplot(
            axes[i],
            perf_rewritten_model_val_data,
            perf_rewritten_model_val_data_rewritten,
            "Model Trained with LLM-rewritten Data",
            left_color="lightgreen",
            right_color="seagreen",
        )
        i += 1

    # Subplot 3: Model Trained with Combined Data
    plot_subplot(
        axes[i],
        perf_combined_model_val_data,
        perf_combined_model_val_data_rewritten,
        "Model Trained with Combined Data",
        left_color="salmon",
        right_color="tomato",
    )

    plt.tight_layout()
    plt.show()

def evaluate_fine_tuned_llm(model_name, eval_dataset=None, config="original"):
    # Load metrics
    accuracy_metric = evaluate.load("accuracy")
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")
    f1_metric = evaluate.load("f1")

    # Measure the prediction time to compute throughput
    start_time = time.time()

    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Tokenize the dataset
    def tokenize_function(example):
        return tokenizer(
            example["processed_full_content"],
            padding="max_length",
            truncation=True,
        )

    if eval_dataset:
        eval_dataset = eval_dataset.map(tokenize_function, batched=True)
    else:
        def get_datasets(config="original"):
            if config == "rewritten":
                train_dataset = tokenized_datasets["rewritten_train"]
                eval_dataset = tokenized_datasets["rewritten_test"]
            elif config == "original":
                train_dataset = tokenized_datasets["train"]
                eval_dataset = tokenized_datasets["test"]
            else:
                train_dataset = concatenate_datasets(
                    [tokenized_datasets["train"], tokenized_datasets["rewritten_train"]]
                )
                eval_dataset = concatenate_datasets(
                    [tokenized_datasets["test"], tokenized_datasets["rewritten_test"]]
                )
            return train_dataset, eval_dataset

        tokenized_datasets = datasets.map(tokenize_function, batched=True)
        _, eval_dataset = get_datasets(config=config)    

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        # Otherwise, compute and return metrics.
        accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
        precision = precision_metric.compute(predictions=predictions, references=labels, average="binary")
        recall = recall_metric.compute(predictions=predictions, references=labels, average="binary")
        f1 = f1_metric.compute(predictions=predictions, references=labels, average="binary")
        
        return {
            "accuracy": accuracy["accuracy"],
            "precision": precision["precision"],
            "recall": recall["recall"],
            "f1_score": f1["f1"],
        }

    training_args = TrainingArguments(
        output_dir="./results",
        per_device_eval_batch_size=16,
        do_train=False,
        do_eval=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    # Check if the eval_dataset has a "label" column
    if "label" in eval_dataset.column_names:
        # When labels exist, compute metrics
        eval_results = trainer.evaluate()
        end_time = time.time()
        
        prediction_time = end_time - start_time
        throughput = len(eval_dataset) / prediction_time if prediction_time > 0 else 0

        print("Evaluation Results:", eval_results)
        result = {
            "accuracy": eval_results["eval_accuracy"],
            "precision": eval_results["eval_precision"],
            "recall": eval_results["eval_recall"],
            "f1_score": eval_results["eval_f1_score"],
            "throughput": throughput,
        }
        print("\n🏆 Final Evaluation Results:")
        for key, value in result.items():
            print(f"🔹 {key.capitalize()}: {value:.4f}")
    else:
        # When no labels exist, use predict() to get predictions
        pred_output = trainer.predict(eval_dataset)
        predictions = np.argmax(pred_output.predictions, axis=-1)
        result = predictions.tolist()
        print("\n🔹 #Predictions:")
        print(len(result))

    return result