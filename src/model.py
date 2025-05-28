import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

class ToxicCommentDetector:
    def __init__(self):
        # Initialize empty dictionaries for models and tokenizers
        self.models = {}
        self.tokenizers = {}
        self.label_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

        self.model_configs = {
            'DistilBERT': {
                'name': 'distilbert-base-uncased',
                'max_len': 128,
                'batch_size': 16,
                'epochs': 3,
                'lr': 2e-5
            },
            'RoBERTa': {
                'name': 'roberta-base',
                'max_len': 128,
                'batch_size': 8,
                'epochs': 3,
                'lr': 1e-5
            },
            'ALBERT': {
                'name': 'albert-base-v2',
                'max_len': 128,
                'batch_size': 16,
                'epochs': 3,
                'lr': 3e-5
            }
        }

    def load_models(self):
        """Load pre-trained models and tokenizers."""
        for model_name, config in self.model_configs.items():
            print(f"Loading {model_name}...")
            self.models[model_name] = AutoModelForSequenceClassification.from_pretrained(config['name'], num_labels=len(self.label_columns))
            self.tokenizers[model_name] = AutoTokenizer.from_pretrained(config['name'])
        print("✅ Models and tokenizers loaded successfully!")

    def load_and_preprocess_data(self, file_path):
        """Load and preprocess the dataset."""
        print(f"📊 Loading dataset from {file_path}...")
        df = pd.read_csv(file_path)
        print(f"✅ Dataset loaded successfully! First few rows:\n{df.head()}")

        # Preprocess the data
        from preprocess import preprocess_data
        df = preprocess_data(df)
        print("✅ Data preprocessing completed!")
        return df

    def train_model(self, model_name, X_train, X_val, y_train, y_val):
        print(f"\n🚀 Training {model_name}...")

        config = self.model_configs[model_name]

        tokenizer = AutoTokenizer.from_pretrained(config['name'])
        model = AutoModelForSequenceClassification.from_pretrained(
            config['name'],
            num_labels=len(self.label_columns),
            problem_type="multi_label_classification"
        )

        train_dataset = ToxicDataset(X_train, y_train, tokenizer, config['max_len'], model_name)
        val_dataset = ToxicDataset(X_val, y_val, tokenizer, config['max_len'], model_name)

        training_args = TrainingArguments(
            output_dir=f'./results_{model_name.lower()}',
            num_train_epochs=config['epochs'],
            per_device_train_batch_size=config['batch_size'],
            per_device_eval_batch_size=config['batch_size'],
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=f'./logs_{model_name.lower()}',
            logging_steps=100,
            eval_strategy="steps",
            eval_steps=500,
            save_strategy="steps",
            save_steps=500,
            load_best_model_at_end=True,
            metric_for_best_model="auc",
            greater_is_better=True,
            learning_rate=config['lr'],
            adam_epsilon=1e-8,
            max_grad_norm=1.0,
            fp16=True if torch.cuda.is_available() else False,
            dataloader_num_workers=0,
            save_total_limit=1,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )

        trainer.train()

        self.models[model_name] = model
        self.tokenizers[model_name] = tokenizer

        eval_results = trainer.evaluate()
        print(f"✅ {model_name} - Validation AUC: {eval_results['eval_auc']:.4f}, F1: {eval_results['eval_f1']:.4f}")

        return eval_results

    def predict(self, text, model_name):
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained yet!")

        model = self.models[model_name]
        tokenizer = self.tokenizers[model_name]

        device = next(model.parameters()).device

        tokenizer_kwargs = {
            'text': text,
            'add_special_tokens': True,
            'max_length': 128,
            'padding': 'max_length',
            'truncation': True,
            'return_attention_mask': True,
            'return_tensors': 'pt'
        }

        if 'distilbert' not in model_name.lower():
            tokenizer_kwargs['return_token_type_ids'] = True

        inputs = tokenizer.encode_plus(**tokenizer_kwargs)

        for key in inputs:
            inputs[key] = inputs[key].to(device)

        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.sigmoid(outputs.logits).cpu().numpy()[0]

        results = {}
        for i, label in enumerate(self.label_columns):
            results[label] = float(predictions[i])

        return results

    def evaluate_all_models(self, X_test, y_test):
        results = {}

        for model_name in self.models.keys():
            print(f"\n🔍 Evaluating {model_name} on test set...")

            model = self.models[model_name]
            tokenizer = self.tokenizers[model_name]

            test_dataset = ToxicDataset(X_test, y_test, tokenizer, 128, model_name)

            trainer = Trainer(
                model=model,
                compute_metrics=compute_metrics,
            )

            eval_results = trainer.evaluate(test_dataset)
            results[model_name] = {
                'auc': eval_results['eval_auc'],
                'f1': eval_results['eval_f1']
            }

            print(f"📊 {model_name} - Test AUC: {eval_results['eval_auc']:.4f}, F1: {eval_results['eval_f1']:.4f}")

        return results