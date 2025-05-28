def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = torch.sigmoid(torch.tensor(predictions)).numpy()

    # Convert to binary predictions
    binary_predictions = (predictions > 0.5).astype(int)

    # Calculate metrics
    auc_scores = []
    f1_scores = []

    for i in range(labels.shape[1]):
        if len(np.unique(labels[:, i])) > 1:  # Check if both classes exist
            auc = roc_auc_score(labels[:, i], predictions[:, i])
            auc_scores.append(auc)
            f1 = f1_score(labels[:, i], binary_predictions[:, i])
            f1_scores.append(f1)

    return {
        'auc': np.mean(auc_scores),
        'f1': np.mean(f1_scores)
    }