def masked_mse_loss(preds, targets):
    """
    Compute the Mean Squared Error (MSE) loss while ignoring positions where target values are zero.

    This is useful in sequence prediction tasks where certain target positions are padded with zeros 
    (e.g., for variable-length outputs or multi-step forecasting near the end of a series).

    Parameters
    ----------
    preds : torch.Tensor
        Predicted values of shape (batch_size, sequence_length).
    targets : torch.Tensor
        Ground truth values of shape (batch_size, sequence_length). 
        Positions with value 0 are treated as invalid and masked out.

    Returns
    -------
    torch.Tensor
        A scalar tensor representing the mean masked MSE loss across the batch.

    """
    # Define the mask. Mask values where target is 0
    mask = (targets != 0).float()
    
    # Calculate the squared error
    se = (preds - targets) ** 2
    
    # Apply the mask to the squared errors
    masked_se = se * mask
    
    # Count the number of valid (non-zero) target values per sample
    valid_counts = mask.sum(dim=1)  

    # Average masked squared error for each sample
    loss_per_sample = masked_se.sum(dim=1) / (valid_counts + 1e-8)

    # Return the mean loss across the batch
    return loss_per_sample.mean()