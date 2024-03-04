
# Defines number of epochs
n_epochs = 1000

losses = []

# For each epoch...
for epoch in range(n_epochs):
    # inner loop
    mini_batch_losses = []
    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        # Performs one train step and returns the 
        # corresponding loss for this mini-batch
        mini_batch_loss = train_step_fn(x_batch, y_batch)
        mini_batch_losses.append(mini_batch_loss)
    # Computes average loss over all mini-batches
    # That's the epoch loss
    loss = np.mean(mini_batch_loss)
    losses.append(loss)
