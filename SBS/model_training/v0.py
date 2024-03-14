
# Defines number of epochs
n_epochs = 1000

for epoch in range(n_epochs):
    # Set model to TRAIN mode
    model.train()
    # Step 1- Computes model's predicted output - forward parss
    yhat = model(x_train_tensor)
    # Step 2 = Computes the loss
    loss = loss_fn(yhat, y_train_tensor)
    # Step 3 - Computes gradients for both "b" and "w" paramters
    loss.backward()
    # Step 4 - Updats parameters using gradients and the learning rate
    optimizer.step()
    optimizer.zero_grad()
